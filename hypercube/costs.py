"""Point-in-time transaction-cost, borrow, and capacity rules for phase P8."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hypercube.config import HypercubeConfig


class CostModelError(ValueError):
    """Raised when a P8 execution input violates the frozen cost contract."""


def execution_liquidity(
    frame: pd.DataFrame,
    config: HypercubeConfig,
) -> pd.DataFrame:
    """Compute dated spread, liquidity, capacity, and short-borrow fields."""

    required = {
        "prc",
        "vol",
        "bid",
        "ask",
        "market_cap_millions",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise CostModelError(f"Missing execution-liquidity fields: {missing}")
    result = frame.copy()
    price = pd.to_numeric(result["prc"], errors="coerce").abs()
    volume = pd.to_numeric(result["vol"], errors="coerce")
    bid = pd.to_numeric(result["bid"], errors="coerce")
    ask = pd.to_numeric(result["ask"], errors="coerce")
    market_cap = pd.to_numeric(result["market_cap_millions"], errors="coerce")
    valid_quote = bid.gt(0.0) & ask.gt(bid)
    quoted = pd.Series(np.nan, index=result.index, dtype=float)
    quoted.loc[valid_quote] = (
        (ask.loc[valid_quote] - bid.loc[valid_quote])
        / (ask.loc[valid_quote] + bid.loc[valid_quote])
    )
    fallback = config.costs.fallback_half_spread_bps / 10_000.0
    result["half_spread"] = quoted.fillna(fallback).clip(
        lower=0.0,
        upper=config.costs.max_half_spread,
    )
    result["spread_source"] = np.where(
        valid_quote,
        "quoted",
        "fallback",
    )
    monthly_dollar_volume = (
        price * volume * config.costs.volume_unit_multiplier
    )
    result["monthly_dollar_volume"] = monthly_dollar_volume
    result["adv_millions"] = (
        monthly_dollar_volume
        / config.costs.trading_days_per_month
        / 1_000_000.0
    )
    adv_capacity = (
        result["adv_millions"]
        * 1_000_000.0
        * config.costs.max_adv_participation
    )
    market_cap_capacity = (
        market_cap
        * 1_000_000.0
        * config.costs.max_market_cap_fraction
    )
    capacity_dollars = np.minimum(adv_capacity, market_cap_capacity)
    notional = config.costs.portfolio_notional_millions_per_leg * 1_000_000.0
    result["capacity_dollars"] = capacity_dollars.clip(lower=0.0)
    result["capacity_weight_limit"] = np.minimum(
        config.costs.max_position_weight,
        result["capacity_dollars"] / notional,
    )
    valid_capacity = (
        price.gt(0.0)
        & volume.gt(0.0)
        & market_cap.gt(0.0)
        & result["capacity_weight_limit"].notna()
    )
    result.loc[~valid_capacity, "capacity_weight_limit"] = 0.0
    result["short_available"] = (
        market_cap.ge(config.costs.minimum_short_market_cap_millions)
        & result["adv_millions"].ge(config.costs.minimum_short_adv_millions)
        & valid_capacity
    )
    hard_borrow = (
        market_cap.lt(config.costs.hard_borrow_market_cap_millions)
        | result["adv_millions"].lt(config.costs.hard_borrow_adv_millions)
    )
    result["borrow_annual_bps"] = np.where(
        hard_borrow,
        config.costs.hard_borrow_annual_bps,
        config.costs.borrow_annual_bps,
    )
    result.loc[~result["short_available"], "borrow_annual_bps"] = np.nan
    numeric = result[
        [
            "half_spread",
            "monthly_dollar_volume",
            "adv_millions",
            "capacity_dollars",
            "capacity_weight_limit",
        ]
    ].to_numpy(float)
    if np.isinf(numeric).any():
        raise CostModelError("Execution-liquidity fields contain infinities.")
    return result


def cap_assignment_weights(
    assignments: pd.DataFrame,
    config: HypercubeConfig,
) -> pd.DataFrame:
    """Apply frozen formation-time capacity and short-availability constraints."""

    required = {
        "weight",
        "leg",
        "capacity_weight_limit",
        "short_available",
    }
    missing = sorted(required - set(assignments.columns))
    if missing:
        raise CostModelError(f"Missing assignment-capacity fields: {missing}")
    result = assignments.copy()
    weight = pd.to_numeric(result["weight"], errors="coerce")
    limit = pd.to_numeric(result["capacity_weight_limit"], errors="coerce")
    if weight.isna().any() or weight.lt(0.0).any():
        raise CostModelError("P8 assignment weights are missing or negative.")
    result["capacity_weight"] = np.minimum(weight, limit.clip(lower=0.0))
    short_blocked = result["leg"].eq("short") & ~result["short_available"].astype(bool)
    result.loc[short_blocked, "capacity_weight"] = 0.0
    result["capacity_fill_ratio"] = np.where(
        weight.gt(0.0),
        result["capacity_weight"] / weight,
        0.0,
    )
    result["capacity_exclusion_reason"] = np.select(
        [
            short_blocked,
            limit.le(0.0),
            result["capacity_weight"].lt(weight),
        ],
        [
            "short_unavailable",
            "missing_or_zero_capacity",
            "capacity_capped",
        ],
        default="filled",
    )
    return result


def one_way_transaction_cost(
    absolute_weight_change: pd.Series,
    half_spread: pd.Series,
    *,
    spread_multiplier: float,
    fixed_slippage_bps: float,
) -> pd.Series:
    """Return one-way transaction cost as a fraction of portfolio capital."""

    change = pd.to_numeric(absolute_weight_change, errors="coerce")
    spread = pd.to_numeric(half_spread, errors="coerce")
    if change.isna().any() or spread.isna().any():
        raise CostModelError("Transaction-cost inputs contain missing values.")
    if change.lt(0.0).any() or spread.lt(0.0).any():
        raise CostModelError("Transaction-cost inputs must be nonnegative.")
    rate = spread_multiplier * spread + fixed_slippage_bps / 10_000.0
    return change * rate


def monthly_borrow_cost(
    signed_weight: pd.Series,
    annual_borrow_bps: pd.Series,
) -> pd.Series:
    """Return monthly borrow drag for eligible short positions."""

    weight = pd.to_numeric(signed_weight, errors="coerce")
    borrow = pd.to_numeric(annual_borrow_bps, errors="coerce").fillna(0.0)
    short_notional = (-weight).clip(lower=0.0)
    return short_notional * borrow / 10_000.0 / 12.0
