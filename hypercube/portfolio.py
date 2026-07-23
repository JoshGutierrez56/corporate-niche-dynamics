"""Cost-aware, capacity-constrained portfolio evaluation for phase P8."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import statsmodels.api as sm

from hypercube.config import HypercubeConfig
from hypercube.costs import (
    CostModelError,
    cap_assignment_weights,
    execution_liquidity,
    monthly_borrow_cost,
    one_way_transaction_cost,
)
from hypercube.data import Scenario, atomic_write_json, sha256_file
from hypercube.returns import scenario_p7_dir, validate_p7_directory
from hypercube.universe import scenario_processed_dir


class PortfolioCostError(ValueError):
    """Raised when a P8 portfolio or audit contract fails."""


P8_VERSION = "hypercube-cost-aware-portfolios-v1"
STRATEGY_COLUMNS = [
    "horizon_months",
    "signal",
    "signal_label",
    "weighting",
    "industry_neutral",
]


def scenario_p8_dir(config: HypercubeConfig, scenario: Scenario) -> Path:
    """Resolve one scenario's P8 output directory."""

    return scenario_processed_dir(config, scenario) / "p8"


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.to_parquet(name, index=False, compression="zstd")
        os.replace(name, path)
    except Exception:
        Path(name).unlink(missing_ok=True)
        raise


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.to_csv(name, index=False, lineterminator="\n")
        os.replace(name, path)
    except Exception:
        Path(name).unlink(missing_ok=True)
        raise


def _file_record(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix == ".parquet":
        record["rows"] = pq.ParquetFile(path).metadata.num_rows
    elif path.suffix == ".csv":
        record["rows"] = max(
            0, sum(1 for _ in path.open("r", encoding="utf-8")) - 1
        )
    return record


def _raw_liquidity(raw_dir: Path, config: HypercubeConfig) -> pd.DataFrame:
    """Load raw monthly execution proxies without retrospective filling."""

    columns = ["permno", "date", "prc", "shrout", "vol", "bid", "ask"]
    raw = pd.read_parquet(raw_dir / "crsp_monthly.parquet", columns=columns)
    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw["market_cap_millions"] = (
        pd.to_numeric(raw["prc"], errors="coerce").abs()
        * pd.to_numeric(raw["shrout"], errors="coerce")
        / 1000.0
    )
    if raw.duplicated(["permno", "date"]).any():
        raise PortfolioCostError("Raw monthly liquidity keys are duplicated.")
    return execution_liquidity(raw, config)


def prepare_cost_assignments(
    p7_dir: Path,
    raw_dir: Path,
    config: HypercubeConfig,
) -> pd.DataFrame:
    """Attach formation-date execution inputs and freeze capacity weights."""

    assignments = pd.read_parquet(p7_dir / "portfolio_assignments.parquet")
    assignments = assignments.loc[
        assignments["horizon_months"].eq(config.costs.primary_horizon_months)
    ].copy()
    events = pd.read_parquet(
        p7_dir / "return_events.parquet",
        columns=["event_id", "permno", "feature_date"],
    )
    events["feature_date"] = pd.to_datetime(events["feature_date"], errors="coerce")
    liquidity = _raw_liquidity(raw_dir, config)
    formation = liquidity.rename(columns={"date": "feature_date"}).copy()
    keep = [
        "permno",
        "feature_date",
        "half_spread",
        "spread_source",
        "monthly_dollar_volume",
        "adv_millions",
        "capacity_dollars",
        "capacity_weight_limit",
        "short_available",
        "borrow_annual_bps",
        "market_cap_millions",
    ]
    joined = assignments.merge(
        events,
        on=["event_id", "feature_date"],
        how="left",
        validate="many_to_one",
    ).merge(
        formation[keep],
        on=["permno", "feature_date"],
        how="left",
        validate="many_to_one",
    )
    if joined["permno"].isna().any():
        raise PortfolioCostError("P8 could not recover assignment PERMNOs.")
    if joined["capacity_weight_limit"].isna().any():
        raise PortfolioCostError("P8 lacks formation-date execution data.")
    joined["liquidity_date"] = joined["feature_date"]
    joined = cap_assignment_weights(joined, config)
    if (joined["liquidity_date"] > joined["feature_date"]).any():
        raise PortfolioCostError("P8 formation capacity uses future liquidity.")
    return joined.sort_values(
        [*STRATEGY_COLUMNS, "feature_date", "leg", "gvkey"],
        kind="stable",
    ).reset_index(drop=True)


def _lagged_execution_liquidity(
    raw_dir: Path,
    config: HypercubeConfig,
) -> pd.DataFrame:
    """Map each dated liquidity observation to the next eligible return month."""

    liquidity = _raw_liquidity(raw_dir, config)
    liquidity = liquidity.rename(columns={"date": "liquidity_date"})
    liquidity["holding_date"] = (
        liquidity["liquidity_date"] + pd.offsets.MonthEnd(1)
    )
    keep = [
        "permno",
        "holding_date",
        "liquidity_date",
        "half_spread",
        "spread_source",
        "adv_millions",
        "capacity_weight_limit",
        "short_available",
        "borrow_annual_bps",
        "market_cap_millions",
    ]
    return liquidity[keep]


def build_executed_positions(
    cost_assignments: pd.DataFrame,
    paths: pd.DataFrame,
    lagged_liquidity: pd.DataFrame,
    config: HypercubeConfig,
    *,
    delay_months: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build capacity-constrained security-month weights after a safe delay."""

    horizon = config.costs.primary_horizon_months
    low = delay_months + 1
    high = delay_months + horizon
    selected_paths = paths.loc[
        paths["holding_offset"].between(low, high),
        [
            "event_id",
            "holding_offset",
            "holding_date",
            "return_used",
            "path_valid_through_month",
            "cash_after_delist",
            "valid_delist_event",
            "rf",
        ],
    ].copy()
    joined = cost_assignments.merge(
        selected_paths,
        on="event_id",
        how="left",
        validate="many_to_many",
    )
    if joined["holding_date"].isna().any():
        raise PortfolioCostError("P8 delayed path is incomplete.")
    if (
        joined["return_used"].isna().any()
        or (~joined["path_valid_through_month"].astype(bool)).any()
    ):
        raise PortfolioCostError("P8 selected an invalid P7 return path.")
    calendar_keys = [*STRATEGY_COLUMNS, "holding_date"]
    calendars = (
        joined.groupby(calendar_keys, observed=True)
        .agg(
            active_cohorts=("feature_date", "nunique"),
            rf=("rf", "first"),
        )
        .reset_index()
    )
    security = joined.loc[~joined["cash_after_delist"].astype(bool)].copy()
    security["signed_target_weight"] = np.where(
        security["leg"].eq("long"),
        security["capacity_weight"],
        -security["capacity_weight"],
    )
    security = security.merge(
        calendars[[*calendar_keys, "active_cohorts"]],
        on=calendar_keys,
        how="left",
        validate="many_to_one",
    )
    security["signed_target_weight"] /= security["active_cohorts"]
    group_keys = [*STRATEGY_COLUMNS, "holding_date", "permno", "gvkey"]
    consistency = security.groupby(group_keys, observed=True).agg(
        return_values=("return_used", "nunique"),
        rf_values=("rf", "nunique"),
    )
    if consistency["return_values"].gt(1).any() or consistency["rf_values"].gt(1).any():
        raise PortfolioCostError("One security-month has inconsistent return data.")
    positions = (
        security.groupby(group_keys, observed=True)
        .agg(
            target_signed_weight=("signed_target_weight", "sum"),
            return_used=("return_used", "first"),
            rf=("rf", "first"),
            forced_delist_this_month=("valid_delist_event", "max"),
        )
        .reset_index()
    )
    positions = positions.merge(
        lagged_liquidity,
        on=["permno", "holding_date"],
        how="left",
        validate="many_to_one",
    )
    fallback = config.costs.fallback_half_spread_bps / 10_000.0
    positions["half_spread"] = positions["half_spread"].fillna(fallback).clip(
        lower=0.0,
        upper=config.costs.max_half_spread,
    )
    positions["spread_source"] = positions["spread_source"].fillna(
        "fallback_missing_lag"
    )
    positions["capacity_weight_limit"] = positions[
        "capacity_weight_limit"
    ].fillna(0.0)
    positions["short_available"] = positions["short_available"].fillna(False)
    positions["borrow_annual_bps"] = positions["borrow_annual_bps"].fillna(0.0)
    positions["liquidity_date"] = pd.to_datetime(
        positions["liquidity_date"], errors="coerce"
    )
    if positions["liquidity_date"].isna().any():
        positions["liquidity_date"] = positions["holding_date"] - pd.offsets.MonthEnd(1)
    if (positions["liquidity_date"] >= positions["holding_date"]).any():
        raise PortfolioCostError("P8 uses liquidity unavailable before the return month.")
    absolute_target = positions["target_signed_weight"].abs()
    absolute_actual = np.minimum(
        absolute_target,
        positions["capacity_weight_limit"].clip(lower=0.0),
    )
    blocked_short = (
        positions["target_signed_weight"].lt(0.0)
        & ~positions["short_available"].astype(bool)
    )
    absolute_actual.loc[blocked_short] = 0.0
    positions["actual_signed_weight"] = (
        np.sign(positions["target_signed_weight"]) * absolute_actual
    )
    positions["dynamic_capacity_fill_ratio"] = np.where(
        absolute_target.gt(0.0),
        absolute_actual / absolute_target,
        0.0,
    )
    positions["delay_months"] = delay_months
    calendars["delay_months"] = delay_months
    return (
        positions.sort_values(
            [*STRATEGY_COLUMNS, "holding_date", "gvkey"], kind="stable"
        ).reset_index(drop=True),
        calendars.sort_values(
            [*STRATEGY_COLUMNS, "holding_date"], kind="stable"
        ).reset_index(drop=True),
    )


def evaluate_cost_aware_months(
    positions: pd.DataFrame,
    calendars: pd.DataFrame,
    factors: pd.DataFrame,
    config: HypercubeConfig,
) -> pd.DataFrame:
    """Apply one-way spread/slippage and monthly borrow costs."""

    factors = factors.copy()
    factors["date"] = pd.to_datetime(factors["date"], errors="coerce")
    factor_columns = ["date", *config.returns.factor_columns]
    calendar = calendars.merge(
        factors[factor_columns],
        left_on="holding_date",
        right_on="date",
        how="left",
        validate="many_to_one",
    ).drop(columns="date")
    if calendar[list(config.returns.factor_columns)].isna().any(axis=None):
        raise PortfolioCostError("P8 lacks factor coverage.")
    rows: list[dict[str, Any]] = []
    grouping = [*STRATEGY_COLUMNS, "delay_months"]
    for keys, strategy_calendar in calendar.groupby(
        grouping, observed=True, sort=True
    ):
        mask = pd.Series(True, index=positions.index)
        for column, value in zip(grouping, keys, strict=True):
            mask &= positions[column].eq(value)
        strategy_positions = positions.loc[mask]
        previous: dict[int, float] = {}
        previous_spreads: dict[int, float] = {}
        previous_forced: set[int] = set()
        for _, month_row in strategy_calendar.sort_values("holding_date").iterrows():
            month = month_row["holding_date"]
            current_frame = strategy_positions.loc[
                strategy_positions["holding_date"].eq(month)
            ].copy()
            current = dict(
                zip(
                    current_frame["permno"].astype(int),
                    current_frame["actual_signed_weight"].astype(float),
                )
            )
            current_spreads = dict(
                zip(
                    current_frame["permno"].astype(int),
                    current_frame["half_spread"].astype(float),
                )
            )
            current_forced = set(
                current_frame.loc[
                    current_frame["forced_delist_this_month"].astype(bool),
                    "permno",
                ].astype(int)
            )
            gross_security = float(
                np.dot(
                    current_frame["actual_signed_weight"].to_numpy(float),
                    current_frame["return_used"].to_numpy(float),
                )
            )
            long_gross = float(
                current_frame["actual_signed_weight"].clip(lower=0.0).sum()
            )
            short_gross = float(
                (-current_frame["actual_signed_weight"]).clip(lower=0.0).sum()
            )
            rf = float(month_row["rf"])
            gross_return = gross_security + (short_gross - long_gross) * rf
            names = set(previous) | set(current)
            delta_rows = []
            for permno in names:
                change = abs(current.get(permno, 0.0) - previous.get(permno, 0.0))
                costed_change = (
                    0.0
                    if permno in previous_forced and permno not in current
                    else change
                )
                delta_rows.append(
                    {
                        "change": change,
                        "costed_change": costed_change,
                        "half_spread": current_spreads.get(
                            permno,
                            previous_spreads.get(
                                permno,
                                config.costs.fallback_half_spread_bps / 10_000.0,
                            ),
                        ),
                    }
                )
            deltas = pd.DataFrame(delta_rows)
            turnover = 0.5 * float(deltas["change"].sum()) if not deltas.empty else 0.0
            costed_turnover = (
                0.5 * float(deltas["costed_change"].sum())
                if not deltas.empty
                else 0.0
            )
            borrow_cost = float(
                monthly_borrow_cost(
                    current_frame["actual_signed_weight"],
                    current_frame["borrow_annual_bps"],
                ).sum()
            )
            base = dict(zip(grouping, keys, strict=True))
            base.update(
                {
                    "holding_date": month,
                    "gross_capacity_return": gross_return,
                    "borrow_cost": borrow_cost,
                    "turnover": turnover,
                    "costed_turnover": costed_turnover,
                    "long_gross_exposure": long_gross,
                    "short_gross_exposure": short_gross,
                    "capacity_fill_ratio": 0.5 * (long_gross + short_gross),
                    "active_cohorts": int(month_row["active_cohorts"]),
                }
            )
            for factor in ("rf", *config.returns.factor_columns):
                base[factor] = float(month_row[factor])
            for scenario in config.costs.scenarios:
                transaction_cost = (
                    float(
                        one_way_transaction_cost(
                            deltas["costed_change"],
                            deltas["half_spread"],
                            spread_multiplier=scenario.quoted_half_spread_multiplier,
                            fixed_slippage_bps=scenario.fixed_slippage_bps,
                        ).sum()
                    )
                    if not deltas.empty
                    else 0.0
                )
                row = dict(base)
                row.update(
                    {
                        "cost_scenario": scenario.name,
                        "transaction_cost": transaction_cost,
                        "net_return": gross_return
                        - transaction_cost
                        - borrow_cost,
                    }
                )
                rows.append(row)
            previous = current
            previous_spreads = current_spreads
            previous_forced = current_forced
    return pd.DataFrame(rows)


def _maximum_drawdown(values: np.ndarray) -> float:
    wealth = np.cumprod(1.0 + values)
    peak = np.maximum.accumulate(wealth)
    return float(np.min(wealth / peak - 1.0))


def summarize_cost_aware_months(
    monthly: pd.DataFrame,
    config: HypercubeConfig,
) -> pd.DataFrame:
    """Summarize gross and net returns without selecting a favorable case."""

    rows: list[dict[str, Any]] = []
    grouping = [*STRATEGY_COLUMNS, "delay_months", "cost_scenario"]
    for keys, group in monthly.groupby(grouping, observed=True, sort=True):
        group = group.sort_values("holding_date")
        gross = group["gross_capacity_return"].to_numpy(float)
        net = group["net_return"].to_numpy(float)
        gross_volatility = float(np.std(gross, ddof=1))
        net_volatility = float(np.std(net, ddof=1))
        design = sm.add_constant(
            group[list(config.returns.factor_columns)].to_numpy(float)
        )
        fitted = sm.OLS(net, design).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": config.costs.primary_horizon_months - 1},
        )
        row = dict(zip(grouping, keys, strict=True))
        row.update(
            {
                "months": int(len(group)),
                "first_month": group["holding_date"].min(),
                "last_month": group["holding_date"].max(),
                "mean_monthly_gross_capacity_return": float(np.mean(gross)),
                "annualized_gross_capacity_return": float(12.0 * np.mean(gross)),
                "annualized_gross_capacity_sharpe": (
                    float(math.sqrt(12.0) * np.mean(gross) / gross_volatility)
                    if gross_volatility > 0.0
                    else np.nan
                ),
                "mean_monthly_net_return": float(np.mean(net)),
                "annualized_net_return": float(12.0 * np.mean(net)),
                "annualized_net_sharpe": (
                    float(math.sqrt(12.0) * np.mean(net) / net_volatility)
                    if net_volatility > 0.0
                    else np.nan
                ),
                "net_maximum_drawdown": _maximum_drawdown(net),
                "net_hit_rate": float(np.mean(net > 0.0)),
                "average_turnover": float(group["turnover"].mean()),
                "average_costed_turnover": float(group["costed_turnover"].mean()),
                "annualized_transaction_cost": float(
                    12.0 * group["transaction_cost"].mean()
                ),
                "annualized_borrow_cost": float(
                    12.0 * group["borrow_cost"].mean()
                ),
                "average_capacity_fill_ratio": float(
                    group["capacity_fill_ratio"].mean()
                ),
                "average_long_gross_exposure": float(
                    group["long_gross_exposure"].mean()
                ),
                "average_short_gross_exposure": float(
                    group["short_gross_exposure"].mean()
                ),
                "net_factor_alpha_monthly": float(fitted.params[0]),
                "net_factor_alpha_hac_standard_error": float(fitted.bse[0]),
                "net_factor_alpha_t_stat": float(fitted.tvalues[0]),
                "net_factor_alpha_p_value": float(fitted.pvalues[0]),
                "net_factor_r_squared": float(fitted.rsquared),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def capacity_diagnostics(cost_assignments: pd.DataFrame) -> pd.DataFrame:
    """Report how formation-time capacity and borrow rules alter each strategy."""

    rows: list[dict[str, Any]] = []
    for keys, group in cost_assignments.groupby(
        STRATEGY_COLUMNS, observed=True, sort=True
    ):
        row = dict(zip(STRATEGY_COLUMNS, keys, strict=True))
        short = group["leg"].eq("short")
        row.update(
            {
                "assignment_rows": int(len(group)),
                "unique_events": int(group["event_id"].nunique()),
                "mean_capacity_fill_ratio": float(
                    group["capacity_fill_ratio"].mean()
                ),
                "median_capacity_fill_ratio": float(
                    group["capacity_fill_ratio"].median()
                ),
                "capacity_capped_rows": int(
                    group["capacity_weight"].lt(group["weight"]).sum()
                ),
                "short_assignment_rows": int(short.sum()),
                "short_unavailable_rows": int(
                    (short & ~group["short_available"].astype(bool)).sum()
                ),
                "median_half_spread": float(group["half_spread"].median()),
                "median_adv_millions": float(group["adv_millions"].median()),
                "median_market_cap_millions": float(
                    group["market_cap_millions"].median()
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def cost_waterfall(
    summary: pd.DataFrame,
    p7_monthly: pd.DataFrame,
    config: HypercubeConfig,
) -> pd.DataFrame:
    """Reconcile P7 gross, capacity, spread/slippage, borrow, and P8 net."""

    p7 = p7_monthly.loc[
        p7_monthly["horizon_months"].eq(config.costs.primary_horizon_months)
    ].copy()
    p7_means = (
        p7.groupby(
            ["horizon_months", "signal", "signal_label", "weighting", "industry_neutral"],
            observed=True,
        )["long_short_return"]
        .mean()
        .mul(12.0)
        .rename("annualized_p7_gross_return")
        .reset_index()
    )
    waterfall = summary.loc[summary["delay_months"].eq(0)].merge(
        p7_means,
        on=STRATEGY_COLUMNS,
        how="left",
        validate="many_to_one",
    )
    waterfall["annualized_capacity_adjustment"] = (
        waterfall["annualized_gross_capacity_return"]
        - waterfall["annualized_p7_gross_return"]
    )
    waterfall["reconciliation_error"] = (
        waterfall["annualized_net_return"]
        - (
            waterfall["annualized_gross_capacity_return"]
            - waterfall["annualized_transaction_cost"]
            - waterfall["annualized_borrow_cost"]
        )
    )
    return waterfall


def delayed_execution_sensitivity(
    summary: pd.DataFrame,
    config: HypercubeConfig,
) -> pd.DataFrame:
    """Compare immediate eligible entry with the frozen one-month extra delay."""

    selected = summary.loc[
        summary["cost_scenario"].eq(config.costs.primary_cost_scenario)
    ].copy()
    base = selected.loc[selected["delay_months"].eq(0)].copy()
    delayed = selected.loc[
        selected["delay_months"].eq(config.costs.delayed_execution_months)
    ].copy()
    keep = [
        *STRATEGY_COLUMNS,
        "annualized_net_return",
        "annualized_net_sharpe",
        "net_maximum_drawdown",
        "average_capacity_fill_ratio",
    ]
    result = base[keep].merge(
        delayed[keep],
        on=STRATEGY_COLUMNS,
        how="inner",
        suffixes=("_base", "_delayed"),
        validate="one_to_one",
    )
    result["annualized_net_return_change"] = (
        result["annualized_net_return_delayed"]
        - result["annualized_net_return_base"]
    )
    return result


def build_cost_aware_bundle(
    p2_dir: Path,
    p5_dir: Path,
    p7_dir: Path,
    raw_dir: Path,
    output_dir: Path,
    config: HypercubeConfig,
    *,
    scenario: str,
) -> dict[str, Any]:
    """Build one atomic P8 bundle without opening synthetic truth."""

    if config.project.phase != "P8":
        raise PortfolioCostError("Cost-aware portfolios require a P8 config.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite completed P8 output: {output_dir}")
    p7_validation = validate_p7_directory(
        p7_dir,
        p2_dir,
        p5_dir,
        raw_dir,
        config,
        scenario=scenario,
        synthetic_truth_path=None,
    )
    if p7_validation["status"] != "PASS":
        raise PortfolioCostError(
            f"P7 validation failed before P8: {p7_validation['errors']}"
        )
    assignments = prepare_cost_assignments(p7_dir, raw_dir, config)
    paths = pd.read_parquet(p7_dir / "event_month_paths.parquet")
    factors = pd.read_parquet(raw_dir / "factor_returns.parquet")
    lagged = _lagged_execution_liquidity(raw_dir, config)
    position_frames = []
    calendar_frames = []
    for delay in (0, config.costs.delayed_execution_months):
        positions, calendars = build_executed_positions(
            assignments,
            paths,
            lagged,
            config,
            delay_months=delay,
        )
        position_frames.append(positions)
        calendar_frames.append(calendars)
    positions = pd.concat(position_frames, ignore_index=True)
    calendars = pd.concat(calendar_frames, ignore_index=True)
    monthly = evaluate_cost_aware_months(positions, calendars, factors, config)
    summary = summarize_cost_aware_months(monthly, config)
    capacity = capacity_diagnostics(assignments)
    p7_monthly = pd.read_csv(
        p7_dir / "portfolio_monthly_returns.csv",
        parse_dates=["holding_date"],
    )
    p7_monthly["industry_neutral"] = (
        p7_monthly["industry_neutral"].astype(str).str.lower().eq("true")
    )
    waterfall = cost_waterfall(summary, p7_monthly, config)
    delayed = delayed_execution_sensitivity(summary, config)
    primary_mask = (
        summary["signal"].eq(config.returns.primary_signal)
        & summary["weighting"].eq(config.costs.primary_weighting)
        & summary["industry_neutral"].eq(config.costs.primary_industry_neutral)
        & summary["cost_scenario"].eq(config.costs.primary_cost_scenario)
        & summary["delay_months"].eq(0)
    )
    if primary_mask.sum() != 1:
        raise PortfolioCostError("P8 primary portfolio is not unique.")
    primary = summary.loc[primary_mask].iloc[0]
    scientific_status = (
        "POSITIVE_SYNTHETIC_NET_NOT_MARKET_EVIDENCE"
        if float(primary["annualized_net_return"]) > 0.0
        else "NULL_OR_NEGATIVE_AFTER_COSTS"
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".p8-staging-", dir=output_dir.parent))
    try:
        _atomic_parquet(assignments, staging / "capacity_assignments.parquet")
        _atomic_parquet(positions, staging / "executed_positions.parquet")
        _atomic_csv(monthly, staging / "cost_aware_monthly_returns.csv")
        _atomic_csv(summary, staging / "cost_aware_summary.csv")
        _atomic_csv(capacity, staging / "capacity_diagnostics.csv")
        _atomic_csv(waterfall, staging / "cost_waterfall.csv")
        _atomic_csv(delayed, staging / "delayed_execution_sensitivity.csv")
        atomic_write_json(
            staging / "cost_metadata.json",
            {
                "schema_version": 1,
                "phase": "P8",
                "version": P8_VERSION,
                "scenario": scenario,
                "primary_horizon_months": config.costs.primary_horizon_months,
                "primary_signal": config.returns.primary_signal,
                "primary_weighting": config.costs.primary_weighting,
                "primary_industry_neutral": config.costs.primary_industry_neutral,
                "primary_cost_scenario": config.costs.primary_cost_scenario,
                "capacity_rule": (
                    "unrenormalized minimum of original weight, position cap, "
                    "formation/current ADV participation, and market-cap cap"
                ),
                "spread_rule": "dated quoted half-spread with frozen fallback",
                "borrow_rule": "monthly ordinary/hard-borrow drag; unavailable shorts excluded",
                "delayed_execution_months": config.costs.delayed_execution_months,
                "synthetic_truth_read": False,
                "costs_applied": True,
                "borrow_model_applied": True,
                "capacity_model_applied": True,
                "scientific_status": scientific_status,
                "primary_result": {
                    "annualized_gross_capacity_return": float(
                        primary["annualized_gross_capacity_return"]
                    ),
                    "annualized_net_return": float(primary["annualized_net_return"]),
                    "annualized_net_sharpe": float(primary["annualized_net_sharpe"]),
                    "net_maximum_drawdown": float(primary["net_maximum_drawdown"]),
                    "average_turnover": float(primary["average_turnover"]),
                    "average_capacity_fill_ratio": float(
                        primary["average_capacity_fill_ratio"]
                    ),
                },
            },
        )
        atomic_write_json(staging / "resolved_config.json", config.model_dump(mode="json"))
        input_paths = [
            p7_dir / "p7_manifest.json",
            p7_dir / "return_events.parquet",
            p7_dir / "event_month_paths.parquet",
            p7_dir / "portfolio_assignments.parquet",
            p7_dir / "portfolio_monthly_returns.csv",
            raw_dir / "crsp_monthly.parquet",
            raw_dir / "factor_returns.parquet",
        ]
        output_paths = [
            path
            for path in staging.iterdir()
            if path.is_file() and path.name != "p8_manifest.json"
        ]
        manifest = {
            "schema_version": 1,
            "phase": "P8",
            "scenario": scenario,
            "seed": config.project.seed,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": [_file_record(path) for path in input_paths],
            "outputs": [_file_record(path) for path in output_paths],
            "rows": {
                "capacity_assignments": int(len(assignments)),
                "executed_positions": int(len(positions)),
                "monthly_scenario_rows": int(len(monthly)),
                "portfolio_summaries": int(len(summary)),
            },
            "synthetic_truth_read": False,
            "costs_applied": True,
            "scientific_status": scientific_status,
        }
        atomic_write_json(staging / "p8_manifest.json", manifest)
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "scenario": scenario,
        "capacity_assignments": int(len(assignments)),
        "executed_positions": int(len(positions)),
        "monthly_scenario_rows": int(len(monthly)),
        "portfolio_summaries": int(len(summary)),
        "scientific_status": scientific_status,
        "costs_applied": True,
        "synthetic_truth_read": False,
    }


def validate_p8_directory(
    output_dir: Path,
    p7_dir: Path,
    config: HypercubeConfig,
    *,
    scenario: str,
) -> dict[str, Any]:
    """Independently validate saved P8 hashes, timing, costs, and summaries."""

    required = (
        "capacity_assignments.parquet",
        "executed_positions.parquet",
        "cost_aware_monthly_returns.csv",
        "cost_aware_summary.csv",
        "capacity_diagnostics.csv",
        "cost_waterfall.csv",
        "delayed_execution_sensitivity.csv",
        "cost_metadata.json",
        "resolved_config.json",
        "p8_manifest.json",
    )
    errors = [
        f"Missing P8 file: {name}"
        for name in required
        if not (output_dir / name).is_file()
    ]
    if errors:
        return {"status": "FAIL", "errors": errors, "warnings": []}
    manifest = json.loads((output_dir / "p8_manifest.json").read_text(encoding="utf-8"))
    for record in manifest["inputs"]:
        path = Path(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            errors.append(f"P8 input hash mismatch: {path.name}")
    for record in manifest["outputs"]:
        path = output_dir / Path(record["path"]).name
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            errors.append(f"P8 output hash mismatch: {path.name}")
            continue
        if path.suffix == ".parquet" and pq.ParquetFile(path).metadata.num_rows != record["rows"]:
            errors.append(f"P8 parquet row-count mismatch: {path.name}")
        if path.suffix == ".csv":
            rows = max(0, sum(1 for _ in path.open("r", encoding="utf-8")) - 1)
            if rows != record["rows"]:
                errors.append(f"P8 CSV row-count mismatch: {path.name}")
    metadata = json.loads((output_dir / "cost_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("synthetic_truth_read") is not False:
        errors.append("P8 construction read synthetic truth.")
    for flag in ("costs_applied", "borrow_model_applied", "capacity_model_applied"):
        if metadata.get(flag) is not True:
            errors.append(f"P8 metadata flag is false: {flag}")
    assignments = pd.read_parquet(output_dir / "capacity_assignments.parquet")
    positions = pd.read_parquet(output_dir / "executed_positions.parquet")
    monthly = pd.read_csv(
        output_dir / "cost_aware_monthly_returns.csv",
        parse_dates=["holding_date"],
    )
    summary = pd.read_csv(
        output_dir / "cost_aware_summary.csv",
        parse_dates=["first_month", "last_month"],
    )
    waterfall = pd.read_csv(output_dir / "cost_waterfall.csv")
    delayed = pd.read_csv(output_dir / "delayed_execution_sensitivity.csv")
    if (assignments["capacity_weight"] - assignments["weight"] > 1e-12).any():
        errors.append("P8 capacity weight exceeds its frozen gross weight.")
    blocked = assignments["leg"].eq("short") & ~assignments["short_available"].astype(bool)
    if assignments.loc[blocked, "capacity_weight"].ne(0.0).any():
        errors.append("P8 retains a short marked unavailable.")
    if (pd.to_datetime(positions["liquidity_date"]) >= pd.to_datetime(positions["holding_date"])).any():
        errors.append("P8 executed position uses non-lagged liquidity.")
    if (
        positions["actual_signed_weight"].abs()
        - positions["capacity_weight_limit"]
        > 1e-12
    ).any():
        errors.append("P8 executed position breaches its capacity limit.")
    if monthly[["transaction_cost", "borrow_cost", "turnover"]].lt(0.0).any(axis=None):
        errors.append("P8 cost or turnover is negative.")
    recomputed_net = (
        monthly["gross_capacity_return"]
        - monthly["transaction_cost"]
        - monthly["borrow_cost"]
    )
    if not np.allclose(
        monthly["net_return"],
        recomputed_net,
        rtol=1e-12,
        atol=1e-12,
    ):
        errors.append("P8 monthly net-return equation does not reconcile.")
    expected_cases = (
        len(config.returns.signals)
        * len(config.returns.weighting_schemes)
        * len(config.returns.industry_neutral_modes)
        * len(config.costs.scenarios)
        * 2
    )
    if len(summary) != expected_cases:
        errors.append("P8 strategy/cost/delay ladder is incomplete.")
    summary_copy = summary.copy()
    summary_copy["industry_neutral"] = (
        summary_copy["industry_neutral"].astype(str).str.lower().eq("true")
    )
    monthly_copy = monthly.copy()
    monthly_copy["industry_neutral"] = (
        monthly_copy["industry_neutral"].astype(str).str.lower().eq("true")
    )
    grouping = [*STRATEGY_COLUMNS, "delay_months", "cost_scenario"]
    for _, saved in summary_copy.iterrows():
        mask = pd.Series(True, index=monthly_copy.index)
        for column in grouping:
            mask &= monthly_copy[column].eq(saved[column])
        group = monthly_copy.loc[mask].sort_values("holding_date")
        annualized = float(12.0 * group["net_return"].mean())
        if not np.isclose(
            saved["annualized_net_return"],
            annualized,
            rtol=1e-10,
            atol=1e-12,
        ):
            errors.append("P8 annualized net return does not recompute.")
            break
        net = group["net_return"].to_numpy(float)
        volatility = float(np.std(net, ddof=1))
        sharpe = (
            float(math.sqrt(12.0) * np.mean(net) / volatility)
            if volatility > 0.0
            else np.nan
        )
        if not np.isclose(
            saved["annualized_net_sharpe"],
            sharpe,
            rtol=1e-10,
            atol=1e-12,
            equal_nan=True,
        ):
            errors.append("P8 net Sharpe does not recompute.")
            break
    if not np.allclose(
        waterfall["reconciliation_error"],
        0.0,
        rtol=0.0,
        atol=1e-12,
    ):
        errors.append("P8 cost waterfall does not reconcile.")
    expected_delayed = (
        len(config.returns.signals)
        * len(config.returns.weighting_schemes)
        * len(config.returns.industry_neutral_modes)
    )
    if len(delayed) != expected_delayed:
        errors.append("P8 delayed-execution ladder is incomplete.")
    if Path(p7_dir / "p7_manifest.json") not in [
        Path(record["path"]) for record in manifest["inputs"]
    ]:
        errors.append("P8 manifest omits the frozen P7 dependency.")
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "warnings": [],
        "scenario": scenario,
        "rows": {
            "capacity_assignments": int(len(assignments)),
            "executed_positions": int(len(positions)),
            "monthly_scenario_rows": int(len(monthly)),
            "portfolio_summaries": int(len(summary)),
        },
        "scientific_status": metadata.get("scientific_status"),
        "primary_result": metadata.get("primary_result"),
        "costs_applied": True,
        "synthetic_truth_read": False,
    }
