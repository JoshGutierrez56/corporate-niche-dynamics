"""Focused tests for phase-P8 costs, capacity, borrow, and timing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from hypercube.config import load_config
from hypercube.costs import (
    cap_assignment_weights,
    execution_liquidity,
    monthly_borrow_cost,
    one_way_transaction_cost,
)


def test_execution_liquidity_uses_half_spread_and_bounded_capacity() -> None:
    config = load_config("configs/synthetic.yaml")
    frame = pd.DataFrame(
        {
            "prc": [100.0],
            "vol": [10_000.0],
            "bid": [99.0],
            "ask": [101.0],
            "market_cap_millions": [1_000.0],
        }
    )
    result = execution_liquidity(frame, config)
    assert np.isclose(result.loc[0, "half_spread"], 0.01)
    assert result.loc[0, "spread_source"] == "quoted"
    assert 0.0 < result.loc[0, "capacity_weight_limit"] <= 0.10


def test_unavailable_short_receives_no_capacity() -> None:
    config = load_config("configs/synthetic.yaml")
    assignments = pd.DataFrame(
        {
            "weight": [0.05, 0.05],
            "leg": ["long", "short"],
            "capacity_weight_limit": [0.02, 0.02],
            "short_available": [True, False],
        }
    )
    result = cap_assignment_weights(assignments, config)
    assert np.isclose(result.loc[0, "capacity_weight"], 0.02)
    assert result.loc[1, "capacity_weight"] == 0.0
    assert result.loc[1, "capacity_exclusion_reason"] == "short_unavailable"


def test_cost_and_borrow_equations_are_transparent() -> None:
    transaction = one_way_transaction_cost(
        pd.Series([0.10]),
        pd.Series([0.005]),
        spread_multiplier=1.0,
        fixed_slippage_bps=10.0,
    )
    borrow = monthly_borrow_cost(pd.Series([-0.20]), pd.Series([300.0]))
    assert np.isclose(transaction.iloc[0], 0.0006)
    assert np.isclose(borrow.iloc[0], 0.0005)
