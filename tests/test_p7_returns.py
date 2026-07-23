"""P7 tests for return timing, delistings, inference, and portfolio weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hypercube.config import load_config
from hypercube.returns import (
    _hac_mean,
    _leg_weights,
    construct_forward_paths,
)


ROOT = Path(__file__).resolve().parents[1]


def _config():
    return load_config(ROOT / "configs" / "synthetic.yaml")


def test_forward_paths_start_next_month_and_hold_cash_after_delisting() -> None:
    """A valid delisting is compounded once and then earns RF through horizon."""

    config = _config()
    events = pd.DataFrame(
        {
            "event_id": ["A|2019-12-31|2019", "B|2019-12-31|2019"],
            "gvkey": ["A", "B"],
            "permno": [1, 2],
            "feature_date": pd.to_datetime(["2020-01-31", "2020-01-31"]),
            "fold": [1, 1],
            "datadate": pd.to_datetime(["2019-12-31", "2019-12-31"]),
            "fyear": [2019, 2019],
        }
    )
    events["availability_date"] = pd.Timestamp("2020-01-01")
    events["formation_date"] = events["feature_date"]
    events["feature_year"] = 2020
    events["sic2"] = 20
    events["sic1"] = 2
    events["market_cap_millions"] = 100.0
    for column in (*config.returns.signals, *config.returns.controls):
        events[column] = 0.0
    history = pd.DataFrame(
        {
            "gvkey": ["A", "A", "B", "B"],
            "date": pd.to_datetime(
                ["2020-02-29", "2020-03-31", "2020-02-29", "2020-03-31"]
            ),
            "permno": [1, 1, 2, 2],
            "ret_total": [0.10, -0.20, 0.05, 0.02],
            "has_delist_event": [False, True, False, True],
            "delist_return_missing": [False, False, False, True],
            "exit_category": [None, "performance_failure", None, "other_unknown"],
            "market_cap_millions": [100.0, 80.0, 100.0, 90.0],
        }
    )
    dates = pd.date_range("2020-02-29", periods=12, freq="ME")
    factors = pd.DataFrame(
        {
            "date": dates,
            "rf": 0.001,
            "mkt_excess": 0.0,
            "smb": 0.0,
            "hml": 0.0,
            "rmw": 0.0,
            "cma": 0.0,
            "mom": 0.0,
        }
    )
    paths, targets = construct_forward_paths(events, history, factors, config)
    assert (paths["holding_date"] > paths["feature_date"]).all()
    a_paths = paths.loc[paths["gvkey"].eq("A")]
    assert a_paths.iloc[2]["cash_after_delist"]
    assert np.isclose(a_paths.iloc[2]["return_used"], 0.001)
    a_six = targets.loc[
        targets["event_id"].str.startswith("A|") & targets["horizon_months"].eq(6)
    ].iloc[0]
    expected = 1.10 * 0.80 * (1.001**4) - 1.0
    assert a_six["target_valid"]
    assert a_six["target_status"] == "complete_after_delist"
    assert np.isclose(a_six["forward_total_return"], expected)
    b_six = targets.loc[
        targets["event_id"].str.startswith("B|") & targets["horizon_months"].eq(6)
    ].iloc[0]
    assert not b_six["target_valid"]
    assert b_six["target_status"] == "missing_delisting_return"
    assert pd.isna(b_six["forward_total_return"])


def test_hac_mean_reports_deterministic_mean() -> None:
    """The saved Fama-MacBeth time-series mean is not outcome-selected."""

    result = _hac_mean(pd.Series([1.0, 2.0, 3.0, 4.0]), 1)
    assert result["mean"] == 2.5
    assert result["standard_error"] > 0.0


def test_industry_neutral_leg_weights_equalize_industry_budgets() -> None:
    """Industry-neutral legs allocate equal gross capital to each industry."""

    frame = pd.DataFrame(
        {
            "sic1": [1, 1, 2],
            "market_cap_millions": [10.0, 30.0, 60.0],
        }
    )
    weights = _leg_weights(frame, "value", True)
    assert np.isclose(weights.sum(), 1.0)
    assert np.isclose(weights.loc[frame["sic1"].eq(1)].sum(), 0.5)
    assert np.isclose(weights.loc[frame["sic1"].eq(2)].sum(), 0.5)
