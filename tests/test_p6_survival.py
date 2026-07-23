"""P6 interval, competing-exit, delayed-entry, and PH model gates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hypercube.config import HypercubeConfig, load_config
from hypercube.survival import (
    CAUSE_COLUMNS,
    construct_survival_intervals,
    fit_cause_specific_model,
    interval_concordance,
)


ROOT = Path(__file__).resolve().parents[1]


def _compact_config() -> HypercubeConfig:
    config = load_config(ROOT / "configs" / "synthetic.yaml")
    survival = config.survival.model_copy(
        update={
            "minimum_train_intervals": 100,
            "minimum_train_events": 5,
            "minimum_test_events": 2,
        }
    )
    return config.model_copy(update={"survival": survival})


def _write_interval_fixture(root: Path, config: HypercubeConfig) -> tuple[Path, Path, Path]:
    p2 = root / "p2"
    p4 = root / "p4"
    p5 = root / "p5"
    p2.mkdir()
    p4.mkdir()
    p5.mkdir()
    event_rows = []
    accounting_rows = []
    metadata_rows = []
    specifications = (
        ("A", 1, pd.Timestamp("2000-06-30"), pd.Timestamp("2002-02-15"), "performance_failure", pd.Timestamp("2002-02-28"), 3),
        ("A", 1, pd.Timestamp("2001-06-30"), pd.Timestamp("2002-02-15"), "performance_failure", pd.Timestamp("2002-02-28"), 4),
        ("B", 2, pd.Timestamp("2000-07-31"), pd.Timestamp("2001-05-15"), "merger", pd.Timestamp("2001-05-31"), 3),
        ("C", 3, pd.Timestamp("2000-08-31"), pd.NaT, pd.NA, pd.Timestamp("2002-12-31"), 3),
    )
    for index, (gvkey, permno, feature_date, exit_date, category, last_date, history) in enumerate(specifications):
        datadate = pd.Timestamp(feature_date.year - 1, 12, 31)
        row = {
            "gvkey": gvkey,
            "permno": permno,
            "datadate": datadate,
            "fyear": datadate.year,
            "horizon_years": 5,
            "feature_date": feature_date,
            "availability_date": feature_date - pd.Timedelta(days=45),
            "formation_date": feature_date,
            "sic2": 20,
        }
        for feature_number, feature in enumerate(config.survival.features):
            row[feature] = 0.1 * index + 0.01 * feature_number
        event_rows.append(row)
        accounting_rows.append(
            {
                "gvkey": gvkey,
                "datadate": datadate,
                "fyear": datadate.year,
                "reporting_history": history,
            }
        )
        metadata_rows.append(
            {
                "gvkey": gvkey,
                "datadate": datadate,
                "fyear": datadate.year,
                "horizon_years": 5,
                "exit_date": exit_date,
                "exit_category": category,
                "last_observed_date": last_date,
            }
        )
    pd.DataFrame(event_rows).to_parquet(p5 / "frontier_dynamics.parquet", index=False)
    pd.DataFrame(accounting_rows).to_parquet(
        p2 / "accounting_availability.parquet", index=False
    )
    pd.DataFrame(metadata_rows).to_parquet(p4 / "model_matrix.parquet", index=False)
    return p2, p4, p5


def test_intervals_reconcile_failure_merger_and_censoring(tmp_path: Path) -> None:
    """Each dated exit terminates one interval and competing causes remain separate."""

    config = _compact_config()
    p2, p4, p5 = _write_interval_fixture(tmp_path, config)
    intervals, reconciliation = construct_survival_intervals(p2, p4, p5, config)
    assert int(intervals["performance_failure_event"].sum()) == 1
    assert int(intervals["merger_event"].sum()) == 1
    assert set(intervals["terminal_reason"]) == {
        "covariate_update",
        "performance_failure",
        "merger",
        "administrative_right_censor",
    }
    assert reconciliation["difference"].eq(0).all()
    ordered = intervals.sort_values(["gvkey", "interval_start_date"])
    prior_stop = ordered.groupby("gvkey")["interval_stop_date"].shift(1)
    assert (prior_stop.dropna() <= ordered.loc[prior_stop.notna(), "interval_start_date"]).all()
    assert (intervals["entry_day"] < intervals["stop_day"]).all()


def _model_fixture(config: HypercubeConfig) -> pd.DataFrame:
    rng = np.random.default_rng(616)
    rows = []
    for index in range(240):
        start = 36000.0 + index * 2.0
        cause = int(index % 20 == 0)
        row = {
            "gvkey": f"{index // 3:04d}",
            "entry_day": start,
            "stop_day": start + 365.0,
            "performance_failure_event": cause,
            "merger_event": int(index % 24 == 0),
        }
        latent = rng.normal() - cause
        for feature_number, feature in enumerate(config.survival.features):
            row[feature] = latent + feature_number * 0.01 + rng.normal(0.0, 0.2)
        rows.append(row)
    return pd.DataFrame(rows)


def test_cause_specific_ph_model_has_finite_coefficients() -> None:
    """The frozen PH estimator fits without mixing failure and merger status."""

    config = _compact_config()
    frame = _model_fixture(config)
    fitted = fit_cause_specific_model(frame, "performance_failure", config)
    assert fitted["event_column"] == CAUSE_COLUMNS["performance_failure"]
    assert np.isfinite(fitted["params"]).all()
    assert np.isfinite(fitted["covariance"]).all()


def test_interval_concordance_uses_only_dated_risk_sets() -> None:
    """A perfectly ordered event risk receives unit risk-set concordance."""

    frame = pd.DataFrame(
        {
            "gvkey": ["A", "B", "C"],
            "entry_day": [0.0, 0.0, 0.0],
            "stop_day": [2.0, 4.0, 5.0],
            "event": [1, 0, 0],
        }
    )
    value, pairs = interval_concordance(frame, np.array([3.0, 2.0, 1.0]), "event")
    assert value == 1.0
    assert pairs == 2
