"""P15E tests for locked-proxy power calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hypercube.data import atomic_write_json
from hypercube.exploratory import atomic_write_csv
from hypercube.power_calibration import (
    calibrate_locked_proxy_power,
    validate_locked_proxy_power_outputs,
)


def _fixtures(tmp_path: Path) -> tuple[Path, Path, Path]:
    rng = np.random.default_rng(1515)
    rows = 1200
    signal = rng.normal(size=rows)
    proxy = signal + rng.normal(scale=0.12, size=rows)
    injected = 0.03 * signal
    expected_fraction = sum(np.exp(-offset / 4.0) for offset in range(6)) / sum(
        np.exp(-offset / 4.0) for offset in range(12)
    )
    returns = expected_fraction * injected + rng.normal(scale=0.025, size=rows)
    targets = pd.DataFrame(
        {
            "gvkey": [f"{index % 120:06d}" for index in range(rows)],
            "datadate": pd.date_range("2002-01-01", periods=rows, freq="D"),
            "fyear": 2002 + np.arange(rows) % 17,
            "horizon_months": 6,
            "target_valid": True,
            "migration_surprise": proxy,
            "forward_excess_return": returns,
        }
    )
    truth = targets[list(("gvkey", "datadate", "fyear"))].copy()
    truth["injected_return_alpha"] = injected
    candidates = targets[list(("gvkey", "datadate", "fyear"))].copy()
    candidates["anchored_axis_innovation"] = proxy
    targets_path = tmp_path / "targets.parquet"
    truth_path = tmp_path / "truth.parquet"
    candidates_path = tmp_path / "candidates.parquet"
    targets.to_parquet(targets_path, index=False)
    truth.to_parquet(truth_path, index=False)
    candidates.to_parquet(candidates_path, index=False)
    return targets_path, truth_path, candidates_path


def _kwargs() -> dict[str, object]:
    return {
        "multipliers": (1.0, 2.0),
        "calibration_replicates": 12,
        "evaluation_replicates": 8,
        "minimum_oracle_detection_rate": 0.50,
        "minimum_observable_detection_rate": 0.50,
        "maximum_null_false_positive_rate": 0.50,
        "minimum_median_oracle_rank_ic": 0.01,
        "minimum_median_observable_rank_ic": 0.01,
    }


def test_locked_proxy_power_selects_and_holds_out(tmp_path: Path) -> None:
    targets_path, truth_path, candidates_path = _fixtures(tmp_path)

    summary, table = calibrate_locked_proxy_power(
        targets_path,
        truth_path,
        candidates_path,
        **_kwargs(),
    )

    assert summary["status"] == "GO"
    assert summary["selected_multiplier"] in (1.0, 2.0)
    assert summary["held_out_evaluation_pass"] is True
    assert summary["new_synthetic_scenario_generated"] is False
    assert len(table) == 2


def test_locked_proxy_power_validator_recomputes(tmp_path: Path) -> None:
    targets_path, truth_path, candidates_path = _fixtures(tmp_path)
    summary, table = calibrate_locked_proxy_power(
        targets_path,
        truth_path,
        candidates_path,
        **_kwargs(),
    )
    summary_path = tmp_path / "summary.json"
    table_path = tmp_path / "table.csv"
    atomic_write_json(summary_path, summary)
    atomic_write_csv(table, table_path)

    report = validate_locked_proxy_power_outputs(
        targets_path,
        truth_path,
        candidates_path,
        summary_path,
        table_path,
        calibration_kwargs=_kwargs(),
    )

    assert report["status"] == "PASS"
    assert report["errors"] == []
