"""P12E tests for deterministic, held-out synthetic power calibration."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hypercube.data import atomic_write_json
from hypercube.exploratory import atomic_write_csv
from hypercube.power_calibration import (
    calibrate_alpha_injection_power,
    validate_power_calibration_outputs,
)


def _fixtures(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(120)
    rows = 900
    signal = rng.normal(size=rows)
    observable = signal + rng.normal(scale=0.10, size=rows)
    injected = 0.03 * signal
    expected_fraction = sum(np.exp(-offset / 4.0) for offset in range(6)) / sum(
        np.exp(-offset / 4.0) for offset in range(12)
    )
    returns = expected_fraction * injected + rng.normal(scale=0.02, size=rows)
    targets = pd.DataFrame(
        {
            "gvkey": [f"{index % 90:06d}" for index in range(rows)],
            "datadate": pd.date_range("2000-01-01", periods=rows, freq="D"),
            "fyear": 2000 + np.arange(rows) % 20,
            "horizon_months": 6,
            "target_valid": True,
            "migration_surprise": observable,
            "forward_excess_return": returns,
        }
    )
    truth = targets.loc[:, ["gvkey", "datadate", "fyear"]].copy()
    truth["injected_return_alpha"] = injected
    targets_path = tmp_path / "targets.parquet"
    truth_path = tmp_path / "truth.parquet"
    targets.to_parquet(targets_path, index=False)
    truth.to_parquet(truth_path, index=False)
    return targets_path, truth_path


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


def test_power_calibration_selects_and_holds_out_a_multiplier(
    tmp_path: Path,
) -> None:
    targets_path, truth_path = _fixtures(tmp_path)

    summary, table = calibrate_alpha_injection_power(
        targets_path,
        truth_path,
        **_kwargs(),
    )

    assert summary["status"] == "PASS"
    assert summary["selected_multiplier"] in (1.0, 2.0)
    assert summary["held_out_evaluation_pass"] is True
    assert summary["new_synthetic_scenario_generated"] is False
    assert summary["stop_before_new_synthetic_generation"] is True
    assert len(table) == 2


def test_power_calibration_validator_recomputes_outputs(tmp_path: Path) -> None:
    targets_path, truth_path = _fixtures(tmp_path)
    summary, table = calibrate_alpha_injection_power(
        targets_path,
        truth_path,
        **_kwargs(),
    )
    summary["generated_at_utc"] = "2026-07-23T00:00:00+00:00"
    summary_path = tmp_path / "summary.json"
    table_path = tmp_path / "table.csv"
    atomic_write_json(summary_path, summary)
    atomic_write_csv(table, table_path)

    report = validate_power_calibration_outputs(
        targets_path,
        truth_path,
        summary_path,
        table_path,
        calibration_kwargs=_kwargs(),
    )

    assert report["status"] == "PASS"
    assert report["errors"] == []
