"""P11E tests for the versioned post-closeout alpha-power diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hypercube.exploratory import (
    AlphaPowerError,
    atomic_write_csv,
    diagnose_alpha_recovery_power,
    validate_alpha_power_outputs,
)
from hypercube.data import atomic_write_json


def _fixtures(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(711)
    rows = 1_200
    truth_signal = rng.normal(size=rows)
    observable = truth_signal + rng.normal(scale=0.25, size=rows)
    injected = 0.03 * truth_signal
    forward = 0.8 * injected + rng.normal(scale=0.03, size=rows)
    targets = pd.DataFrame(
        {
            "gvkey": [f"{index % 120:06d}" for index in range(rows)],
            "datadate": pd.date_range("2000-12-31", periods=rows, freq="D"),
            "fyear": 2000 + np.arange(rows) % 20,
            "horizon_months": 6,
            "target_valid": True,
            "migration_surprise": observable,
            "forward_excess_return": forward,
            "fold": 1 + np.arange(rows) % 4,
        }
    )
    truth = targets.loc[:, ["gvkey", "datadate", "fyear"]].copy()
    truth["migration_surprise"] = truth_signal
    truth["injected_return_alpha"] = injected
    targets_path = tmp_path / "targets.parquet"
    truth_path = tmp_path / "truth.parquet"
    targets.to_parquet(targets_path, index=False)
    truth.to_parquet(truth_path, index=False)
    return targets_path, truth_path


def test_diagnostic_detects_a_powerful_oracle(tmp_path: Path) -> None:
    targets_path, truth_path = _fixtures(tmp_path)

    summary, folds = diagnose_alpha_recovery_power(targets_path, truth_path)

    assert summary["status"] == "DESCRIPTIVE_EXPLORATORY"
    assert summary["interpretation"] == "ORACLE_DETECTABLE"
    assert summary["oracle_detectable_at_5pct"] is True
    assert summary["frozen_p0_p10_outputs_modified"] is False
    assert summary["real_data_run"] is False
    assert summary["rows"] == 1_200
    assert len(folds) == 4


def test_validator_recomputes_saved_outputs(tmp_path: Path) -> None:
    targets_path, truth_path = _fixtures(tmp_path)
    summary, folds = diagnose_alpha_recovery_power(targets_path, truth_path)
    summary["generated_at_utc"] = "2026-07-23T00:00:00+00:00"
    summary_path = tmp_path / "summary.json"
    folds_path = tmp_path / "folds.csv"
    atomic_write_json(summary_path, summary)
    atomic_write_csv(folds, folds_path)

    report = validate_alpha_power_outputs(
        targets_path,
        truth_path,
        summary_path,
        folds_path,
    )

    assert report["status"] == "PASS"
    assert report["errors"] == []


def test_diagnostic_rejects_missing_truth(tmp_path: Path) -> None:
    targets_path, truth_path = _fixtures(tmp_path)
    truth = pd.read_parquet(truth_path).iloc[:-1]
    truth.to_parquet(truth_path, index=False)

    with pytest.raises(AlphaPowerError, match="does not fully match"):
        diagnose_alpha_recovery_power(targets_path, truth_path)
