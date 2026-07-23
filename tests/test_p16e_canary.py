"""P16E tests for realized migration and null return gates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hypercube.canary import evaluate_realized_canary


def test_realized_canary_recovers_signal_and_null(tmp_path: Path) -> None:
    rng = np.random.default_rng(1616)
    rows = 2400
    signal = rng.normal(size=rows)
    proxy = signal + rng.normal(scale=0.15, size=rows)
    injected = 0.12 * signal
    migration_return = 0.82 * injected + rng.normal(scale=0.04, size=rows)
    null_return = rng.normal(scale=0.04, size=rows)
    keys = pd.DataFrame(
        {
            "gvkey": [f"{index % 240:06d}" for index in range(rows)],
            "datadate": pd.date_range("2002-01-01", periods=rows, freq="D"),
            "fyear": 2002 + np.arange(rows) % 17,
        }
    )
    migration_targets = keys.copy()
    migration_targets["horizon_months"] = 6
    migration_targets["target_valid"] = True
    migration_targets["migration_surprise"] = proxy
    migration_targets["forward_excess_return"] = migration_return
    null_targets = migration_targets.copy()
    null_targets["forward_excess_return"] = null_return
    candidates = keys.copy()
    candidates["anchored_axis_innovation"] = proxy
    truth = keys.copy()
    truth["migration_surprise"] = signal
    truth["injected_return_alpha"] = injected
    paths = {
        "migration_targets": tmp_path / "migration_targets.parquet",
        "null_targets": tmp_path / "null_targets.parquet",
        "migration_candidates": tmp_path / "migration_candidates.parquet",
        "null_candidates": tmp_path / "null_candidates.parquet",
        "truth": tmp_path / "truth.parquet",
    }
    migration_targets.to_parquet(paths["migration_targets"], index=False)
    null_targets.to_parquet(paths["null_targets"], index=False)
    candidates.to_parquet(paths["migration_candidates"], index=False)
    candidates.to_parquet(paths["null_candidates"], index=False)
    truth.to_parquet(paths["truth"], index=False)

    summary, metrics = evaluate_realized_canary(
        paths["migration_targets"],
        paths["truth"],
        paths["migration_candidates"],
        paths["null_targets"],
        paths["null_candidates"],
    )

    assert summary["status"] == "PASS"
    assert all(summary["gates"].values())
    assert len(metrics) == 2
