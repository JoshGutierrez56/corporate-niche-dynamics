"""Run the P20E incremental survival-utility audit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

import pandas as pd

from hypercube.config import load_config
from hypercube.data import SCENARIOS, atomic_write_json, sha256_file
from hypercube.survival_utility import (
    P20E_VERSION,
    paired_comparisons,
    run_survival_utility,
    utility_gate,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BANNED_SOURCE_PARTS = ("synthetic_truth", "return", "portfolio", "\\p7\\", "\\p8\\")


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False, lineterminator="\n")
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.to_parquet(temporary, index=False)
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(PROJECT_ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _weighted_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    fields = (
        "roc_auc",
        "average_precision",
        "brier_score",
        "log_loss",
        "expected_calibration_error",
    )
    for key, group in metrics.groupby(
        ["scenario", "horizon_years", "model"], sort=True
    ):
        weights = group["test_rows"].astype(float)
        row: dict[str, Any] = {
            "scenario": key[0],
            "horizon_years": int(key[1]),
            "model": key[2],
            "folds": int(len(group)),
            "test_rows": int(group["test_rows"].sum()),
            "test_failures": int(group["test_failures"].sum()),
        }
        for field in fields:
            row[field] = float((group[field] * weights).sum() / weights.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _existing_hypercube_comparisons(p4_metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["scenario", "horizon_years", "fold"]
    wide = p4_metrics.pivot(index=keys, columns="model")
    rows: list[dict[str, Any]] = []
    for benchmark in (
        "profitability_logit",
        "distress_logit",
        "industry_rate",
        "occupied_cell_rate",
    ):
        auc = (
            wide[("roc_auc", "combined_axes_logit")]
            - wide[("roc_auc", benchmark)]
        )
        brier = (
            wide[("brier_score", benchmark)]
            - wide[("brier_score", "combined_axes_logit")]
        )
        rows.append(
            {
                "candidate_model": "combined_axes_logit",
                "benchmark_model": benchmark,
                "folds": int(len(auc)),
                "mean_auc_improvement": float(auc.mean()),
                "median_auc_improvement": float(auc.median()),
                "auc_wins": int(auc.gt(0.0).sum()),
                "mean_brier_improvement": float(brier.mean()),
                "brier_wins": int(brier.gt(0.0).sum()),
            }
        )
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/synthetic.yaml")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("artifacts/p20e")
    )
    parser.add_argument(
        "--tables-dir", type=Path, default=Path("artifacts/tables")
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/manifests/p20e_survival_utility.json"),
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)

    source_paths = [
        PROJECT_ROOT / "artifacts" / "tables" / "p4_fold_metrics.csv",
        PROJECT_ROOT / "artifacts" / "tables" / "p6_fold_metrics.csv",
        PROJECT_ROOT / "artifacts" / "manifests" / "p4_validation.json",
        PROJECT_ROOT / "artifacts" / "manifests" / "p6_validation.json",
    ]
    for scenario in SCENARIOS:
        source_paths.append(
            PROJECT_ROOT
            / "data"
            / "processed"
            / "synthetic"
            / scenario
            / "p4"
            / "model_matrix.parquet"
        )
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        normalized = str(path).lower()
        if any(part in normalized for part in BANNED_SOURCE_PARTS):
            raise RuntimeError(f"P20E attempted a banned source: {path}")
    source_records = [_record(path) for path in source_paths]

    predictions: list[pd.DataFrame] = []
    metrics: list[pd.DataFrame] = []
    folds: list[pd.DataFrame] = []
    for scenario in SCENARIOS:
        matrix = pd.read_parquet(
            PROJECT_ROOT
            / "data"
            / "processed"
            / "synthetic"
            / scenario
            / "p4"
            / "model_matrix.parquet"
        )
        scenario_predictions, scenario_metrics, scenario_folds = (
            run_survival_utility(matrix, config, scenario)
        )
        predictions.append(scenario_predictions)
        metrics.append(scenario_metrics)
        folds.append(scenario_folds)

    prediction_frame = pd.concat(predictions, ignore_index=True)
    metric_frame = pd.concat(metrics, ignore_index=True)
    fold_frame = pd.concat(folds, ignore_index=True)
    comparisons = paired_comparisons(metric_frame)
    gate = utility_gate(comparisons)
    summary = _weighted_summary(metric_frame)

    p4_metrics = pd.read_csv(source_paths[0])
    existing_comparisons = _existing_hypercube_comparisons(p4_metrics)
    p6_failure = pd.read_csv(source_paths[1]).loc[
        lambda frame: frame["cause"].eq("performance_failure")
    ].reset_index(drop=True)

    output_dir = PROJECT_ROOT / args.output_dir
    tables_dir = PROJECT_ROOT / args.tables_dir
    report_path = PROJECT_ROOT / args.report
    output_paths = {
        "predictions": output_dir / "survival_predictions.parquet",
        "fold_definitions": output_dir / "fold_definitions.csv",
        "metrics": tables_dir / "p20e_survival_fold_metrics.csv",
        "summary": tables_dir / "p20e_survival_model_summary.csv",
        "comparisons": tables_dir / "p20e_survival_pairwise.csv",
        "existing_hypercube": (
            tables_dir / "p20e_existing_hypercube_comparisons.csv"
        ),
        "p6_failure": tables_dir / "p20e_p6_failure_metrics.csv",
    }
    _atomic_parquet(prediction_frame, output_paths["predictions"])
    _atomic_csv(fold_frame, output_paths["fold_definitions"])
    _atomic_csv(metric_frame, output_paths["metrics"])
    _atomic_csv(summary, output_paths["summary"])
    _atomic_csv(comparisons, output_paths["comparisons"])
    _atomic_csv(existing_comparisons, output_paths["existing_hypercube"])
    _atomic_csv(p6_failure, output_paths["p6_failure"])

    source_records_after = [_record(path) for path in source_paths]
    if source_records_after != source_records:
        raise RuntimeError("A frozen P4/P6 source changed during P20E.")
    p6_summary = {
        "folds": int(len(p6_failure)),
        "mean_concordance": float(p6_failure["concordance"].mean()),
        "mean_roc_auc": float(p6_failure["roc_auc"].mean()),
        "minimum_roc_auc": float(p6_failure["roc_auc"].min()),
        "maximum_roc_auc": float(p6_failure["roc_auc"].max()),
    }
    report = {
        "schema_version": 1,
        "phase": "P20E",
        "version": P20E_VERSION,
        "status": "PASS",
        "primary_result": gate,
        "p6_performance_failure": p6_summary,
        "rows": {
            "predictions": int(len(prediction_frame)),
            "metric_rows": int(len(metric_frame)),
            "comparison_rows": int(len(comparisons)),
            "fold_rows": int(len(fold_frame)),
        },
        "models": list(metric_frame["model"].drop_duplicates()),
        "scenarios": list(SCENARIOS),
        "horizons_years": list(config.viability.horizons_years),
        "model_fit": True,
        "synthetic_truth_read": False,
        "return_data_read": False,
        "real_data_read": False,
        "portfolio_run": False,
        "source_records": source_records,
        "output_records": [
            _record(path) for path in output_paths.values()
        ],
    }
    atomic_write_json(report_path, report)
    print(
        "P20E",
        report["status"],
        gate["verdict"],
        f"mean_auc={gate['mean_auc_improvement']:.6f}",
        f"wins={gate['auc_wins']}/{gate['folds']}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
