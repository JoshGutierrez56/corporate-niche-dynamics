"""Independently validate the saved P20E survival-utility audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from hypercube.data import atomic_write_json, sha256_file
from hypercube.survival_utility import (
    MODEL_ORDER,
    paired_comparisons,
    prediction_metrics,
    utility_gate,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _check_record(
    record: dict[str, Any], errors: list[str], label: str
) -> None:
    path = PROJECT_ROOT / record["path"]
    if not path.is_file():
        errors.append(f"{label} missing: {record['path']}")
        return
    if path.stat().st_size != record["bytes"]:
        errors.append(f"{label} byte mismatch: {record['path']}")
    if sha256_file(path) != record["sha256"]:
        errors.append(f"{label} hash mismatch: {record['path']}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-report",
        type=Path,
        default=Path("artifacts/manifests/p20e_survival_utility.json"),
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("artifacts/p20e/survival_predictions.parquet"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("artifacts/tables/p20e_survival_fold_metrics.csv"),
    )
    parser.add_argument(
        "--comparisons",
        type=Path,
        default=Path("artifacts/tables/p20e_survival_pairwise.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/manifests/p20e_survival_utility_validation.json"
        ),
    )
    args = parser.parse_args(argv)

    report = json.loads(
        (PROJECT_ROOT / args.input_report).read_text(encoding="utf-8")
    )
    errors: list[str] = []
    for record in report["source_records"]:
        _check_record(record, errors, "source")
        lowered = record["path"].lower()
        if any(
            banned in lowered
            for banned in ("synthetic_truth", "return", "portfolio", "/p7/", "/p8/")
        ):
            errors.append(f"Banned P20E source recorded: {record['path']}")
    for record in report["output_records"]:
        _check_record(record, errors, "output")

    predictions = pd.read_parquet(PROJECT_ROOT / args.predictions)
    metrics = pd.read_csv(PROJECT_ROOT / args.metrics)
    comparisons = pd.read_csv(PROJECT_ROOT / args.comparisons)
    if set(predictions["model"]) != set(MODEL_ORDER):
        errors.append("Saved predictions do not contain the frozen model pair.")
    if not predictions["predicted_failure_probability"].between(0.0, 1.0).all():
        errors.append("Saved failure probabilities fall outside [0, 1].")
    if not np.allclose(
        predictions["calibrated_survival_probability"],
        1.0 - predictions["predicted_failure_probability"],
        rtol=0.0,
        atol=1e-12,
    ):
        errors.append("Failure and survival probabilities do not complement.")
    identity = [
        "scenario",
        "horizon_years",
        "fold",
        "model",
        "permno",
        "gvkey",
        "datadate",
    ]
    if predictions.duplicated(identity).any():
        errors.append("Duplicate saved P20E prediction identity.")

    keys = ["scenario", "horizon_years", "fold", "model"]
    saved = metrics.set_index(keys).sort_index()
    checked_metrics = 0
    for key, group in predictions.groupby(keys, sort=True):
        recomputed = prediction_metrics(
            group["failure_within_horizon"].astype(int).to_numpy(),
            group["predicted_failure_probability"].to_numpy(float),
        )
        if key not in saved.index:
            errors.append(f"Missing metric row: {key}")
            continue
        for name, value in recomputed.items():
            if not np.isclose(
                float(saved.loc[key, name]),
                float(value),
                rtol=1e-10,
                atol=1e-12,
                equal_nan=True,
            ):
                errors.append(f"Metric mismatch {key} {name}.")
        checked_metrics += 1

    recomputed_comparisons = paired_comparisons(metrics)
    comparison_columns = [
        "auc_improvement",
        "average_precision_improvement",
        "brier_improvement",
        "log_loss_improvement",
    ]
    if len(recomputed_comparisons) != len(comparisons):
        errors.append("P20E comparison-row count mismatch.")
    else:
        for column in comparison_columns:
            if not np.allclose(
                recomputed_comparisons[column],
                comparisons[column],
                rtol=1e-10,
                atol=1e-12,
            ):
                errors.append(f"P20E comparison mismatch: {column}.")
    recomputed_gate = utility_gate(recomputed_comparisons)
    saved_gate = report["primary_result"]
    for field in (
        "verdict",
        "gates",
        "auc_wins",
        "folds",
        "brier_wins",
        "positive_scenario_horizon_cells",
        "scenario_horizon_cells",
    ):
        if recomputed_gate[field] != saved_gate[field]:
            errors.append(f"P20E primary gate mismatch: {field}.")
    for field in (
        "mean_auc_improvement",
        "median_auc_improvement",
        "mean_brier_improvement",
    ):
        if not np.isclose(
            float(recomputed_gate[field]),
            float(saved_gate[field]),
            rtol=1e-10,
            atol=1e-12,
        ):
            errors.append(f"P20E primary gate mismatch: {field}.")

    for name in ("p4_validation.json", "p6_validation.json"):
        receipt = json.loads(
            (
                PROJECT_ROOT / "artifacts" / "manifests" / name
            ).read_text(encoding="utf-8")
        )
        if receipt.get("status") != "PASS":
            errors.append(f"Upstream receipt is not PASS: {name}")
    if report.get("synthetic_truth_read") is not False:
        errors.append("P20E synthetic-truth declaration is not false.")
    if report.get("return_data_read") is not False:
        errors.append("P20E return-data declaration is not false.")
    if report.get("real_data_read") is not False:
        errors.append("P20E real-data declaration is not false.")

    result = {
        "schema_version": 1,
        "phase": "P20E",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "prediction_rows": int(len(predictions)),
        "metric_rows_checked": checked_metrics,
        "comparison_rows_checked": int(len(recomputed_comparisons)),
        "primary_verdict": recomputed_gate["verdict"],
        "source_records_checked": int(len(report["source_records"])),
        "output_records_checked": int(len(report["output_records"])),
    }
    atomic_write_json(PROJECT_ROOT / args.output, result)
    if errors:
        print("\n".join(errors))
        return 1
    print(
        "P20E_VALIDATION PASS",
        f"predictions={len(predictions)}",
        f"metrics={checked_metrics}",
        f"verdict={recomputed_gate['verdict']}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
