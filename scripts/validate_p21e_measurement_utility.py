"""Independently validate the saved P21E measurement-utility audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from hypercube.data import SCENARIOS, atomic_write_json, sha256_file
from hypercube.measurement_utility import (
    crowding_gate,
    drift_alert_gate,
    peer_map_gate,
    product_gate,
    summarize_drift_events,
    summarize_peer_geometry,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _check_record(
    record: dict[str, Any],
    errors: list[str],
    label: str,
) -> None:
    path = PROJECT_ROOT / record["path"]
    if not path.is_file():
        errors.append(f"{label} missing: {record['path']}")
        return
    if path.stat().st_size != record["bytes"]:
        errors.append(f"{label} byte mismatch: {record['path']}")
    if sha256_file(path) != record["sha256"]:
        errors.append(f"{label} hash mismatch: {record['path']}")


def _compare_frames(
    saved: pd.DataFrame,
    recomputed: pd.DataFrame,
    errors: list[str],
    label: str,
) -> None:
    try:
        pd.testing.assert_frame_equal(
            saved,
            recomputed,
            check_dtype=False,
            rtol=1e-10,
            atol=1e-12,
        )
    except AssertionError as exc:
        errors.append(f"{label} mismatch: {exc}")


def _compare_result(
    saved: dict[str, Any],
    recomputed: dict[str, Any],
    errors: list[str],
    label: str,
) -> None:
    if saved.keys() != recomputed.keys():
        errors.append(f"{label} keys mismatch.")
        return
    for key, expected in recomputed.items():
        value = saved[key]
        if isinstance(expected, float):
            if not np.isclose(
                float(value),
                expected,
                rtol=1e-10,
                atol=1e-12,
            ):
                errors.append(f"{label} float mismatch: {key}.")
        elif value != expected:
            errors.append(f"{label} mismatch: {key}.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-report",
        type=Path,
        default=Path("artifacts/manifests/p21e_measurement_utility.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts/manifests/p21e_measurement_utility_validation.json"
        ),
    )
    args = parser.parse_args(argv)

    report = json.loads(
        (PROJECT_ROOT / args.input_report).read_text(encoding="utf-8")
    )
    errors: list[str] = []
    for record in report["source_records"]:
        _check_record(record, errors, "source")
    for record in report["output_records"]:
        _check_record(record, errors, "output")

    drift_events = pd.read_parquet(
        PROJECT_ROOT / "artifacts" / "p21e" / "drift_alert_events.parquet"
    )
    peer_rows = pd.read_parquet(
        PROJECT_ROOT / "artifacts" / "p21e" / "peer_geometry_rows.parquet"
    )
    saved_drift_metrics = pd.read_csv(
        PROJECT_ROOT / "artifacts" / "tables" / "p21e_drift_alert_metrics.csv"
    )
    saved_peer_metrics = pd.read_csv(
        PROJECT_ROOT / "artifacts" / "tables" / "p21e_peer_geometry_metrics.csv"
    )
    if drift_events.duplicated(
        ["scenario", "score", "gvkey", "datadate", "fyear"]
    ).any():
        errors.append("Duplicate saved P21E drift event identity.")
    if peer_rows.duplicated(
        ["scenario", "space", "gvkey", "datadate", "fyear"]
    ).any():
        errors.append("Duplicate saved P21E peer-row identity.")
    if not peer_rows["recall_at_k"].between(0.0, 1.0).all():
        errors.append("Saved P21E peer recall falls outside [0, 1].")
    if not peer_rows["random_recall"].between(0.0, 1.0).all():
        errors.append("Saved P21E random recall falls outside [0, 1].")

    recomputed_drift_metrics = summarize_drift_events(drift_events)
    _compare_frames(
        saved_drift_metrics,
        recomputed_drift_metrics,
        errors,
        "drift metrics",
    )
    coverage: dict[str, dict[str, int]] = {}
    for scenario in SCENARIOS:
        axes_path = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "synthetic"
            / scenario
            / "p3"
            / "axis_scores.parquet"
        )
        axes = pd.read_parquet(axes_path, columns=["fyear"])
        complete_rows = int(
            peer_rows.loc[
                peer_rows["scenario"].eq(scenario)
                & peer_rows["space"].eq("anchored")
            ].shape[0]
        )
        coverage[scenario] = {
            "p3_evaluation_rows": int(axes["fyear"].between(2013, 2018).sum()),
            "complete_rows": complete_rows,
        }
    recomputed_peer_metrics = summarize_peer_geometry(peer_rows, coverage)
    _compare_frames(
        saved_peer_metrics,
        recomputed_peer_metrics,
        errors,
        "peer metrics",
    )

    drift_result = drift_alert_gate(recomputed_drift_metrics)
    peer_result = peer_map_gate(recomputed_peer_metrics)
    crowding_result = crowding_gate(recomputed_peer_metrics)
    product_result = product_gate(
        drift_result,
        peer_result,
        crowding_result,
    )
    saved_results = report["use_case_results"]
    _compare_result(
        saved_results["strategic_drift_alerts"],
        drift_result,
        errors,
        "strategic drift result",
    )
    _compare_result(
        saved_results["structural_peer_discovery"],
        peer_result,
        errors,
        "peer-map result",
    )
    _compare_result(
        saved_results["competitive_crowding_state"],
        crowding_result,
        errors,
        "crowding result",
    )
    _compare_result(
        report["primary_result"],
        product_result,
        errors,
        "primary result",
    )

    required_false = (
        "model_fit",
        "return_data_read",
        "injected_return_alpha_read",
        "survival_label_read",
        "exit_category_read",
        "true_viability_read",
        "real_data_read",
        "portfolio_run",
        "separate_edgar_corpus_touched",
    )
    for field in required_false:
        if report.get(field) is not False:
            errors.append(f"P21E declaration is not false: {field}.")
    allowed_truth = {
        *(f"latent_axis_{index}" for index in range(1, 7)),
        "migration_surprise",
    }
    if set(report.get("synthetic_truth_columns_read", [])) != allowed_truth:
        errors.append("P21E synthetic-truth column declaration is invalid.")

    result = {
        "schema_version": 1,
        "phase": "P21E",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "drift_event_rows_checked": int(len(drift_events)),
        "peer_geometry_rows_checked": int(len(peer_rows)),
        "drift_metric_rows_checked": int(len(recomputed_drift_metrics)),
        "peer_metric_rows_checked": int(len(recomputed_peer_metrics)),
        "primary_verdict": product_result["verdict"],
        "source_records_checked": int(len(report["source_records"])),
        "output_records_checked": int(len(report["output_records"])),
    }
    atomic_write_json(PROJECT_ROOT / args.output, result)
    if errors:
        print("\n".join(errors))
        return 1
    print(
        "P21E_VALIDATION PASS",
        f"drift_rows={len(drift_events)}",
        f"peer_rows={len(peer_rows)}",
        f"verdict={product_result['verdict']}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
