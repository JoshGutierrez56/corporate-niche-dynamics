"""Run the P21E measurement-utility and product-closeout audit."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence

import pandas as pd

from hypercube.data import SCENARIOS, atomic_write_json, sha256_file
from hypercube.measurement_utility import (
    LATENT_AXES,
    P21E_VERSION,
    build_drift_events,
    build_peer_geometry,
    crowding_gate,
    drift_alert_gate,
    peer_map_gate,
    product_gate,
    summarize_drift_events,
    summarize_peer_geometry,
)
from hypercube.proxy_redesign import build_proxy_candidates


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def _use_case_matrix(
    drift: dict[str, Any],
    peer: dict[str, Any],
    crowding: dict[str, Any],
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "use_case": "strategic_drift_alerts",
                "status": drift["verdict"],
                "primary_evidence": (
                    f"extreme precision={drift['overall_precision']:.6f}; "
                    f"sign accuracy={drift['overall_sign_accuracy']:.6f}"
                ),
                "allowed_claim": (
                    "synthetic strategic-change measurement and alert ranking"
                ),
                "prohibited_claim": "real-world event prediction or causality",
            },
            {
                "use_case": "structural_peer_discovery",
                "status": peer["verdict"],
                "primary_evidence": (
                    f"Recall@20={peer['mean_recall_at_20']:.6f}; "
                    f"random lift={peer['recall_lift_vs_random']:.3f}x"
                ),
                "allowed_claim": "synthetic latent-neighborhood recovery",
                "prohibited_claim": "validated real-company comparable selection",
            },
            {
                "use_case": "competitive_crowding_state",
                "status": crowding["verdict"],
                "primary_evidence": (
                    f"latent-density Spearman="
                    f"{crowding['crowding_spearman']:.6f}"
                ),
                "allowed_claim": "synthetic local-density measurement",
                "prohibited_claim": "validated competitive threat forecast",
            },
            {
                "use_case": "regime_stress_sensing",
                "status": "EXPLORATORY",
                "primary_evidence": (
                    "P20E regime-shift survival AUC lift was positive in 7/8 folds"
                ),
                "allowed_claim": "candidate for a separately frozen evaluation",
                "prohibited_claim": "general survival or distress prediction",
            },
            {
                "use_case": "return_prediction",
                "status": "REJECTED",
                "primary_evidence": "P19E primary net return=-5.16%",
                "allowed_claim": "negative-result benchmark",
                "prohibited_claim": "alpha or investability",
            },
            {
                "use_case": "general_survival_prediction",
                "status": "REJECTED",
                "primary_evidence": "P20E mean incremental AUC=-0.002732",
                "allowed_claim": "negative-result benchmark",
                "prohibited_claim": "firm-survival prediction",
            },
            {
                "use_case": "economic_archetype_taxonomy",
                "status": "REJECTED",
                "primary_evidence": "P18E noise/unassigned=94.68%-96.50%",
                "allowed_claim": "negative-result benchmark",
                "prohibited_claim": "stable economic taxonomy",
            },
            {
                "use_case": "synthetic_method_benchmark",
                "status": "SUPPORTED",
                "primary_evidence": "signed, independently reproducible phase chain",
                "allowed_claim": "auditable synthetic research benchmark",
                "prohibited_claim": "real-data validation",
            },
        ]
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/p21e"),
    )
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("artifacts/tables"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/manifests/p21e_measurement_utility.json"),
    )
    args = parser.parse_args(argv)

    source_paths = [
        PROJECT_ROOT / "docs" / "p21e_measurement_utility_protocol.md",
        PROJECT_ROOT / "artifacts" / "tables" / "p5_recovery_metrics.csv",
        PROJECT_ROOT
        / "artifacts"
        / "manifests"
        / "p14f_p7_eligibility_audit.json",
        PROJECT_ROOT
        / "artifacts"
        / "manifests"
        / "p20e_survival_utility.json",
    ]
    for scenario in SCENARIOS:
        source_paths.extend(
            [
                PROJECT_ROOT
                / "data"
                / "processed"
                / "synthetic"
                / scenario
                / "p3"
                / "axis_scores.parquet",
                PROJECT_ROOT
                / "data"
                / "processed"
                / "synthetic"
                / scenario
                / "p5"
                / "frontier_dynamics.parquet",
                PROJECT_ROOT
                / "data"
                / "raw"
                / "synthetic"
                / scenario
                / "synthetic_truth.parquet",
            ]
        )
    for path in source_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    source_records = [_record(path) for path in source_paths]

    drift_frames: list[pd.DataFrame] = []
    peer_frames: list[pd.DataFrame] = []
    edge_frames: list[pd.DataFrame] = []
    coverage: dict[str, dict[str, int]] = {}
    for scenario in SCENARIOS:
        p3_path = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "synthetic"
            / scenario
            / "p3"
            / "axis_scores.parquet"
        )
        p5_path = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "synthetic"
            / scenario
            / "p5"
            / "frontier_dynamics.parquet"
        )
        truth_path = (
            PROJECT_ROOT
            / "data"
            / "raw"
            / "synthetic"
            / scenario
            / "synthetic_truth.parquet"
        )
        candidates, _, _ = build_proxy_candidates(p5_path)
        drift_truth = pd.read_parquet(
            truth_path,
            columns=["gvkey", "datadate", "fyear", "migration_surprise"],
        )
        drift_frames.append(
            build_drift_events(candidates, drift_truth, scenario)
        )

        axes = pd.read_parquet(p3_path)
        peer_truth = pd.read_parquet(
            truth_path,
            columns=["gvkey", "datadate", "fyear", *LATENT_AXES],
        )
        scenario_rows, scenario_edges, scenario_coverage = build_peer_geometry(
            axes,
            peer_truth,
            scenario,
        )
        peer_frames.append(scenario_rows)
        edge_frames.append(scenario_edges)
        coverage[scenario] = scenario_coverage

    drift_events = pd.concat(drift_frames, ignore_index=True)
    drift_metrics = summarize_drift_events(drift_events)
    drift_result = drift_alert_gate(drift_metrics)
    peer_rows = pd.concat(peer_frames, ignore_index=True)
    peer_examples = pd.concat(edge_frames, ignore_index=True)
    peer_metrics = summarize_peer_geometry(peer_rows, coverage)
    peer_result = peer_map_gate(peer_metrics)
    crowding_result = crowding_gate(peer_metrics)
    product_result = product_gate(
        drift_result,
        peer_result,
        crowding_result,
    )
    use_cases = _use_case_matrix(
        drift_result,
        peer_result,
        crowding_result,
    )

    output_dir = PROJECT_ROOT / args.output_dir
    tables_dir = PROJECT_ROOT / args.tables_dir
    output_paths = {
        "drift_events": output_dir / "drift_alert_events.parquet",
        "peer_rows": output_dir / "peer_geometry_rows.parquet",
        "peer_examples": output_dir / "peer_examples.parquet",
        "drift_metrics": tables_dir / "p21e_drift_alert_metrics.csv",
        "peer_metrics": tables_dir / "p21e_peer_geometry_metrics.csv",
        "use_cases": tables_dir / "p21e_use_case_matrix.csv",
    }
    _atomic_parquet(drift_events, output_paths["drift_events"])
    _atomic_parquet(peer_rows, output_paths["peer_rows"])
    _atomic_parquet(peer_examples, output_paths["peer_examples"])
    _atomic_csv(drift_metrics, output_paths["drift_metrics"])
    _atomic_csv(peer_metrics, output_paths["peer_metrics"])
    _atomic_csv(use_cases, output_paths["use_cases"])

    if [_record(path) for path in source_paths] != source_records:
        raise RuntimeError("A frozen P21E source changed during the audit.")
    report = {
        "schema_version": 1,
        "phase": "P21E",
        "version": P21E_VERSION,
        "status": "PASS",
        "primary_result": product_result,
        "use_case_results": {
            "strategic_drift_alerts": drift_result,
            "structural_peer_discovery": peer_result,
            "competitive_crowding_state": crowding_result,
        },
        "rows": {
            "drift_event_rows": int(len(drift_events)),
            "drift_metric_rows": int(len(drift_metrics)),
            "peer_geometry_rows": int(len(peer_rows)),
            "peer_example_edges": int(len(peer_examples)),
            "peer_metric_rows": int(len(peer_metrics)),
            "use_case_rows": int(len(use_cases)),
        },
        "scenarios": list(SCENARIOS),
        "evaluation_years": [2013, 2018],
        "neighbors": 20,
        "model_fit": False,
        "synthetic_truth_columns_read": [
            *LATENT_AXES,
            "migration_surprise",
        ],
        "return_data_read": False,
        "injected_return_alpha_read": False,
        "survival_label_read": False,
        "exit_category_read": False,
        "true_viability_read": False,
        "real_data_read": False,
        "portfolio_run": False,
        "separate_edgar_corpus_touched": False,
        "source_records": source_records,
        "output_records": [
            _record(path) for path in output_paths.values()
        ],
    }
    atomic_write_json(PROJECT_ROOT / args.report, report)
    print(
        "P21E",
        report["status"],
        product_result["verdict"],
        f"drift={drift_result['verdict']}",
        f"peer={peer_result['verdict']}",
        f"crowding={crowding_result['verdict']}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
