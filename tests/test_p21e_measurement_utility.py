"""P21E measurement-utility contract tests."""

from __future__ import annotations

import pandas as pd

from hypercube.measurement_utility import (
    BENCHMARK,
    PRIMARY_DRIFT_SCORE,
    crowding_gate,
    drift_alert_gate,
    peer_map_gate,
    product_gate,
)


def _drift_metrics(
    primary_precision: float,
    benchmark_precision: float,
) -> pd.DataFrame:
    rows = []
    for scope, value in (
        ("overall", "all"),
        ("block", "2013_2015"),
        ("block", "2016_2018"),
    ):
        for score, precision in (
            (PRIMARY_DRIFT_SCORE, primary_precision),
            (BENCHMARK, benchmark_precision),
        ):
            rows.append(
                {
                    "scope": scope,
                    "scope_value": value,
                    "score": score,
                    "precision": precision,
                    "recall": precision,
                    "sign_accuracy": 0.80,
                    "coverage": 0.97,
                }
            )
    return pd.DataFrame(rows)


def _peer_metrics(recall: float, crowding: float) -> pd.DataFrame:
    rows = []
    for scope, value in (
        ("overall", "all"),
        ("scenario", "null_alpha"),
        ("scenario", "migration_alpha"),
        ("scenario", "regime_shift"),
    ):
        rows.append(
            {
                "scope": scope,
                "scope_value": value,
                "space": "anchored",
                "coverage": 0.90,
                "mean_recall_at_20": recall,
                "mean_random_recall": 0.008,
                "recall_lift_vs_random": recall / 0.008,
                "crowding_spearman": crowding,
            }
        )
    return pd.DataFrame(rows)


def test_drift_gate_requires_material_benchmark_improvement() -> None:
    passed = drift_alert_gate(_drift_metrics(0.30, 0.20))
    assert passed["verdict"] == "SUPPORTED"
    failed = drift_alert_gate(_drift_metrics(0.30, 0.27))
    assert failed["verdict"] == "NOT_SUPPORTED"
    assert failed["gates"]["precision_improvement_at_least_0_05"] is False


def test_peer_and_crowding_gates_are_independent() -> None:
    peer_only = _peer_metrics(0.06, 0.05)
    assert peer_map_gate(peer_only)["verdict"] == "SUPPORTED"
    assert crowding_gate(peer_only)["verdict"] == "NOT_SUPPORTED"

    crowding_only = _peer_metrics(0.02, 0.25)
    assert peer_map_gate(crowding_only)["verdict"] == "NOT_SUPPORTED"
    assert crowding_gate(crowding_only)["verdict"] == "SUPPORTED"


def test_product_gate_requires_drift_and_one_structural_use() -> None:
    supported = {"verdict": "SUPPORTED"}
    rejected = {"verdict": "NOT_SUPPORTED"}
    result = product_gate(supported, rejected, supported)
    assert result["verdict"] == "MEASUREMENT_MONITOR_SUPPORTED"
    result = product_gate(rejected, supported, supported)
    assert result["verdict"] == "RESEARCH_BENCHMARK_ONLY"
