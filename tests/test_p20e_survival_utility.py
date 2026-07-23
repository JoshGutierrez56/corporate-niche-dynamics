"""P20E incremental-survival-utility contract tests."""

from __future__ import annotations

import pandas as pd

from hypercube.survival_utility import (
    FINANCIAL_FEATURES,
    HYPERCUBE_FEATURES,
    MODEL_FEATURES,
    paired_comparisons,
    utility_gate,
)


def _metrics(augmented_auc: float, augmented_brier: float) -> pd.DataFrame:
    rows = []
    for scenario in ("null_alpha", "migration_alpha", "regime_shift"):
        for horizon in (3, 5):
            for fold in range(1, 5):
                for model, auc, brier in (
                    ("financial_baseline_logit", 0.60, 0.10),
                    (
                        "financial_plus_hypercube_logit",
                        augmented_auc,
                        augmented_brier,
                    ),
                ):
                    rows.append(
                        {
                            "scenario": scenario,
                            "horizon_years": horizon,
                            "fold": fold,
                            "model": model,
                            "roc_auc": auc,
                            "average_precision": auc / 10.0,
                            "brier_score": brier,
                            "log_loss": brier * 2.0,
                        }
                    )
    return pd.DataFrame(rows)


def test_feature_contract_adds_axes_without_replacing_financials() -> None:
    assert set(FINANCIAL_FEATURES).isdisjoint(HYPERCUBE_FEATURES)
    assert MODEL_FEATURES["financial_baseline_logit"] == FINANCIAL_FEATURES
    assert set(MODEL_FEATURES["financial_plus_hypercube_logit"]) == {
        *FINANCIAL_FEATURES,
        *HYPERCUBE_FEATURES,
    }


def test_robust_gate_passes_only_consistent_material_improvement() -> None:
    comparisons = paired_comparisons(_metrics(0.62, 0.09))
    result = utility_gate(comparisons)
    assert result["verdict"] == "ROBUST_INCREMENTAL_UTILITY"
    assert result["auc_wins"] == 24
    assert result["positive_scenario_horizon_cells"] == 6


def test_robust_gate_rejects_small_improvement() -> None:
    comparisons = paired_comparisons(_metrics(0.605, 0.099))
    result = utility_gate(comparisons)
    assert result["verdict"] == "NO_ROBUST_INCREMENTAL_UTILITY"
    assert result["gates"]["mean_auc_improvement_at_least_0_01"] is False
