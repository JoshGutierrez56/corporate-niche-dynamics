"""Leakage-safe incremental survival-utility audit for phase P20E."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from hypercube.config import HypercubeConfig
from hypercube.viability import (
    ANCHORED_AXES,
    DISTRESS_FEATURES,
    PROFITABILITY_FEATURES,
    RELATIVE_AXES,
    _apply_platt,
    _fit_platt,
    _fold_frames,
    _logistic_pipeline,
    fold_definitions,
    prediction_metrics,
)


P20E_VERSION = "hypercube-survival-utility-v1"
FINANCIAL_FEATURES = (*PROFITABILITY_FEATURES, *DISTRESS_FEATURES)
HYPERCUBE_FEATURES = (*RELATIVE_AXES, *ANCHORED_AXES)
MODEL_FEATURES = {
    "financial_baseline_logit": FINANCIAL_FEATURES,
    "financial_plus_hypercube_logit": (
        *FINANCIAL_FEATURES,
        *HYPERCUBE_FEATURES,
    ),
}
MODEL_ORDER = tuple(MODEL_FEATURES)


def _fit_one_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    features: tuple[str, ...],
    config: HypercubeConfig,
) -> tuple[np.ndarray, float, float]:
    """Fit one frozen ridge logit and validation-only calibrator."""

    y_train = train["failure_within_horizon"].astype(int).to_numpy()
    y_validation = validation["failure_within_horizon"].astype(int).to_numpy()
    candidates: list[tuple[float, float, Any]] = []
    for c_value in config.viability.ridge_c_grid:
        pipeline = _logistic_pipeline(features, c_value)
        pipeline.fit(train[list(features)], y_train)
        validation_raw = pipeline.predict_proba(validation[list(features)])[:, 1]
        score = float(brier_score_loss(y_validation, validation_raw))
        candidates.append((score, c_value, pipeline))
    validation_brier, selected_c, fitted = min(
        candidates, key=lambda item: (item[0], item[1])
    )
    validation_raw = fitted.predict_proba(validation[list(features)])[:, 1]
    test_raw = fitted.predict_proba(test[list(features)])[:, 1]
    calibrator = _fit_platt(validation_raw, y_validation)
    return _apply_platt(calibrator, test_raw), selected_c, validation_brier


def run_survival_utility(
    matrix: pd.DataFrame,
    config: HypercubeConfig,
    scenario: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the fixed P20E model pair on the original purged P4 folds."""

    required = {
        "permno",
        "gvkey",
        "datadate",
        "fyear",
        "feature_date",
        "horizon_end",
        "horizon_years",
        "failure_within_horizon",
        *FINANCIAL_FEATURES,
        *HYPERCUBE_FEATURES,
    }
    missing = sorted(required.difference(matrix.columns))
    if missing:
        raise ValueError(f"P20E model matrix is missing columns: {missing}")

    predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    identity = [
        "permno",
        "gvkey",
        "datadate",
        "fyear",
        "feature_date",
        "horizon_end",
        "horizon_years",
        "failure_within_horizon",
    ]
    for horizon in config.viability.horizons_years:
        horizon_matrix = matrix.loc[matrix["horizon_years"].eq(horizon)].copy()
        for definition in fold_definitions(config, horizon):
            train, validation, test = _fold_frames(horizon_matrix, definition)
            outcome = test["failure_within_horizon"].astype(int).to_numpy()
            fold_rows.append({"scenario": scenario, **definition})
            for model_name in MODEL_ORDER:
                probability, selected_c, validation_brier = _fit_one_model(
                    train,
                    validation,
                    test,
                    MODEL_FEATURES[model_name],
                    config,
                )
                saved = test[identity].copy()
                saved.insert(0, "scenario", scenario)
                saved["fold"] = int(definition["fold"])
                saved["model"] = model_name
                saved["predicted_failure_probability"] = probability
                saved["calibrated_survival_probability"] = 1.0 - probability
                predictions.append(saved)
                metrics.append(
                    {
                        "scenario": scenario,
                        "horizon_years": int(horizon),
                        "fold": int(definition["fold"]),
                        "model": model_name,
                        "train_rows": int(len(train)),
                        "train_failures": int(
                            train["failure_within_horizon"].astype(int).sum()
                        ),
                        "validation_rows": int(len(validation)),
                        "validation_failures": int(
                            validation["failure_within_horizon"].astype(int).sum()
                        ),
                        "test_rows": int(len(test)),
                        "test_failures": int(outcome.sum()),
                        "selected_c": float(selected_c),
                        "validation_brier_uncalibrated": validation_brier,
                        **prediction_metrics(outcome, probability),
                    }
                )
    return (
        pd.concat(predictions, ignore_index=True),
        pd.DataFrame(metrics),
        pd.DataFrame(fold_rows),
    )


def paired_comparisons(metrics: pd.DataFrame) -> pd.DataFrame:
    """Compare the augmented model with the conventional baseline by fold."""

    keys = ["scenario", "horizon_years", "fold"]
    selected = metrics.pivot(index=keys, columns="model")
    rows: list[dict[str, Any]] = []
    for key in selected.index:
        scenario, horizon, fold = key
        rows.append(
            {
                "scenario": scenario,
                "horizon_years": int(horizon),
                "fold": int(fold),
                "auc_improvement": float(
                    selected.loc[key, ("roc_auc", "financial_plus_hypercube_logit")]
                    - selected.loc[key, ("roc_auc", "financial_baseline_logit")]
                ),
                "average_precision_improvement": float(
                    selected.loc[
                        key,
                        ("average_precision", "financial_plus_hypercube_logit"),
                    ]
                    - selected.loc[
                        key, ("average_precision", "financial_baseline_logit")
                    ]
                ),
                "brier_improvement": float(
                    selected.loc[key, ("brier_score", "financial_baseline_logit")]
                    - selected.loc[
                        key, ("brier_score", "financial_plus_hypercube_logit")
                    ]
                ),
                "log_loss_improvement": float(
                    selected.loc[key, ("log_loss", "financial_baseline_logit")]
                    - selected.loc[
                        key, ("log_loss", "financial_plus_hypercube_logit")
                    ]
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(keys).reset_index(drop=True)


def utility_gate(comparisons: pd.DataFrame) -> dict[str, Any]:
    """Apply the frozen robust-incremental-utility decision rule."""

    grouped = comparisons.groupby(["scenario", "horizon_years"])[
        "auc_improvement"
    ].mean()
    mean_auc = float(comparisons["auc_improvement"].mean())
    auc_wins = int(comparisons["auc_improvement"].gt(0.0).sum())
    mean_brier = float(comparisons["brier_improvement"].mean())
    positive_cells = int(grouped.gt(0.0).sum())
    gates: Mapping[str, bool] = {
        "mean_auc_improvement_at_least_0_01": mean_auc >= 0.01,
        "auc_wins_at_least_16_of_24": auc_wins >= 16,
        "mean_brier_improvement_positive": mean_brier > 0.0,
        "all_scenario_horizon_cells_positive": positive_cells == 6,
    }
    passed = all(gates.values())
    return {
        "verdict": (
            "ROBUST_INCREMENTAL_UTILITY"
            if passed
            else "NO_ROBUST_INCREMENTAL_UTILITY"
        ),
        "gates": dict(gates),
        "mean_auc_improvement": mean_auc,
        "median_auc_improvement": float(
            comparisons["auc_improvement"].median()
        ),
        "auc_wins": auc_wins,
        "folds": int(len(comparisons)),
        "mean_brier_improvement": mean_brier,
        "brier_wins": int(comparisons["brier_improvement"].gt(0.0).sum()),
        "positive_scenario_horizon_cells": positive_cells,
        "scenario_horizon_cells": int(len(grouped)),
    }
