"""Nested time-split continuous viability models for phase P4."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Callable, Mapping

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import sklearn
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from hypercube.axes import scenario_p3_dir, validate_p3_directory
from hypercube.config import HypercubeConfig
from hypercube.data import Scenario, atomic_write_json, sha256_file
from hypercube.labels import LABEL_KEY, construct_fixed_horizon_labels


class ViabilityModelError(ValueError):
    """Raised when a P4 label, fold, or model gate cannot be satisfied."""


AXIS_SLUGS = (
    "demand_strength_pricing_power",
    "competitive_defensibility",
    "innovation_intensity",
    "go_to_market_efficiency",
    "unit_economics_profit_quality",
    "scalability_capital_efficiency",
)
RELATIVE_AXES = tuple(f"relative_{item}" for item in AXIS_SLUGS)
ANCHORED_AXES = tuple(f"anchored_{item}" for item in AXIS_SLUGS)
PROFITABILITY_FEATURES = (
    "operating_margin",
    "gross_profitability_lag_assets",
)
DISTRESS_FEATURES = (
    "log_market_cap",
    "book_leverage",
    "working_capital_assets",
    "operating_income_assets",
    "equity_to_liabilities",
    "sales_to_assets",
)
MODEL_ORDER = (
    "profitability_logit",
    "distress_logit",
    "industry_rate",
    "occupied_cell_rate",
    "relative_axes_logit",
    "combined_axes_logit",
    "hist_gradient_boosting",
)
SIMPLE_BENCHMARKS = MODEL_ORDER[:4]
P4_VERSION = "hypercube-viability-v1"


def scenario_p4_dir(config: HypercubeConfig, scenario: Scenario) -> Path:
    """Resolve one scenario's P4 output directory."""

    return scenario_p3_dir(config, scenario).parent / "p4"


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide finite values while treating zero denominators as unavailable."""

    result = pd.to_numeric(numerator, errors="coerce") / pd.to_numeric(
        denominator, errors="coerce"
    ).replace(0.0, np.nan)
    return result.where(np.isfinite(result))


def build_model_matrix(
    p2_dir: Path,
    p3_dir: Path,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Join P3 axes/components to transparent distress controls and labels."""

    axes = pd.read_parquet(p3_dir / "axis_scores.parquet")
    components = pd.read_parquet(p3_dir / "component_features.parquet")
    accounting = pd.read_parquet(p2_dir / "accounting_availability.parquet")
    keys = ["gvkey", "datadate", "fyear"]
    component_columns = [*keys, *PROFITABILITY_FEATURES]
    axis_columns = [
        *keys,
        "sic2",
        "market_cap_millions",
        *RELATIVE_AXES,
        *ANCHORED_AXES,
    ]
    accounting_columns = [
        *keys,
        "at",
        "dltt",
        "dlc",
        "act",
        "lct",
        "oiadp",
        "ceq",
        "sale",
    ]
    events = axes[axis_columns].merge(
        components[component_columns], on=keys, how="left", validate="one_to_one"
    )
    events = events.merge(
        accounting[accounting_columns], on=keys, how="left", validate="one_to_one"
    )
    liabilities = events["at"] - events["ceq"]
    events["log_market_cap"] = np.log1p(
        pd.to_numeric(events["market_cap_millions"], errors="coerce").clip(lower=0.0)
    )
    events["book_leverage"] = _safe_ratio(events["dltt"] + events["dlc"], events["at"])
    events["working_capital_assets"] = _safe_ratio(
        events["act"] - events["lct"], events["at"]
    )
    events["operating_income_assets"] = _safe_ratio(events["oiadp"], events["at"])
    events["equity_to_liabilities"] = _safe_ratio(events["ceq"], liabilities)
    events["sales_to_assets"] = _safe_ratio(events["sale"], events["at"])
    keep = [
        *keys,
        "sic2",
        "market_cap_millions",
        *PROFITABILITY_FEATURES,
        *DISTRESS_FEATURES,
        *RELATIVE_AXES,
        *ANCHORED_AXES,
    ]
    matrix = labels.merge(events[keep], on=keys, how="left", validate="many_to_one")
    matrix["event_year"] = pd.to_datetime(matrix["feature_date"]).dt.year.astype(int)
    feature_columns = [
        *PROFITABILITY_FEATURES,
        *DISTRESS_FEATURES,
        *RELATIVE_AXES,
        *ANCHORED_AXES,
    ]
    values = matrix[feature_columns].to_numpy(dtype=float)
    if np.isinf(values).any():
        raise ViabilityModelError("P4 model matrix contains infinite features.")
    return matrix.sort_values(list(LABEL_KEY)).reset_index(drop=True)


def fold_definitions(config: HypercubeConfig, horizon: int) -> list[dict[str, int]]:
    """Create expanding, purged nested calendar folds for one horizon."""

    definitions: list[dict[str, int]] = []
    for index, outer in enumerate(config.viability.outer_folds, start=1):
        validation_end = outer.start_year - horizon - 1
        validation_start = validation_end - config.viability.validation_years + 1
        train_end = validation_start - horizon - 1
        definitions.append(
            {
                "fold": index,
                "horizon_years": horizon,
                "train_start_year": config.synthetic.start_year,
                "train_end_year": train_end,
                "validation_start_year": validation_start,
                "validation_end_year": validation_end,
                "test_start_year": outer.start_year,
                "test_end_year": outer.end_year,
                "purge_years": horizon,
            }
        )
    return definitions


def _fold_frames(
    matrix: pd.DataFrame,
    definition: Mapping[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select observed labels and enforce label-availability cutoffs."""

    observed = matrix.loc[matrix["failure_within_horizon"].notna()].copy()
    train = observed.loc[
        observed["event_year"].between(
            definition["train_start_year"], definition["train_end_year"]
        )
    ].copy()
    validation = observed.loc[
        observed["event_year"].between(
            definition["validation_start_year"], definition["validation_end_year"]
        )
    ].copy()
    test = observed.loc[
        observed["event_year"].between(
            definition["test_start_year"], definition["test_end_year"]
        )
    ].copy()
    validation_cutoff = pd.Timestamp(definition["validation_start_year"], 1, 1)
    test_cutoff = pd.Timestamp(definition["test_start_year"], 1, 1)
    if not (pd.to_datetime(train["label_observed_date"]) < validation_cutoff).all():
        raise ViabilityModelError("Training labels were not known before validation.")
    if not (pd.to_datetime(validation["label_observed_date"]) < test_cutoff).all():
        raise ViabilityModelError("Validation labels were not known before outer test.")
    for name, frame in (("train", train), ("validation", validation), ("test", test)):
        if frame.empty:
            raise ViabilityModelError(f"Empty {name} split: {definition}")
    return train, validation, test


def _logistic_pipeline(features: tuple[str, ...], c_value: float) -> Pipeline:
    """Create a train-only imputation, scaling, and ridge-logit pipeline."""

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=c_value,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=0,
                ),
            ),
        ]
    )


def _tree_pipeline(config: HypercubeConfig) -> Pipeline:
    """Create the sole constrained nonlinear benchmark."""

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=config.viability.tree_max_iter,
                    max_leaf_nodes=config.viability.tree_max_leaf_nodes,
                    learning_rate=config.viability.tree_learning_rate,
                    min_samples_leaf=config.viability.tree_min_samples_leaf,
                    l2_regularization=config.viability.tree_l2_regularization,
                    random_state=config.project.seed,
                ),
            ),
        ]
    )


def _bounded(probability: np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(probability, dtype=float), 1e-6, 1.0 - 1e-6)


def _fit_platt(raw_probability: np.ndarray, outcome: np.ndarray) -> dict[str, Any]:
    """Fit a one-dimensional sigmoid calibrator on the inner validation set."""

    probability = _bounded(raw_probability)
    y = np.asarray(outcome, dtype=int)
    if np.unique(y).size < 2:
        return {"kind": "identity", "model": None, "intercept": 0.0, "slope": 1.0}
    logit = np.log(probability / (1.0 - probability)).reshape(-1, 1)
    model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    model.fit(logit, y)
    return {
        "kind": "platt",
        "model": model,
        "intercept": float(model.intercept_[0]),
        "slope": float(model.coef_[0, 0]),
    }


def _apply_platt(calibrator: Mapping[str, Any], raw_probability: np.ndarray) -> np.ndarray:
    probability = _bounded(raw_probability)
    if calibrator["kind"] == "identity":
        return probability
    logit = np.log(probability / (1.0 - probability)).reshape(-1, 1)
    return _bounded(calibrator["model"].predict_proba(logit)[:, 1])


def _calibration_diagnostics(
    outcome: np.ndarray, probability: np.ndarray
) -> tuple[float, float, float]:
    """Return test calibration intercept, slope, and 10-bin ECE."""

    y = np.asarray(outcome, dtype=int)
    p = _bounded(probability)
    intercept = math.nan
    slope = math.nan
    if np.unique(y).size == 2:
        logit = np.log(p / (1.0 - p)).reshape(-1, 1)
        model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
        model.fit(logit, y)
        intercept = float(model.intercept_[0])
        slope = float(model.coef_[0, 0])
    bins = pd.qcut(pd.Series(p), q=min(10, len(p)), duplicates="drop")
    calibration = pd.DataFrame({"y": y, "p": p, "bin": bins}).groupby(
        "bin", observed=True
    ).agg(n=("y", "size"), actual=("y", "mean"), predicted=("p", "mean"))
    ece = float(
        (
            calibration["n"]
            / calibration["n"].sum()
            * (calibration["actual"] - calibration["predicted"]).abs()
        ).sum()
    )
    return intercept, slope, ece


def prediction_metrics(outcome: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    """Compute discrimination and calibration metrics from saved probabilities."""

    y = np.asarray(outcome, dtype=int)
    p = _bounded(probability)
    intercept, slope, ece = _calibration_diagnostics(y, p)
    spearman = spearmanr(y, p).statistic if len(y) > 1 else math.nan
    return {
        "roc_auc": float(roc_auc_score(y, p)) if np.unique(y).size == 2 else math.nan,
        "average_precision": float(average_precision_score(y, p)),
        "brier_score": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "spearman": float(spearman),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "expected_calibration_error": ece,
    }


def _fit_rate_model(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    group_builder: Callable[[pd.DataFrame, pd.DataFrame | None], tuple[pd.Series, Any]],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit a smoothed historical rate using a train-defined grouping contract."""

    train_group, state = group_builder(train, None)
    validation_group, _ = group_builder(validation, state)
    test_group, _ = group_builder(test, state)
    y = train["failure_within_horizon"].astype(int)
    global_rate = float(y.mean())
    table = pd.DataFrame({"group": train_group, "y": y}).groupby("group")["y"].agg(
        ["sum", "count"]
    )
    table["probability"] = (table["sum"] + 20.0 * global_rate) / (
        table["count"] + 20.0
    )
    validation_probability = validation_group.map(table["probability"]).fillna(global_rate)
    test_probability = test_group.map(table["probability"]).fillna(global_rate)
    fitted = {
        "global_rate": global_rate,
        "rates": {str(key): float(value) for key, value in table["probability"].items()},
        "state": state,
    }
    return (
        validation_probability.to_numpy(dtype=float),
        test_probability.to_numpy(dtype=float),
        fitted,
    )


def _industry_groups(
    frame: pd.DataFrame, state: Any
) -> tuple[pd.Series, dict[str, Any]]:
    groups = frame["sic2"].astype("Int64").astype("string").fillna("MISSING")
    return groups, {}


def _cell_groups(
    frame: pd.DataFrame, state: Any
) -> tuple[pd.Series, dict[str, Any]]:
    """Assign training-tertile 3^6 cells without using future cut points."""

    if state is None:
        medians = {
            feature: float(pd.to_numeric(frame[feature], errors="coerce").median())
            for feature in RELATIVE_AXES
        }
        edges = {}
        for feature in RELATIVE_AXES:
            values = pd.to_numeric(frame[feature], errors="coerce").fillna(
                medians[feature]
            )
            edges[feature] = [float(item) for item in values.quantile([1 / 3, 2 / 3])]
        state = {"medians": medians, "edges": edges}
    code = np.zeros(len(frame), dtype=int)
    multiplier = 1
    for feature in RELATIVE_AXES:
        values = pd.to_numeric(frame[feature], errors="coerce").fillna(
            state["medians"][feature]
        )
        bins = np.searchsorted(np.asarray(state["edges"][feature]), values, side="right")
        code += bins * multiplier
        multiplier *= 3
    return pd.Series(code.astype(str), index=frame.index, dtype="string"), state


def _fit_one_model(
    name: str,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    config: HypercubeConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], list[dict[str, Any]]]:
    """Fit one frozen model using only inner train/validation information."""

    y_train = train["failure_within_horizon"].astype(int).to_numpy()
    y_validation = validation["failure_within_horizon"].astype(int).to_numpy()
    trials: list[dict[str, Any]] = []
    if name in {"profitability_logit", "distress_logit", "relative_axes_logit", "combined_axes_logit"}:
        features = {
            "profitability_logit": PROFITABILITY_FEATURES,
            "distress_logit": DISTRESS_FEATURES,
            "relative_axes_logit": RELATIVE_AXES,
            "combined_axes_logit": (*RELATIVE_AXES, *ANCHORED_AXES),
        }[name]
        candidates: list[tuple[float, float, Pipeline]] = []
        for c_value in config.viability.ridge_c_grid:
            pipeline = _logistic_pipeline(features, c_value)
            pipeline.fit(train[list(features)], y_train)
            validation_raw = pipeline.predict_proba(validation[list(features)])[:, 1]
            score = float(brier_score_loss(y_validation, validation_raw))
            trials.append(
                {
                    "model": name,
                    "candidate": f"C={c_value:g}",
                    "validation_brier": score,
                }
            )
            candidates.append((score, c_value, pipeline))
        _, selected_c, fitted = min(candidates, key=lambda item: (item[0], item[1]))
        validation_raw = fitted.predict_proba(validation[list(features)])[:, 1]
        test_raw = fitted.predict_proba(test[list(features)])[:, 1]
        metadata: dict[str, Any] = {
            "kind": "ridge_logistic",
            "features": list(features),
            "selected_c": selected_c,
            "estimator": fitted,
        }
    elif name == "hist_gradient_boosting":
        features = (*RELATIVE_AXES, *ANCHORED_AXES, *DISTRESS_FEATURES)
        fitted = _tree_pipeline(config)
        fitted.fit(train[list(features)], y_train)
        validation_raw = fitted.predict_proba(validation[list(features)])[:, 1]
        test_raw = fitted.predict_proba(test[list(features)])[:, 1]
        trials.append(
            {
                "model": name,
                "candidate": "fixed_constrained_tree",
                "validation_brier": float(
                    brier_score_loss(y_validation, validation_raw)
                ),
            }
        )
        metadata = {
            "kind": "hist_gradient_boosting",
            "features": list(features),
            "selected_c": None,
            "estimator": fitted,
        }
    elif name == "industry_rate":
        validation_raw, test_raw, fitted = _fit_rate_model(
            train, validation, test, _industry_groups
        )
        trials.append(
            {
                "model": name,
                "candidate": "sic2_smoothed_rate",
                "validation_brier": float(
                    brier_score_loss(y_validation, validation_raw)
                ),
            }
        )
        metadata = {"kind": "historical_rate", "selected_c": None, "estimator": fitted}
    elif name == "occupied_cell_rate":
        validation_raw, test_raw, fitted = _fit_rate_model(
            train, validation, test, _cell_groups
        )
        trials.append(
            {
                "model": name,
                "candidate": "training_tertile_729_cell_rate",
                "validation_brier": float(
                    brier_score_loss(y_validation, validation_raw)
                ),
            }
        )
        metadata = {"kind": "historical_rate", "selected_c": None, "estimator": fitted}
    else:
        raise ViabilityModelError(f"Unknown P4 model: {name}")

    calibrator = _fit_platt(validation_raw, y_validation)
    validation_probability = _apply_platt(calibrator, validation_raw)
    test_probability = _apply_platt(calibrator, test_raw)
    metadata["calibrator"] = calibrator
    metadata["validation_brier_calibrated"] = float(
        brier_score_loss(y_validation, validation_probability)
    )
    return validation_probability, test_probability, metadata, trials


def _atomic_joblib(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    joblib.dump(payload, temporary, compress=3)
    os.replace(temporary, path)


def run_nested_viability(
    matrix: pd.DataFrame,
    config: HypercubeConfig,
    models_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, int]]]:
    """Run the frozen nested expanding-window model ladder."""

    predictions: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    definitions: list[dict[str, int]] = []
    models_dir.mkdir(parents=True, exist_ok=True)
    prediction_identity = [
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
        horizon_matrix = matrix.loc[matrix["horizon_years"] == horizon].copy()
        for definition in fold_definitions(config, horizon):
            definitions.append(definition)
            train, validation, test = _fold_frames(horizon_matrix, definition)
            y_train = train["failure_within_horizon"].astype(int)
            y_validation = validation["failure_within_horizon"].astype(int)
            if len(train) < config.viability.minimum_train_observations:
                raise ViabilityModelError(f"Too few training rows: {definition}")
            if int(y_train.sum()) < config.viability.minimum_train_failures:
                raise ViabilityModelError(f"Too few training failures: {definition}")
            if int(y_validation.sum()) < config.viability.minimum_validation_failures:
                raise ViabilityModelError(f"Too few validation failures: {definition}")
            for model_name in MODEL_ORDER:
                _, test_probability, fitted, model_trials = _fit_one_model(
                    model_name, train, validation, test, config
                )
                artifact_name = f"h{horizon}_fold{definition['fold']}_{model_name}.joblib"
                _atomic_joblib(
                    models_dir / artifact_name,
                    {
                        "version": P4_VERSION,
                        "scenario": config.data.scenario,
                        "horizon_years": horizon,
                        "fold": definition,
                        "model": model_name,
                        "fitted": fitted,
                        "sklearn_version": sklearn.__version__,
                    },
                )
                prediction = test[prediction_identity].copy()
                prediction["fold"] = definition["fold"]
                prediction["model"] = model_name
                prediction["predicted_failure_probability"] = test_probability
                prediction["calibrated_survival_probability"] = 1.0 - test_probability
                predictions.append(prediction)
                calculated = prediction_metrics(
                    prediction["failure_within_horizon"].astype(int).to_numpy(),
                    prediction["predicted_failure_probability"].to_numpy(),
                )
                metrics.append(
                    {
                        "horizon_years": horizon,
                        "fold": definition["fold"],
                        "model": model_name,
                        "train_rows": int(len(train)),
                        "train_failures": int(y_train.sum()),
                        "validation_rows": int(len(validation)),
                        "validation_failures": int(y_validation.sum()),
                        "test_rows": int(len(test)),
                        "test_failures": int(
                            test["failure_within_horizon"].astype(int).sum()
                        ),
                        "selected_c": fitted.get("selected_c"),
                        "calibrator_kind": fitted["calibrator"]["kind"],
                        "calibrator_intercept": fitted["calibrator"]["intercept"],
                        "calibrator_slope": fitted["calibrator"]["slope"],
                        **calculated,
                    }
                )
                for trial in model_trials:
                    trials.append(
                        {
                            "horizon_years": horizon,
                            "fold": definition["fold"],
                            **trial,
                        }
                    )
    return (
        pd.concat(predictions, ignore_index=True),
        pd.DataFrame(metrics),
        pd.DataFrame(trials),
        definitions,
    )


def benchmark_comparisons(metrics: pd.DataFrame) -> pd.DataFrame:
    """Compare the primary continuous model against every simple benchmark."""

    primary = metrics.loc[metrics["model"] == "combined_axes_logit"]
    rows: list[dict[str, Any]] = []
    for benchmark in SIMPLE_BENCHMARKS:
        other = metrics.loc[metrics["model"] == benchmark]
        merged = primary.merge(
            other,
            on=["horizon_years", "fold"],
            suffixes=("_primary", "_benchmark"),
            validate="one_to_one",
        )
        for row in merged.itertuples(index=False):
            rows.append(
                {
                    "horizon_years": row.horizon_years,
                    "fold": row.fold,
                    "primary_model": "combined_axes_logit",
                    "benchmark_model": benchmark,
                    "auc_difference": row.roc_auc_primary - row.roc_auc_benchmark,
                    "brier_improvement": row.brier_score_benchmark
                    - row.brier_score_primary,
                    "log_loss_improvement": row.log_loss_benchmark
                    - row.log_loss_primary,
                }
            )
    return pd.DataFrame(rows)


def _inventory(directory: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name == "p4_manifest.json":
            continue
        record: dict[str, Any] = {
            "name": path.relative_to(directory).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if path.suffix == ".parquet":
            record["rows"] = int(pq.ParquetFile(path).metadata.num_rows)
        records.append(record)
    return records


def build_viability_bundle(
    p2_dir: Path,
    p3_dir: Path,
    raw_dir: Path,
    output_dir: Path,
    config: HypercubeConfig,
    *,
    scenario: str,
) -> dict[str, Any]:
    """Build and atomically publish one complete P4 viability bundle."""

    if config.project.phase != "P4":
        raise ViabilityModelError("Viability construction requires a P4 config.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite completed P4 output: {output_dir}")
    p3_validation = validate_p3_directory(p3_dir, config)
    if p3_validation["status"] != "PASS":
        raise ViabilityModelError(f"P3 input validation failed: {p3_validation['errors']}")
    axes = pd.read_parquet(p3_dir / "axis_scores.parquet")
    labels = construct_fixed_horizon_labels(
        axes, raw_dir, config.viability.horizons_years
    )
    matrix = build_model_matrix(p2_dir, p3_dir, labels)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.p4-", dir=output_dir.parent))
    try:
        predictions, metrics, trials, definitions = run_nested_viability(
            matrix, config, staging / "models"
        )
        comparisons = benchmark_comparisons(metrics)
        labels.to_parquet(staging / "viability_labels.parquet", index=False, compression="zstd")
        matrix.to_parquet(staging / "model_matrix.parquet", index=False, compression="zstd")
        predictions.to_parquet(
            staging / "oos_predictions.parquet", index=False, compression="zstd"
        )
        metrics.to_csv(staging / "fold_metrics.csv", index=False, lineterminator="\n")
        trials.to_csv(
            staging / "hyperparameter_trials.csv", index=False, lineterminator="\n"
        )
        comparisons.to_csv(
            staging / "benchmark_comparisons.csv", index=False, lineterminator="\n"
        )
        atomic_write_json(staging / "fold_definitions.json", {"folds": definitions})
        label_summary = (
            labels.groupby(["horizon_years", "label_status", "censor_reason"], dropna=False)
            .size()
            .rename("rows")
            .reset_index()
        )
        label_summary.to_csv(staging / "label_summary.csv", index=False, lineterminator="\n")
        diagnostics: dict[str, Any] = {
            "status": "PASS",
            "phase": "P4",
            "scenario": scenario,
            "version": P4_VERSION,
            "rows": {
                "events": int(axes.shape[0]),
                "labels": int(labels.shape[0]),
                "observed_labels": int(labels["failure_within_horizon"].notna().sum()),
                "model_matrix": int(matrix.shape[0]),
                "oos_predictions": int(predictions.shape[0]),
            },
            "failures": {
                str(horizon): int(
                    labels.loc[labels["horizon_years"] == horizon, "failure_within_horizon"]
                    .eq(1)
                    .sum()
                )
                for horizon in config.viability.horizons_years
            },
            "models_fitted": list(MODEL_ORDER),
            "horizons_years": list(config.viability.horizons_years),
            "outer_folds": definitions,
            "synthetic_truth_read": False,
            "return_test_run": False,
            "portfolio_run": False,
            "sklearn_version": sklearn.__version__,
        }
        atomic_write_json(staging / "p4_diagnostics.json", diagnostics)
        atomic_write_json(staging / "resolved_config.json", config.model_dump(mode="json"))
        metadata = {
            "version": P4_VERSION,
            "outcome": "performance-related delisting within fixed horizon",
            "competing_exit_policy": "censor merger, voluntary/administrative, and other/unknown exits",
            "right_censor_policy": "require raw linked security follow-up through horizon",
            "primary_model": config.viability.primary_model,
            "model_order": list(MODEL_ORDER),
            "simple_benchmarks": list(SIMPLE_BENCHMARKS),
            "ridge_c_grid": list(config.viability.ridge_c_grid),
            "selection_metric": "inner-validation Brier score",
            "calibration": "Platt sigmoid fitted on inner validation only",
            "nonlinear_search": "none; one fixed constrained histogram gradient booster",
            "occupied_cell_role": "secondary 3^6 training-tertile density benchmark",
            "synthetic_truth_read": False,
            "return_test_run": False,
        }
        atomic_write_json(staging / "modeling_metadata.json", metadata)
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "phase": "P4",
            "scenario": scenario,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": config.project.seed,
            "inputs": [
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "rows": int(pq.ParquetFile(path).metadata.num_rows),
                }
                for path in (
                    p2_dir / "accounting_availability.parquet",
                    p3_dir / "component_features.parquet",
                    p3_dir / "axis_scores.parquet",
                    raw_dir / "crsp_monthly.parquet",
                    raw_dir / "crsp_delist.parquet",
                    raw_dir / "ccm_link.parquet",
                )
            ],
            "files": _inventory(staging),
            "synthetic_truth_read": False,
        }
        atomic_write_json(staging / "p4_manifest.json", manifest)
        if output_dir.exists():
            output_dir.rmdir()
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return diagnostics


def validate_p4_directory(
    output_dir: Path,
    p2_dir: Path,
    p3_dir: Path,
    raw_dir: Path,
    config: HypercubeConfig,
) -> dict[str, Any]:
    """Independently reopen, recompute, and validate a saved P4 bundle."""

    required = (
        "viability_labels.parquet",
        "model_matrix.parquet",
        "oos_predictions.parquet",
        "fold_metrics.csv",
        "hyperparameter_trials.csv",
        "benchmark_comparisons.csv",
        "fold_definitions.json",
        "label_summary.csv",
        "p4_diagnostics.json",
        "resolved_config.json",
        "modeling_metadata.json",
        "p4_manifest.json",
    )
    missing = [name for name in required if not (output_dir / name).is_file()]
    if missing:
        return {"status": "FAIL", "errors": [f"Missing P4 outputs: {missing}"], "warnings": []}
    errors: list[str] = []
    labels = pd.read_parquet(output_dir / "viability_labels.parquet")
    matrix = pd.read_parquet(output_dir / "model_matrix.parquet")
    predictions = pd.read_parquet(output_dir / "oos_predictions.parquet")
    metrics = pd.read_csv(output_dir / "fold_metrics.csv")
    metadata = json.loads((output_dir / "modeling_metadata.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((output_dir / "p4_diagnostics.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "p4_manifest.json").read_text(encoding="utf-8"))
    definitions = json.loads((output_dir / "fold_definitions.json").read_text(encoding="utf-8"))["folds"]

    if labels.duplicated(list(LABEL_KEY)).any():
        errors.append("Duplicate P4 label keys.")
    if matrix.duplicated(list(LABEL_KEY)).any():
        errors.append("Duplicate P4 model-matrix keys.")
    prediction_key = [*LABEL_KEY, "fold", "model"]
    if predictions.duplicated(prediction_key).any():
        errors.append("Duplicate OOS prediction keys.")
    probabilities = predictions["predicted_failure_probability"]
    if not probabilities.between(0.0, 1.0).all():
        errors.append("A failure probability is outside [0, 1].")
    if not np.allclose(
        predictions["calibrated_survival_probability"], 1.0 - probabilities
    ):
        errors.append("Survival probability is not one minus failure probability.")
    if metadata.get("synthetic_truth_read") is not False or diagnostics.get("synthetic_truth_read") is not False:
        errors.append("P4 does not certify exclusion of synthetic truth.")
    if metadata.get("return_test_run") is not False or diagnostics.get("return_test_run") is not False:
        errors.append("A return test was incorrectly marked as run in P4.")
    expected_models = set(MODEL_ORDER)
    if set(metrics["model"]) != expected_models or set(predictions["model"]) != expected_models:
        errors.append("P4 model ladder is incomplete or expanded.")

    original_axes = pd.read_parquet(p3_dir / "axis_scores.parquet")
    recomputed_labels = construct_fixed_horizon_labels(
        original_axes, raw_dir, config.viability.horizons_years
    )
    compare_columns = [
        *LABEL_KEY,
        "failure_within_horizon",
        "label_status",
        "censor_reason",
        "horizon_end",
        "label_observed_date",
    ]
    try:
        pd.testing.assert_frame_equal(
            labels[compare_columns].sort_values(list(LABEL_KEY)).reset_index(drop=True),
            recomputed_labels[compare_columns]
            .sort_values(list(LABEL_KEY))
            .reset_index(drop=True),
            check_dtype=False,
        )
    except AssertionError:
        errors.append("Saved P4 labels do not independently recompute.")

    expected_definitions = [
        item
        for horizon in config.viability.horizons_years
        for item in fold_definitions(config, horizon)
    ]
    if definitions != expected_definitions:
        errors.append("Saved fold definitions differ from frozen configuration.")
    for definition in definitions:
        horizon = definition["horizon_years"]
        subset = matrix.loc[matrix["horizon_years"] == horizon]
        try:
            _fold_frames(subset, definition)
        except ViabilityModelError as exc:
            errors.append(str(exc))

    recomputed_metric_rows = []
    for keys, group in predictions.groupby(["horizon_years", "fold", "model"], sort=True):
        calculated = prediction_metrics(
            group["failure_within_horizon"].astype(int).to_numpy(),
            group["predicted_failure_probability"].to_numpy(),
        )
        recomputed_metric_rows.append(
            {"horizon_years": keys[0], "fold": keys[1], "model": keys[2], **calculated}
        )
    recomputed_metrics = pd.DataFrame(recomputed_metric_rows)
    joined = metrics.merge(
        recomputed_metrics,
        on=["horizon_years", "fold", "model"],
        suffixes=("_saved", "_recomputed"),
        validate="one_to_one",
    )
    metric_names = [
        "roc_auc",
        "average_precision",
        "brier_score",
        "log_loss",
        "spearman",
        "calibration_intercept",
        "calibration_slope",
        "expected_calibration_error",
    ]
    for metric in metric_names:
        if not np.allclose(
            joined[f"{metric}_saved"],
            joined[f"{metric}_recomputed"],
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
        ):
            errors.append(f"Saved metric does not recompute: {metric}.")
    inventory = {item["name"]: item for item in manifest.get("files", [])}
    for name, record in inventory.items():
        path = output_dir / name
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            errors.append(f"Manifest hash mismatch: {name}.")
        if path.suffix == ".parquet" and int(pq.ParquetFile(path).metadata.num_rows) != int(
            record.get("rows", -1)
        ):
            errors.append(f"Manifest row-count mismatch: {name}.")
    if any((output_dir / name).exists() for name in ("return_tests.csv", "portfolio.csv")):
        errors.append("Return or portfolio output exists inside P4.")
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "warnings": [],
        "rows": {
            "labels": int(len(labels)),
            "observed_labels": int(labels["failure_within_horizon"].notna().sum()),
            "oos_predictions": int(len(predictions)),
            "metric_rows": int(len(metrics)),
        },
        "models_fitted": list(MODEL_ORDER),
        "return_test_run": False,
        "portfolio_run": False,
    }
