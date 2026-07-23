"""Cause-specific start/stop survival models for phase P6."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping
import warnings

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import norm, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.duration.hazard_regression import PHReg

from hypercube.config import HypercubeConfig
from hypercube.data import Scenario, atomic_write_json, sha256_file
from hypercube.dynamics import scenario_p5_dir, validate_p5_directory
from hypercube.viability import scenario_p4_dir


class SurvivalModelError(ValueError):
    """Raised when a P6 interval, cause, or time-split gate fails."""


P6_VERSION = "hypercube-survival-v1"
ACCOUNTING_KEY = ("gvkey", "datadate", "fyear")
CAUSE_COLUMNS = {
    "performance_failure": "performance_failure_event",
    "merger": "merger_event",
}


def scenario_p6_dir(config: HypercubeConfig, scenario: Scenario) -> Path:
    """Resolve one scenario's P6 output directory."""

    return scenario_p5_dir(config, scenario).parent / "p6"


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.to_parquet(name, index=False)
        os.replace(name, path)
    except Exception:
        Path(name).unlink(missing_ok=True)
        raise


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.to_csv(name, index=False, lineterminator="\n")
        os.replace(name, path)
    except Exception:
        Path(name).unlink(missing_ok=True)
        raise


def _atomic_joblib(payload: Any, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    joblib.dump(payload, temporary, compress=3)
    os.replace(temporary, path)


def _inventory(directory: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name == "p6_manifest.json":
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


def construct_survival_intervals(
    p2_dir: Path,
    p4_dir: Path,
    p5_dir: Path,
    config: HypercubeConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create nonoverlapping delayed-entry intervals with mutually exclusive exits."""

    surface = pd.read_parquet(p5_dir / "frontier_dynamics.parquet")
    surface = surface.loc[
        surface["horizon_years"].eq(config.survival.primary_horizon_years)
    ].copy()
    accounting = pd.read_parquet(
        p2_dir / "accounting_availability.parquet",
        columns=[*ACCOUNTING_KEY, "reporting_history"],
    )
    event_info = pd.read_parquet(
        p4_dir / "model_matrix.parquet",
        columns=[
            *ACCOUNTING_KEY,
            "horizon_years",
            "exit_date",
            "exit_category",
            "last_observed_date",
        ],
    )
    event_info = event_info.loc[
        event_info["horizon_years"].eq(config.survival.primary_horizon_years)
    ].drop(columns="horizon_years")
    event_info = event_info.drop_duplicates(list(ACCOUNTING_KEY))
    if event_info.duplicated("gvkey", keep=False).any():
        consistency = event_info.groupby("gvkey")[["exit_date", "exit_category", "last_observed_date"]].nunique(dropna=False)
        if consistency.gt(1).any().any():
            raise SurvivalModelError("A firm has inconsistent exit/follow-up metadata.")
    frame = surface.merge(
        accounting,
        on=list(ACCOUNTING_KEY),
        how="left",
        validate="one_to_one",
    ).merge(
        event_info,
        on=list(ACCOUNTING_KEY),
        how="left",
        validate="one_to_one",
    )
    frame = frame.loc[
        frame["reporting_history"].ge(config.survival.minimum_reporting_history)
    ].copy()
    if frame.empty or frame["last_observed_date"].isna().any():
        raise SurvivalModelError("P6 has no intervals or a row lacks raw follow-up.")
    frame = frame.sort_values(["gvkey", "feature_date", "datadate"])
    frame["next_feature_date"] = frame.groupby("gvkey", sort=False)[
        "feature_date"
    ].shift(-1)
    frame["interval_start_date"] = pd.to_datetime(frame["feature_date"])
    exit_after_start = pd.to_datetime(frame["exit_date"]).where(
        pd.to_datetime(frame["exit_date"]) > frame["interval_start_date"]
    )
    candidates = pd.concat(
        [
            pd.to_datetime(frame["next_feature_date"]).rename("next"),
            exit_after_start.rename("exit"),
            pd.to_datetime(frame["last_observed_date"]).rename("followup"),
        ],
        axis=1,
    )
    frame["interval_stop_date"] = candidates.min(axis=1)
    valid = frame["interval_stop_date"] > frame["interval_start_date"]
    frame = frame.loc[valid].copy()
    exit_is_stop = exit_after_start.loc[frame.index].eq(frame["interval_stop_date"])
    next_is_stop = pd.to_datetime(frame["next_feature_date"]).eq(
        frame["interval_stop_date"]
    ) & ~exit_is_stop
    frame["terminal_reason"] = "administrative_right_censor"
    frame.loc[next_is_stop, "terminal_reason"] = "covariate_update"
    frame.loc[exit_is_stop, "terminal_reason"] = frame.loc[
        exit_is_stop, "exit_category"
    ].fillna("other_unknown")
    frame["performance_failure_event"] = (
        exit_is_stop & frame["exit_category"].eq("performance_failure")
    ).astype("int8")
    frame["merger_event"] = (
        exit_is_stop & frame["exit_category"].eq("merger")
    ).astype("int8")
    if (frame[["performance_failure_event", "merger_event"]].sum(axis=1) > 1).any():
        raise SurvivalModelError("Cause-specific events are not mutually exclusive.")
    origin = pd.Timestamp(config.survival.time_origin_date)
    frame["entry_day"] = (
        frame["interval_start_date"] - origin
    ).dt.total_seconds() / 86400.0
    frame["stop_day"] = (
        frame["interval_stop_date"] - origin
    ).dt.total_seconds() / 86400.0
    if (frame["entry_day"] < 0.0).any() or (frame["stop_day"] <= frame["entry_day"]).any():
        raise SurvivalModelError("P6 delayed-entry times are invalid.")
    frame["interval_year"] = frame["interval_start_date"].dt.year.astype(int)
    frame["sic1"] = (pd.to_numeric(frame["sic2"], errors="coerce") // 10).astype(
        "Int64"
    )
    frame["interval_id"] = (
        frame["gvkey"].astype(str)
        + "|"
        + frame["interval_start_date"].dt.strftime("%Y-%m-%d")
    )
    if frame["interval_id"].duplicated().any():
        raise SurvivalModelError("Duplicate firm/start survival intervals.")

    first_feature = frame.groupby("gvkey")["interval_start_date"].min()
    exit_table = (
        frame.loc[frame["exit_date"].notna(), ["gvkey", "exit_date", "exit_category"]]
        .drop_duplicates("gvkey")
        .set_index("gvkey")
    )
    reconciliation_rows: list[dict[str, Any]] = []
    categories = (
        "performance_failure",
        "merger",
        "voluntary_administrative",
        "other_unknown",
    )
    for category in categories:
        eligible = exit_table.loc[
            exit_table["exit_category"].eq(category)
            & (pd.to_datetime(exit_table["exit_date"]) > first_feature.reindex(exit_table.index))
        ]
        assigned = frame.loc[frame["terminal_reason"].eq(category), "gvkey"].nunique()
        reconciliation_rows.append(
            {
                "exit_category": category,
                "eligible_dated_exits": int(len(eligible)),
                "assigned_terminal_intervals": int(assigned),
                "difference": int(assigned - len(eligible)),
            }
        )
    keep = [
        "interval_id",
        *ACCOUNTING_KEY,
        "permno",
        "feature_date",
        "availability_date",
        "formation_date",
        "reporting_history",
        "interval_start_date",
        "interval_stop_date",
        "entry_day",
        "stop_day",
        "interval_year",
        "sic2",
        "sic1",
        "exit_date",
        "exit_category",
        "terminal_reason",
        "performance_failure_event",
        "merger_event",
        *config.survival.features,
    ]
    intervals = frame[keep].sort_values(["gvkey", "interval_start_date"]).reset_index(
        drop=True
    )
    reconciliation = pd.DataFrame(reconciliation_rows)
    return intervals, reconciliation


def _preprocessor() -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=False)),
            ("scaler", StandardScaler()),
        ]
    )


def _cluster_covariance(result: Any, clusters: pd.Series) -> tuple[np.ndarray, str]:
    """Construct an issuer-clustered sandwich covariance from score residuals."""

    bread = np.asarray(result.cov_params(), dtype=float)
    scores = np.nan_to_num(np.asarray(result.score_residuals, dtype=float))
    codes, uniques = pd.factorize(clusters.astype(str), sort=False)
    summed = np.zeros((len(uniques), scores.shape[1]), dtype=float)
    np.add.at(summed, codes, scores)
    meat = summed.T @ summed
    n = len(scores)
    p = scores.shape[1]
    g = len(uniques)
    correction = (g / (g - 1.0)) * ((n - 1.0) / max(n - p, 1.0)) if g > 1 else 1.0
    covariance = bread @ meat @ bread * correction
    diagonal = np.diag(covariance)
    if not np.isfinite(covariance).all() or (diagonal < 0.0).any():
        return bread, "unclustered_fallback"
    return covariance, "issuer_clustered_sandwich"


def fit_cause_specific_model(
    frame: pd.DataFrame,
    cause: str,
    config: HypercubeConfig,
) -> dict[str, Any]:
    """Fit the frozen cause-specific PH model and clustered uncertainty."""

    if cause not in CAUSE_COLUMNS:
        raise SurvivalModelError(f"Unknown P6 cause: {cause}")
    event_column = CAUSE_COLUMNS[cause]
    if int(frame[event_column].sum()) < config.survival.minimum_train_events:
        raise SurvivalModelError(f"Too few {cause} events for PH estimation.")
    preprocess = _preprocessor()
    transformed = preprocess.fit_transform(frame[list(config.survival.features)])
    model = PHReg(
        frame["stop_day"].to_numpy(float),
        transformed,
        status=frame[event_column].to_numpy(int),
        entry=frame["entry_day"].to_numpy(float),
        ties=config.survival.ties,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = model.fit(disp=0)
    params = np.asarray(result.params, dtype=float)
    if not np.isfinite(params).all():
        raise SurvivalModelError(f"Nonfinite {cause} PH coefficients.")
    covariance, covariance_kind = _cluster_covariance(result, frame["gvkey"])
    return {
        "cause": cause,
        "event_column": event_column,
        "preprocess": preprocess,
        "params": params,
        "covariance": covariance,
        "covariance_kind": covariance_kind,
        "result": result,
        "warnings": [str(item.message) for item in caught],
        "features": list(config.survival.features),
    }


def coefficient_table(
    fitted: Mapping[str, Any],
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Return coefficients, issuer-clustered uncertainty, and hazard ratios."""

    params = np.asarray(fitted["params"], dtype=float)
    covariance = np.asarray(fitted["covariance"], dtype=float)
    standard_error = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    z_score = np.divide(
        params,
        standard_error,
        out=np.full_like(params, np.nan),
        where=standard_error > 0.0,
    )
    p_value = 2.0 * norm.sf(np.abs(z_score))
    return pd.DataFrame(
        {
            "cause": fitted["cause"],
            "feature": fitted["features"],
            "coefficient_per_train_sd": params,
            "clustered_standard_error": standard_error,
            "z_score": z_score,
            "p_value": p_value,
            "hazard_ratio": np.exp(params),
            "hazard_ratio_ci_lower": np.exp(params - 1.96 * standard_error),
            "hazard_ratio_ci_upper": np.exp(params + 1.96 * standard_error),
            "covariance_kind": fitted["covariance_kind"],
            "intervals": len(frame),
            "issuers": frame["gvkey"].nunique(),
            "events": int(frame[fitted["event_column"]].sum()),
        }
    )


def ph_diagnostic_table(fitted: Mapping[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    """Correlate event Schoenfeld residuals with log calendar event time."""

    residuals = np.asarray(fitted["result"].schoenfeld_residuals, dtype=float)
    event = frame[fitted["event_column"]].eq(1).to_numpy()
    log_time = np.log(frame["stop_day"].to_numpy(float))
    rows = []
    for index, feature in enumerate(fitted["features"]):
        valid = event & np.isfinite(residuals[:, index]) & np.isfinite(log_time)
        if int(valid.sum()) >= 5:
            statistic = spearmanr(residuals[valid, index], log_time[valid])
            correlation = float(statistic.statistic)
            p_value = float(statistic.pvalue)
        else:
            correlation = math.nan
            p_value = math.nan
        rows.append(
            {
                "cause": fitted["cause"],
                "feature": feature,
                "event_rows": int(valid.sum()),
                "schoenfeld_time_spearman": correlation,
                "p_value": p_value,
            }
        )
    return pd.DataFrame(rows)


def interval_concordance(frame: pd.DataFrame, risk: np.ndarray, event_column: str) -> tuple[float, int]:
    """Compute risk-set concordance for dated start/stop intervals."""

    starts = frame["entry_day"].to_numpy(float)
    stops = frame["stop_day"].to_numpy(float)
    events = frame[event_column].to_numpy(int)
    firms = frame["gvkey"].astype(str).to_numpy()
    concordant = 0.0
    comparable = 0
    for index in np.flatnonzero(events == 1):
        at_risk = (
            (starts <= stops[index])
            & (stops > stops[index])
            & (firms != firms[index])
        )
        others = risk[at_risk]
        if len(others) == 0:
            continue
        concordant += float((risk[index] > others).sum())
        concordant += 0.5 * float((risk[index] == others).sum())
        comparable += len(others)
    return (
        (concordant / comparable if comparable else math.nan),
        int(comparable),
    )


def _probability_metrics(y: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    if np.unique(y).size < 2:
        return {
            "roc_auc": math.nan,
            "average_precision": math.nan,
            "brier_score": float(brier_score_loss(y, probability)),
            "calibration_intercept": math.nan,
            "calibration_slope": math.nan,
        }
    bounded = np.clip(probability, 1e-6, 1.0 - 1e-6)
    logit = np.log(bounded / (1.0 - bounded)).reshape(-1, 1)
    calibration = LogisticRegression(penalty=None, max_iter=1000).fit(logit, y)
    return {
        "roc_auc": float(roc_auc_score(y, probability)),
        "average_precision": float(average_precision_score(y, probability)),
        "brier_score": float(brier_score_loss(y, probability)),
        "calibration_intercept": float(calibration.intercept_[0]),
        "calibration_slope": float(calibration.coef_[0, 0]),
    }


def run_time_split_models(
    intervals: pd.DataFrame,
    config: HypercubeConfig,
    models_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    """Fit frozen prior-period PH models and evaluate untouched calendar windows."""

    metric_rows: list[dict[str, Any]] = []
    predictions: list[pd.DataFrame] = []
    model_receipts: list[dict[str, Any]] = []
    features = list(config.survival.features)
    for fold_number, fold in enumerate(config.survival.outer_folds, start=1):
        cutoff = pd.Timestamp(fold.start_year, 1, 1)
        train = intervals.loc[intervals["interval_stop_date"] < cutoff].copy()
        test = intervals.loc[
            intervals["interval_year"].between(fold.start_year, fold.end_year)
        ].copy()
        if len(train) < config.survival.minimum_train_intervals or test.empty:
            raise SurvivalModelError(f"P6 fold has insufficient rows: {fold_number}")
        for cause in config.survival.causes:
            event_column = CAUSE_COLUMNS[cause]
            train_events = int(train[event_column].sum())
            test_events = int(test[event_column].sum())
            if train_events < config.survival.minimum_train_events:
                raise SurvivalModelError(f"P6 fold has too few train {cause} events.")
            if test_events < config.survival.minimum_test_events:
                raise SurvivalModelError(f"P6 fold has too few test {cause} events.")
            fitted = fit_cause_specific_model(train, cause, config)
            x_train = fitted["preprocess"].transform(train[features])
            x_test = fitted["preprocess"].transform(test[features])
            risk_train = x_train @ fitted["params"]
            risk_test = x_test @ fitted["params"]
            calibrator = LogisticRegression(penalty=None, max_iter=1000).fit(
                risk_train.reshape(-1, 1), train[event_column].to_numpy(int)
            )
            probability = calibrator.predict_proba(risk_test.reshape(-1, 1))[:, 1]
            c_index, comparable_pairs = interval_concordance(
                test, risk_test, event_column
            )
            calculated = _probability_metrics(
                test[event_column].to_numpy(int), probability
            )
            metric_rows.append(
                {
                    "cause": cause,
                    "fold": fold_number,
                    "test_start_year": fold.start_year,
                    "test_end_year": fold.end_year,
                    "train_intervals": len(train),
                    "train_issuers": train["gvkey"].nunique(),
                    "train_events": train_events,
                    "train_max_stop_date": train["interval_stop_date"].max(),
                    "test_intervals": len(test),
                    "test_issuers": test["gvkey"].nunique(),
                    "test_events": test_events,
                    "concordance": c_index,
                    "comparable_pairs": comparable_pairs,
                    **calculated,
                }
            )
            prediction = test[
                [
                    "interval_id",
                    "gvkey",
                    "interval_start_date",
                    "interval_stop_date",
                    "entry_day",
                    "stop_day",
                    "interval_year",
                    "sic1",
                    event_column,
                ]
            ].copy()
            prediction = prediction.rename(columns={event_column: "event"})
            prediction["cause"] = cause
            prediction["fold"] = fold_number
            prediction["risk_score"] = risk_test
            prediction["predicted_interval_event_probability"] = probability
            predictions.append(prediction)
            artifact_path = models_dir / f"{cause}_fold{fold_number}.joblib"
            payload = {
                "version": P6_VERSION,
                "cause": cause,
                "fold": fold_number,
                "test_window": [fold.start_year, fold.end_year],
                "train_max_stop_date": train["interval_stop_date"].max(),
                "features": features,
                "preprocess": fitted["preprocess"],
                "params": fitted["params"],
                "covariance": fitted["covariance"],
                "calibrator": calibrator,
            }
            _atomic_joblib(payload, artifact_path)
            model_receipts.append(
                {
                    "cause": cause,
                    "fold": fold_number,
                    "artifact": artifact_path.name,
                    "train_max_stop_date": train[
                        "interval_stop_date"
                    ].max().isoformat(),
                    "warnings": fitted["warnings"],
                }
            )
    return pd.DataFrame(metric_rows), pd.concat(predictions, ignore_index=True), model_receipts


def subgroup_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Report predefined decade and broad-industry stability diagnostics."""

    rows: list[dict[str, Any]] = []
    frame = predictions.copy()
    frame["decade"] = (frame["interval_year"] // 10 * 10).astype(int)
    for dimension in ("decade", "sic1"):
        for (cause, value), group in frame.groupby(["cause", dimension], dropna=False):
            y = group["event"].to_numpy(int)
            p = group["predicted_interval_event_probability"].to_numpy(float)
            metrics = _probability_metrics(y, p)
            rows.append(
                {
                    "dimension": dimension,
                    "value": str(value),
                    "cause": cause,
                    "intervals": len(group),
                    "issuers": group["gvkey"].nunique(),
                    "events": int(y.sum()),
                    "roc_auc": metrics["roc_auc"],
                    "brier_score": metrics["brier_score"],
                }
            )
    return pd.DataFrame(rows)


def build_survival_bundle(
    p2_dir: Path,
    p3_dir: Path,
    p4_dir: Path,
    p5_dir: Path,
    raw_dir: Path,
    output_dir: Path,
    config: HypercubeConfig,
    *,
    scenario: str,
) -> dict[str, Any]:
    """Build and atomically publish one complete P6 survival bundle."""

    if config.project.phase != "P6":
        raise SurvivalModelError("Survival construction requires a P6 config.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite completed P6 output: {output_dir}")
    p5_validation = validate_p5_directory(
        p5_dir,
        p2_dir,
        p3_dir,
        p4_dir,
        raw_dir,
        config,
        synthetic_truth_path=None,
    )
    if p5_validation["status"] != "PASS":
        raise SurvivalModelError(f"P5 input failed validation: {p5_validation['errors']}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        models_dir = staging / "models"
        models_dir.mkdir()
        intervals, reconciliation = construct_survival_intervals(
            p2_dir, p4_dir, p5_dir, config
        )
        coefficients = []
        ph_rows = []
        full_receipts = []
        for cause in config.survival.causes:
            fitted = fit_cause_specific_model(intervals, cause, config)
            coefficients.append(coefficient_table(fitted, intervals))
            ph_rows.append(ph_diagnostic_table(fitted, intervals))
            artifact_path = models_dir / f"{cause}_full.joblib"
            _atomic_joblib(
                {
                    "version": P6_VERSION,
                    "cause": cause,
                    "features": list(config.survival.features),
                    "preprocess": fitted["preprocess"],
                    "params": fitted["params"],
                    "covariance": fitted["covariance"],
                    "covariance_kind": fitted["covariance_kind"],
                },
                artifact_path,
            )
            full_receipts.append(
                {
                    "cause": cause,
                    "artifact": artifact_path.name,
                    "events": int(intervals[CAUSE_COLUMNS[cause]].sum()),
                    "warnings": fitted["warnings"],
                    "covariance_kind": fitted["covariance_kind"],
                }
            )
        coefficient_frame = pd.concat(coefficients, ignore_index=True)
        ph_frame = pd.concat(ph_rows, ignore_index=True)
        fold_metrics, fold_predictions, fold_receipts = run_time_split_models(
            intervals, config, models_dir
        )
        subgroup = subgroup_metrics(fold_predictions)
        missingness = pd.DataFrame(
            {
                "feature": list(config.survival.features),
                "missing_rows": [
                    int(intervals[item].isna().sum()) for item in config.survival.features
                ],
                "missing_rate": [
                    float(intervals[item].isna().mean()) for item in config.survival.features
                ],
            }
        )
        _atomic_parquet(intervals, staging / "survival_intervals.parquet")
        _atomic_parquet(fold_predictions, staging / "fold_predictions.parquet")
        _atomic_csv(coefficient_frame, staging / "cause_coefficients.csv")
        _atomic_csv(ph_frame, staging / "ph_diagnostics.csv")
        _atomic_csv(fold_metrics, staging / "fold_metrics.csv")
        _atomic_csv(subgroup, staging / "subgroup_metrics.csv")
        _atomic_csv(reconciliation, staging / "exit_reconciliation.csv")
        _atomic_csv(missingness, staging / "interval_missingness.csv")
        diagnostics = {
            "scenario": scenario,
            "intervals": len(intervals),
            "issuers": intervals["gvkey"].nunique(),
            "interval_start": intervals["interval_start_date"].min().isoformat(),
            "interval_stop": intervals["interval_stop_date"].max().isoformat(),
            "terminal_reason_counts": {
                str(key): int(value)
                for key, value in intervals["terminal_reason"].value_counts().items()
            },
            "events": {
                cause: int(intervals[column].sum())
                for cause, column in CAUSE_COLUMNS.items()
            },
            "reconciliation_difference": int(reconciliation["difference"].abs().sum()),
            "full_model_receipts": full_receipts,
            "fold_model_receipts": fold_receipts,
            "synthetic_truth_read": False,
            "return_test_run": False,
            "portfolio_run": False,
            "causal_claim": False,
        }
        atomic_write_json(staging / "p6_diagnostics.json", diagnostics)
        atomic_write_json(staging / "resolved_config.json", config.model_dump(mode="json"))
        metadata = {
            "version": P6_VERSION,
            "unit": "nonoverlapping firm start/stop interval",
            "delayed_entry": "absolute days since 1900-01-01",
            "causes": list(config.survival.causes),
            "competing_exit_policy": "censor at actual competing-exit date",
            "features": list(config.survival.features),
            "estimator": "statsmodels PHReg, Breslow ties",
            "uncertainty": "issuer-clustered score-residual sandwich",
            "ph_diagnostic": "Schoenfeld residual Spearman correlation with log event time",
            "time_split_calibration": config.survival.calibration,
            "causal_claim": False,
            "return_fields_read": False,
        }
        atomic_write_json(staging / "modeling_metadata.json", metadata)
        input_paths = (
            p2_dir / "accounting_availability.parquet",
            p4_dir / "model_matrix.parquet",
            p5_dir / "frontier_dynamics.parquet",
        )
        manifest = {
            "schema_version": 1,
            "phase": "P6",
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
                for path in input_paths
            ],
            "files": _inventory(staging),
            "synthetic_truth_read": False,
            "return_test_run": False,
        }
        atomic_write_json(staging / "p6_manifest.json", manifest)
        if output_dir.exists():
            output_dir.rmdir()
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return diagnostics


def validate_p6_directory(
    output_dir: Path,
    p2_dir: Path,
    p4_dir: Path,
    p5_dir: Path,
    config: HypercubeConfig,
) -> dict[str, Any]:
    """Independently validate interval reconciliation, metrics, timing, and hashes."""

    required = (
        "survival_intervals.parquet",
        "fold_predictions.parquet",
        "cause_coefficients.csv",
        "ph_diagnostics.csv",
        "fold_metrics.csv",
        "subgroup_metrics.csv",
        "exit_reconciliation.csv",
        "interval_missingness.csv",
        "p6_diagnostics.json",
        "resolved_config.json",
        "modeling_metadata.json",
        "p6_manifest.json",
    )
    missing = [name for name in required if not (output_dir / name).is_file()]
    if missing:
        return {"status": "FAIL", "errors": [f"Missing P6 outputs: {missing}"], "warnings": []}
    errors: list[str] = []
    intervals = pd.read_parquet(output_dir / "survival_intervals.parquet")
    predictions = pd.read_parquet(output_dir / "fold_predictions.parquet")
    coefficients = pd.read_csv(output_dir / "cause_coefficients.csv")
    ph = pd.read_csv(output_dir / "ph_diagnostics.csv")
    metrics = pd.read_csv(output_dir / "fold_metrics.csv")
    reconciliation = pd.read_csv(output_dir / "exit_reconciliation.csv")
    diagnostics = json.loads((output_dir / "p6_diagnostics.json").read_text(encoding="utf-8"))
    metadata = json.loads((output_dir / "modeling_metadata.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "p6_manifest.json").read_text(encoding="utf-8"))
    recomputed_intervals, recomputed_reconciliation = construct_survival_intervals(
        p2_dir, p4_dir, p5_dir, config
    )
    compare = [
        "interval_id",
        "interval_start_date",
        "interval_stop_date",
        "entry_day",
        "stop_day",
        "terminal_reason",
        "performance_failure_event",
        "merger_event",
    ]
    try:
        pd.testing.assert_frame_equal(
            intervals[compare].sort_values("interval_id").reset_index(drop=True),
            recomputed_intervals[compare]
            .sort_values("interval_id")
            .reset_index(drop=True),
            check_dtype=False,
        )
    except AssertionError:
        errors.append("Saved P6 intervals do not independently recompute.")
    try:
        pd.testing.assert_frame_equal(
            reconciliation.sort_values("exit_category").reset_index(drop=True),
            recomputed_reconciliation.sort_values("exit_category").reset_index(drop=True),
            check_dtype=False,
        )
    except AssertionError:
        errors.append("Saved exit reconciliation does not recompute.")
    if intervals["interval_id"].duplicated().any():
        errors.append("Duplicate P6 interval IDs.")
    if (intervals["stop_day"] <= intervals["entry_day"]).any():
        errors.append("A P6 interval has nonpositive exposure time.")
    ordered = intervals.sort_values(["gvkey", "interval_start_date"])
    previous_stop = ordered.groupby("gvkey")["interval_stop_date"].shift(1)
    if (previous_stop > ordered["interval_start_date"]).any():
        errors.append("P6 intervals overlap within firm.")
    if (intervals[[*CAUSE_COLUMNS.values()]].sum(axis=1) > 1).any():
        errors.append("P6 cause events are not mutually exclusive.")
    if reconciliation["difference"].abs().sum() != 0:
        errors.append("P6 exit categories do not reconcile.")
    if set(coefficients["cause"]) != set(config.survival.causes):
        errors.append("P6 coefficient causes are incomplete.")
    for cause in config.survival.causes:
        if set(coefficients.loc[coefficients["cause"].eq(cause), "feature"]) != set(
            config.survival.features
        ):
            errors.append(f"P6 coefficient features are incomplete for {cause}.")
    if set(ph["cause"]) != set(config.survival.causes):
        errors.append("P6 PH diagnostics are incomplete.")
    expected_metric_rows = len(config.survival.causes) * len(config.survival.outer_folds)
    if len(metrics) != expected_metric_rows:
        errors.append("P6 time-split metric ladder is incomplete.")
    metric_rows = []
    for (cause, fold), group in predictions.groupby(["cause", "fold"], sort=True):
        probability = group["predicted_interval_event_probability"].to_numpy(float)
        y = group["event"].to_numpy(int)
        calculated = _probability_metrics(y, probability)
        c_index, pairs = interval_concordance(
            group,
            group["risk_score"].to_numpy(float),
            CAUSE_COLUMNS.get(cause, "event") if CAUSE_COLUMNS.get(cause, "event") in group else "event",
        )
        metric_rows.append(
            {
                "cause": cause,
                "fold": fold,
                "concordance_recomputed": c_index,
                "comparable_pairs_recomputed": pairs,
                **{f"{key}_recomputed": value for key, value in calculated.items()},
            }
        )
    recomputed_metrics = pd.DataFrame(metric_rows)
    joined = metrics.merge(recomputed_metrics, on=["cause", "fold"], validate="one_to_one")
    for metric in (
        "concordance",
        "roc_auc",
        "average_precision",
        "brier_score",
        "calibration_intercept",
        "calibration_slope",
    ):
        if not np.allclose(
            joined[metric], joined[f"{metric}_recomputed"], equal_nan=True
        ):
            errors.append(f"P6 metric does not recompute: {metric}.")
    if not (pd.to_datetime(metrics["train_max_stop_date"]) < pd.to_datetime(metrics["test_start_year"].astype(str) + "-01-01")).all():
        errors.append("P6 training outcomes reach the outer test period.")
    if diagnostics.get("synthetic_truth_read") is not False or manifest.get(
        "synthetic_truth_read"
    ) is not False:
        errors.append("P6 does not certify synthetic-truth exclusion.")
    if metadata.get("causal_claim") is not False or diagnostics.get("causal_claim") is not False:
        errors.append("P6 incorrectly makes a causal claim.")
    if diagnostics.get("return_test_run") is not False or diagnostics.get("portfolio_run") is not False:
        errors.append("P6 incorrectly marks a return or portfolio test as run.")
    inventory = {item["name"]: item for item in manifest.get("files", [])}
    for name, record in inventory.items():
        path = output_dir / name
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            errors.append(f"Manifest hash mismatch: {name}.")
        if path.suffix == ".parquet" and int(pq.ParquetFile(path).metadata.num_rows) != int(record.get("rows", -1)):
            errors.append(f"Manifest row-count mismatch: {name}.")
    forbidden = ("return_tests.csv", "portfolio.csv", "clusters.parquet")
    if any((output_dir / name).exists() for name in forbidden):
        errors.append("A downstream P7-P9 output exists inside P6.")
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "warnings": [],
        "rows": {
            "intervals": len(intervals),
            "fold_predictions": len(predictions),
            "coefficient_rows": len(coefficients),
            "metric_rows": len(metrics),
        },
        "events": {
            cause: int(intervals[column].sum())
            for cause, column in CAUSE_COLUMNS.items()
        },
        "models_fitted": [f"cause_specific_ph:{cause}" for cause in config.survival.causes],
        "return_test_run": False,
        "portfolio_run": False,
        "causal_claim": False,
    }
