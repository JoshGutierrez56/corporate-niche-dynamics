"""Point-in-time viability frontier and niche dynamics for phase P5."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from hypercube.axes import scenario_p3_dir
from hypercube.config import HypercubeConfig
from hypercube.data import Scenario, atomic_write_json, sha256_file
from hypercube.viability import (
    ANCHORED_AXES,
    DISTRESS_FEATURES,
    PROFITABILITY_FEATURES,
    RELATIVE_AXES,
    _apply_platt,
    fold_definitions,
    scenario_p4_dir,
    validate_p4_directory,
)


class DynamicsError(ValueError):
    """Raised when a P5 point-in-time or recovery contract is violated."""


P5_VERSION = "hypercube-dynamics-v1"
EVENT_KEY = ("gvkey", "datadate", "fyear", "horizon_years")
ACCOUNTING_KEY = ("gvkey", "datadate", "fyear")
AXIS_COLUMNS = tuple(RELATIVE_AXES)
NUMERIC_MIGRATION_CONTROLS = (
    "previous_viability_log_odds",
    "log_market_cap",
    "book_to_market",
    "momentum_12_2",
    "prior_1m_return",
    "operating_margin",
    "asset_growth",
    "book_leverage",
    "working_capital_assets",
    "operating_income_assets",
    "equity_to_liabilities",
    "sales_to_assets",
    "log_dollar_volume_12m",
    "realized_volatility_12m",
    "equity_beta_24m",
)
CATEGORICAL_MIGRATION_CONTROLS = ("sic2",)


def scenario_p5_dir(config: HypercubeConfig, scenario: Scenario) -> Path:
    """Resolve one scenario's P5 output directory."""

    return scenario_p4_dir(config, scenario).parent / "p5"


def _logit(values: pd.Series | np.ndarray, clip: float) -> np.ndarray:
    probabilities = np.clip(np.asarray(values, dtype=float), clip, 1.0 - clip)
    return np.log(probabilities / (1.0 - probabilities))


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
        if not path.is_file() or path.name == "p5_manifest.json":
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


def select_oos_surface(p4_dir: Path, config: HypercubeConfig) -> pd.DataFrame:
    """Apply frozen P4 artifacts to every outer-test event, including censored rows."""

    matrix = pd.read_parquet(p4_dir / "model_matrix.parquet")
    matrix_columns = [
        "permno",
        *EVENT_KEY,
        "feature_date",
        "horizon_end",
        "failure_within_horizon",
        "availability_date",
        "formation_date",
        "label_observed_date",
        "sic2",
        "market_cap_millions",
        *PROFITABILITY_FEATURES,
        *DISTRESS_FEATURES,
        *RELATIVE_AXES,
        *ANCHORED_AXES,
    ]
    pieces: list[pd.DataFrame] = []
    for horizon in config.viability.horizons_years:
        for definition in fold_definitions(config, horizon):
            test = matrix.loc[
                matrix["horizon_years"].eq(horizon)
                & matrix["event_year"].between(
                    definition["test_start_year"], definition["test_end_year"]
                ),
                matrix_columns,
            ].copy()
            artifact_path = (
                p4_dir
                / "models"
                / f"h{horizon}_fold{definition['fold']}_{config.dynamics.primary_model}.joblib"
            )
            if test.empty or not artifact_path.is_file():
                raise DynamicsError(f"Missing P4 test rows or model artifact: {artifact_path}")
            artifact = joblib.load(artifact_path)
            if (
                artifact.get("horizon_years") != horizon
                or artifact.get("model") != config.dynamics.primary_model
                or artifact.get("fold", {}).get("fold") != definition["fold"]
            ):
                raise DynamicsError(f"P4 artifact provenance mismatch: {artifact_path}")
            fitted = artifact["fitted"]
            features = list(fitted["features"])
            raw_failure = fitted["estimator"].predict_proba(test[features])[:, 1]
            calibrated_failure = _apply_platt(fitted["calibrator"], raw_failure)
            test["fold"] = int(definition["fold"])
            test["model"] = config.dynamics.primary_model
            test["predicted_failure_probability"] = calibrated_failure
            test["calibrated_survival_probability"] = 1.0 - calibrated_failure
            pieces.append(test)
    selected = pd.concat(pieces, ignore_index=True)
    if selected.duplicated(list(EVENT_KEY)).any():
        raise DynamicsError("The primary OOS surface contains duplicate event keys.")
    if selected["formation_date"].isna().any():
        raise DynamicsError("A primary OOS prediction lacks formation metadata.")
    if (
        pd.to_datetime(selected["feature_date"])
        < pd.to_datetime(selected["formation_date"])
    ).any():
        raise DynamicsError("A P5 feature precedes eligible formation.")

    clip = config.dynamics.probability_clip
    threshold = config.dynamics.survival_probability_threshold
    selected["viability_level"] = selected["calibrated_survival_probability"]
    selected["viability_log_odds"] = _logit(selected["viability_level"], clip)
    selected["annualized_constant_hazard_proxy"] = (
        -np.log(np.clip(selected["viability_level"].to_numpy(float), clip, 1.0))
        / selected["horizon_years"].to_numpy(float)
    )
    selected["viability_margin_log_odds"] = selected["viability_log_odds"] - float(
        _logit(np.array([threshold]), clip)[0]
    )
    selected["above_frontier"] = selected["viability_level"] >= threshold
    selected["feature_year"] = pd.to_datetime(selected["feature_date"]).dt.year.astype(int)
    cohort = selected.groupby(["horizon_years", "feature_date"], sort=False)[
        "viability_level"
    ]
    selected["viability_cohort_size"] = cohort.transform("size").astype("int32")
    selected["viability_percentile"] = cohort.rank(method="average", pct=True)
    selected.loc[
        selected["viability_cohort_size"] < config.dynamics.cross_section_minimum,
        "viability_percentile",
    ] = np.nan
    return selected.sort_values(list(EVENT_KEY)).reset_index(drop=True)


def _rolling_market_controls(panel_path: Path, factor_path: Path) -> pd.DataFrame:
    """Create lagged market controls using only months preceding each event month."""

    panel = pd.read_parquet(
        panel_path,
        columns=["gvkey", "date", "ret_total", "prc", "vol"],
    ).sort_values(["gvkey", "date"])
    factors = pd.read_parquet(factor_path, columns=["date", "mkt_excess", "rf"])
    panel = panel.merge(factors, on="date", how="left", validate="many_to_one")
    panel["dollar_volume"] = panel["prc"].abs() * panel["vol"].clip(lower=0.0)

    pieces: list[pd.DataFrame] = []
    for _, group in panel.groupby("gvkey", sort=False):
        group = group.copy()
        returns = pd.to_numeric(group["ret_total"], errors="coerce")
        market = pd.to_numeric(group["mkt_excess"], errors="coerce")
        rf = pd.to_numeric(group["rf"], errors="coerce")
        excess = returns - rf
        prior_return = returns.shift(1)
        momentum_base = (1.0 + returns.shift(2)).clip(lower=0.01)
        group["momentum_12_2"] = (
            momentum_base.rolling(11, min_periods=8).apply(np.prod, raw=True) - 1.0
        )
        group["prior_1m_return"] = prior_return
        group["realized_volatility_12m"] = prior_return.rolling(
            12, min_periods=8
        ).std(ddof=1)
        group["log_dollar_volume_12m"] = np.log1p(
            group["dollar_volume"].shift(1).rolling(12, min_periods=8).mean()
        )
        x = market.shift(1)
        y = excess.shift(1)
        xy = (x * y).rolling(24, min_periods=18).mean()
        x_mean = x.rolling(24, min_periods=18).mean()
        y_mean = y.rolling(24, min_periods=18).mean()
        x2 = (x * x).rolling(24, min_periods=18).mean()
        variance = x2 - x_mean * x_mean
        group["equity_beta_24m"] = (xy - x_mean * y_mean) / variance.replace(
            0.0, np.nan
        )
        pieces.append(
            group[
                [
                    "gvkey",
                    "date",
                    "momentum_12_2",
                    "prior_1m_return",
                    "realized_volatility_12m",
                    "log_dollar_volume_12m",
                    "equity_beta_24m",
                ]
            ]
        )
    return pd.concat(pieces, ignore_index=True)


def add_known_characteristic_controls(
    surface: pd.DataFrame,
    p2_dir: Path,
    raw_dir: Path,
) -> pd.DataFrame:
    """Attach the frozen size/value/momentum/fundamental/liquidity controls."""

    accounting = pd.read_parquet(
        p2_dir / "accounting_availability.parquet",
        columns=[*ACCOUNTING_KEY, "at", "ceq"],
    ).sort_values(["gvkey", "datadate"])
    accounting["prior_at"] = accounting.groupby("gvkey", sort=False)["at"].shift(1)
    accounting["prior_fyear"] = accounting.groupby("gvkey", sort=False)["fyear"].shift(1)
    accounting["asset_growth"] = accounting["at"] / accounting["prior_at"] - 1.0
    gap = accounting["fyear"] - accounting["prior_fyear"]
    accounting.loc[gap.ne(1), "asset_growth"] = np.nan
    accounting = accounting[[*ACCOUNTING_KEY, "ceq", "asset_growth"]]
    controls = surface.merge(
        accounting, on=list(ACCOUNTING_KEY), how="left", validate="many_to_one"
    )
    controls["book_to_market"] = controls["ceq"] / controls[
        "market_cap_millions"
    ].replace(0.0, np.nan)
    market = _rolling_market_controls(
        p2_dir / "firm_month_panel.parquet",
        raw_dir / "factor_returns.parquet",
    )
    controls = controls.merge(
        market,
        left_on=["gvkey", "feature_date"],
        right_on=["gvkey", "date"],
        how="left",
        validate="many_to_one",
    ).drop(columns="date")
    numeric = controls[list(NUMERIC_MIGRATION_CONTROLS[1:])].to_numpy(float)
    if np.isinf(numeric).any():
        raise DynamicsError("Known-characteristic controls contain infinities.")
    return controls


def _neighbor_density(
    reference: pd.DataFrame,
    queries: pd.DataFrame,
    *,
    neighbors: int,
    exclude_self: bool,
) -> np.ndarray:
    """Compute bounded inverse-distance density with reference-only scaling."""

    if reference.empty or queries.empty:
        return np.full(len(queries), np.nan)
    reference_values = reference[list(AXIS_COLUMNS)].to_numpy(float)
    query_values = queries[list(AXIS_COLUMNS)].to_numpy(float)
    valid_reference = np.isfinite(reference_values).all(axis=1)
    valid_query = np.isfinite(query_values).all(axis=1)
    output = np.full(len(queries), np.nan)
    if int(valid_query.sum()) == 0 or int(valid_reference.sum()) < neighbors + int(
        exclude_self
    ):
        return output
    scaler = StandardScaler().fit(reference_values[valid_reference])
    fitted_reference = scaler.transform(reference_values[valid_reference])
    fitted_queries = scaler.transform(query_values[valid_query])
    reference_valid = reference.loc[valid_reference].reset_index(drop=True)
    query_valid = queries.loc[valid_query].reset_index(drop=True)
    count = min(neighbors + int(exclude_self), len(reference_valid))
    fitted = NearestNeighbors(n_neighbors=count, metric="euclidean").fit(
        fitted_reference
    )
    distances, indices = fitted.kneighbors(fitted_queries)
    reference_ids = (
        reference_valid["gvkey"].astype(str)
        + "|"
        + reference_valid["datadate"].astype(str)
    ).to_numpy()
    query_ids = (
        query_valid["gvkey"].astype(str) + "|" + query_valid["datadate"].astype(str)
    ).to_numpy()
    densities: list[float] = []
    for row_index, row_distances in enumerate(distances):
        if exclude_self:
            keep = reference_ids[indices[row_index]] != query_ids[row_index]
            selected = row_distances[keep][:neighbors]
        else:
            selected = row_distances[:neighbors]
        densities.append(
            np.nan if len(selected) < neighbors else 1.0 / (1.0 + float(selected.mean()))
        )
    output[np.flatnonzero(valid_query)] = np.asarray(densities)
    return output


def contemporaneous_crowding(
    surface: pd.DataFrame,
    p3_dir: Path,
    config: HypercubeConfig,
) -> pd.DataFrame:
    """Measure same-date state-space crowding with SIC2 then market fallback."""

    axes = pd.read_parquet(
        p3_dir / "axis_scores.parquet",
        columns=[*ACCOUNTING_KEY, "feature_date", "sic2", *AXIS_COLUMNS],
    )
    query = surface[[*ACCOUNTING_KEY, "feature_date", "sic2"]].drop_duplicates(
        list(ACCOUNTING_KEY)
    )
    query = query.merge(
        axes,
        on=[*ACCOUNTING_KEY, "feature_date", "sic2"],
        how="left",
        validate="one_to_one",
    )
    query["crowding_density"] = np.nan
    query["crowding_reference_level"] = "unavailable"
    query["crowding_reference_count"] = 0
    minimum = config.dynamics.crowding_neighbors + 1
    for date, date_queries in query.groupby("feature_date", sort=True):
        dated_reference = axes.loc[axes["feature_date"].eq(date)]
        for sic2, industry_queries in date_queries.groupby("sic2", sort=True):
            industry_reference = dated_reference.loc[dated_reference["sic2"].eq(sic2)]
            if len(industry_reference) >= minimum:
                reference = industry_reference
                level = "sic2"
            elif len(dated_reference) >= minimum:
                reference = dated_reference
                level = "market"
            else:
                continue
            values = _neighbor_density(
                reference,
                industry_queries,
                neighbors=config.dynamics.crowding_neighbors,
                exclude_self=True,
            )
            query.loc[industry_queries.index, "crowding_density"] = values
            query.loc[industry_queries.index, "crowding_reference_level"] = level
            query.loc[industry_queries.index, "crowding_reference_count"] = len(reference)
    return query[
        [
            *ACCOUNTING_KEY,
            "crowding_density",
            "crowding_reference_level",
            "crowding_reference_count",
        ]
    ]


def historical_success_density(
    surface: pd.DataFrame,
    p4_dir: Path,
    config: HypercubeConfig,
) -> pd.DataFrame:
    """Measure density among prior successful events known before each year."""

    matrix = pd.read_parquet(p4_dir / "model_matrix.parquet")
    output = surface[[*EVENT_KEY, "feature_date", *AXIS_COLUMNS]].copy()
    output["historical_success_density"] = np.nan
    output["successful_reference_count"] = 0
    output["successful_reference_cutoff"] = pd.NaT
    for (horizon, year), queries in output.groupby(
        ["horizon_years", output["feature_date"].dt.year], sort=True
    ):
        cutoff = pd.Timestamp(int(year), 1, 1)
        reference = matrix.loc[
            matrix["horizon_years"].eq(horizon)
            & matrix["failure_within_horizon"].eq(0)
            & (pd.to_datetime(matrix["feature_date"]) < cutoff)
            & (pd.to_datetime(matrix["label_observed_date"]) < cutoff),
            [*ACCOUNTING_KEY, *AXIS_COLUMNS],
        ].copy()
        if len(reference) < config.dynamics.successful_density_minimum:
            continue
        values = _neighbor_density(
            reference,
            queries,
            neighbors=config.dynamics.successful_density_neighbors,
            exclude_self=False,
        )
        output.loc[queries.index, "historical_success_density"] = values
        output.loc[queries.index, "successful_reference_count"] = len(reference)
        output.loc[queries.index, "successful_reference_cutoff"] = cutoff - pd.Timedelta(
            days=1
        )
    return output[
        [
            *EVENT_KEY,
            "historical_success_density",
            "successful_reference_count",
            "successful_reference_cutoff",
        ]
    ]


def add_time_dynamics(surface: pd.DataFrame) -> pd.DataFrame:
    """Construct annualized movement without crossing P4 model vintages."""

    frame = surface.sort_values(["gvkey", "horizon_years", "feature_date"]).copy()
    groups = frame.groupby(["gvkey", "horizon_years"], sort=False)
    frame["previous_feature_date"] = groups["feature_date"].shift(1)
    frame["previous_fold"] = groups["fold"].shift(1)
    frame["previous_viability_level"] = groups["viability_level"].shift(1)
    frame["previous_viability_log_odds"] = groups["viability_log_odds"].shift(1)
    elapsed = (
        pd.to_datetime(frame["feature_date"])
        - pd.to_datetime(frame["previous_feature_date"])
    ).dt.days / 365.25
    frame["elapsed_years"] = elapsed
    comparable = frame["previous_fold"].eq(frame["fold"]) & elapsed.gt(0.0)
    frame["model_vintage_changed"] = frame["previous_fold"].notna() & ~frame[
        "previous_fold"
    ].eq(frame["fold"])
    frame["velocity_probability"] = (
        frame["viability_level"] - frame["previous_viability_level"]
    ) / elapsed
    frame["velocity_log_odds"] = (
        frame["viability_log_odds"] - frame["previous_viability_log_odds"]
    ) / elapsed
    frame.loc[~comparable, ["velocity_probability", "velocity_log_odds"]] = np.nan
    previous_velocity = groups["velocity_log_odds"].shift(1)
    frame["acceleration_log_odds"] = (
        frame["velocity_log_odds"] - previous_velocity
    ) / elapsed
    frame.loc[~comparable | previous_velocity.isna(), "acceleration_log_odds"] = np.nan
    previous_frontier = groups["above_frontier"].shift(1)
    frame["frontier_crossing"] = pd.Series(pd.NA, index=frame.index, dtype="Int8")
    frame.loc[comparable, "frontier_crossing"] = 0
    frame.loc[comparable & ~previous_frontier.astype("boolean") & frame["above_frontier"], "frontier_crossing"] = 1
    frame.loc[comparable & previous_frontier.astype("boolean") & ~frame["above_frontier"], "frontier_crossing"] = -1
    previous_crowding = groups["crowding_density"].shift(1)
    frame["competitive_encroachment"] = (
        frame["crowding_density"] - previous_crowding
    ) / elapsed
    frame.loc[elapsed.le(0.0), "competitive_encroachment"] = np.nan
    return frame.sort_values(list(EVENT_KEY)).reset_index(drop=True)


def _migration_pipeline(config: HypercubeConfig) -> Pipeline:
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                list(NUMERIC_MIGRATION_CONTROLS),
            ),
            (
                "industry",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL_MIGRATION_CONTROLS),
            ),
        ],
        remainder="drop",
    )
    return Pipeline(
        [
            ("preprocess", preprocessing),
            ("ridge", Ridge(alpha=config.dynamics.migration_ridge_alpha)),
        ]
    )


def residualize_migration(
    dynamics: pd.DataFrame,
    config: HypercubeConfig,
    models_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit expanding prior-year ridge expectations for viability velocity."""

    frame = dynamics.copy()
    frame["expected_velocity_log_odds"] = np.nan
    frame["migration_surprise"] = np.nan
    frame["migration_train_rows"] = pd.Series(pd.NA, index=frame.index, dtype="Int64")
    frame["migration_train_max_feature_date"] = pd.NaT
    diagnostics: list[dict[str, Any]] = []
    models_dir.mkdir(parents=True, exist_ok=True)
    for horizon in sorted(frame["horizon_years"].unique()):
        horizon_mask = frame["horizon_years"].eq(horizon)
        for year in sorted(frame.loc[horizon_mask, "feature_year"].unique()):
            prediction_mask = horizon_mask & frame["feature_year"].eq(year) & frame[
                "velocity_log_odds"
            ].notna()
            training_mask = horizon_mask & frame["feature_year"].lt(year) & frame[
                "velocity_log_odds"
            ].notna()
            train = frame.loc[training_mask]
            predict = frame.loc[prediction_mask]
            row: dict[str, Any] = {
                "horizon_years": int(horizon),
                "prediction_year": int(year),
                "train_rows": int(len(train)),
                "prediction_rows": int(len(predict)),
                "alpha": config.dynamics.migration_ridge_alpha,
                "status": "insufficient_history",
                "train_max_feature_date": None,
            }
            if predict.empty or len(train) < config.dynamics.migration_minimum_train_observations:
                diagnostics.append(row)
                continue
            training_max = pd.to_datetime(train["feature_date"]).max()
            if training_max >= pd.Timestamp(int(year), 1, 1):
                raise DynamicsError("Migration expectation includes the prediction year.")
            model = _migration_pipeline(config)
            features = [*NUMERIC_MIGRATION_CONTROLS, *CATEGORICAL_MIGRATION_CONTROLS]
            model.fit(train[features], train["velocity_log_odds"].to_numpy(float))
            expected = model.predict(predict[features])
            frame.loc[predict.index, "expected_velocity_log_odds"] = expected
            frame.loc[predict.index, "migration_surprise"] = (
                predict["velocity_log_odds"].to_numpy(float) - expected
            )
            frame.loc[predict.index, "migration_train_rows"] = len(train)
            frame.loc[predict.index, "migration_train_max_feature_date"] = training_max
            artifact = models_dir / f"h{int(horizon)}_predict_{int(year)}.joblib"
            _atomic_joblib(
                {
                    "version": P5_VERSION,
                    "horizon_years": int(horizon),
                    "prediction_year": int(year),
                    "train_max_feature_date": training_max,
                    "train_rows": len(train),
                    "features": features,
                    "model": model,
                },
                artifact,
            )
            row.update(
                {
                    "status": "fitted",
                    "train_max_feature_date": training_max,
                    "coefficient_l2_norm": float(
                        np.linalg.norm(model.named_steps["ridge"].coef_)
                    ),
                    "intercept": float(model.named_steps["ridge"].intercept_),
                    "artifact": artifact.name,
                }
            )
            diagnostics.append(row)
    return frame, pd.DataFrame(diagnostics)


def build_dynamics_frame(
    p2_dir: Path,
    p3_dir: Path,
    p4_dir: Path,
    raw_dir: Path,
    config: HypercubeConfig,
    models_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the full P5 feature frame without reading synthetic truth."""

    surface = select_oos_surface(p4_dir, config)
    surface = add_known_characteristic_controls(surface, p2_dir, raw_dir)
    crowding = contemporaneous_crowding(surface, p3_dir, config)
    success = historical_success_density(surface, p4_dir, config)
    surface = surface.merge(
        crowding, on=list(ACCOUNTING_KEY), how="left", validate="many_to_one"
    ).merge(success, on=list(EVENT_KEY), how="left", validate="one_to_one")
    dynamics = add_time_dynamics(surface)
    dynamics, model_diagnostics = residualize_migration(dynamics, config, models_dir)
    numeric_columns = dynamics.select_dtypes(include=[np.number]).columns
    if np.isinf(dynamics[numeric_columns].to_numpy(float)).any():
        raise DynamicsError("P5 dynamics contain infinite numeric values.")
    if dynamics.duplicated(list(EVENT_KEY)).any():
        raise DynamicsError("P5 dynamics contain duplicate event keys.")
    return dynamics, model_diagnostics


def _truth_recovery(
    frame: pd.DataFrame,
    truth_path: Path,
    config: HypercubeConfig,
) -> dict[str, Any]:
    """Evaluate predeclared synthetic signs without accessing return injections."""

    truth = pd.read_parquet(
        truth_path,
        columns=[*ACCOUNTING_KEY, "true_viability", "migration_surprise"],
    )
    primary = frame.loc[
        frame["horizon_years"].eq(config.dynamics.recovery_primary_horizon_years)
    ].merge(truth, on=list(ACCOUNTING_KEY), how="left", validate="one_to_one")
    if primary[["true_viability", "migration_surprise_y"]].isna().any().any():
        raise DynamicsError("Synthetic truth does not fully match P5 event keys.")

    def correlation(left: str, right: str) -> tuple[float, int]:
        subset = primary[[left, right]].dropna()
        value = float(spearmanr(subset[left], subset[right]).statistic)
        return value, int(len(subset))

    level, level_rows = correlation("viability_log_odds", "true_viability")
    velocity, velocity_rows = correlation("velocity_log_odds", "migration_surprise_y")
    surprise, surprise_rows = correlation("migration_surprise_x", "migration_surprise_y")
    gates = {
        "level": level >= config.dynamics.recovery_minimum_level_spearman,
        "velocity": velocity >= config.dynamics.recovery_minimum_velocity_spearman,
        "migration_surprise": surprise
        >= config.dynamics.recovery_minimum_surprise_spearman,
    }
    return {
        "primary_horizon_years": config.dynamics.recovery_primary_horizon_years,
        "level_spearman": level,
        "level_rows": level_rows,
        "velocity_spearman": velocity,
        "velocity_rows": velocity_rows,
        "migration_surprise_spearman": surprise,
        "migration_surprise_rows": surprise_rows,
        "thresholds": {
            "level": config.dynamics.recovery_minimum_level_spearman,
            "velocity": config.dynamics.recovery_minimum_velocity_spearman,
            "migration_surprise": config.dynamics.recovery_minimum_surprise_spearman,
        },
        "gates": gates,
        "status": "PASS" if all(gates.values()) else "FAIL",
        "return_injection_read": False,
    }


def build_dynamics_bundle(
    p2_dir: Path,
    p3_dir: Path,
    p4_dir: Path,
    raw_dir: Path,
    output_dir: Path,
    config: HypercubeConfig,
    *,
    scenario: str,
) -> dict[str, Any]:
    """Build and atomically publish one P5 dynamics bundle."""

    if config.project.phase != "P5":
        raise DynamicsError("Dynamics construction requires a P5 config.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite completed P5 output: {output_dir}")
    p4_validation = validate_p4_directory(p4_dir, p2_dir, p3_dir, raw_dir, config)
    if p4_validation["status"] != "PASS":
        raise DynamicsError(f"P4 input failed validation: {p4_validation['errors']}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent))
    try:
        frame, model_diagnostics = build_dynamics_frame(
            p2_dir,
            p3_dir,
            p4_dir,
            raw_dir,
            config,
            staging / "models",
        )
        _atomic_parquet(frame, staging / "frontier_dynamics.parquet")
        _atomic_csv(model_diagnostics, staging / "migration_model_diagnostics.csv")
        summary = {
            "scenario": scenario,
            "rows": int(len(frame)),
            "horizon_rows": {
                str(int(key)): int(value)
                for key, value in frame.groupby("horizon_years").size().items()
            },
            "velocity_rows": int(frame["velocity_log_odds"].notna().sum()),
            "acceleration_rows": int(frame["acceleration_log_odds"].notna().sum()),
            "migration_surprise_rows": int(frame["migration_surprise"].notna().sum()),
            "frontier_crossings": {
                str(int(key)): int(value)
                for key, value in frame["frontier_crossing"].dropna().value_counts().items()
            },
            "crowding_coverage": float(frame["crowding_density"].notna().mean()),
            "historical_success_density_coverage": float(
                frame["historical_success_density"].notna().mean()
            ),
            "model_vintage_boundary_rows": int(frame["model_vintage_changed"].sum()),
            "synthetic_truth_read": False,
            "return_test_run": False,
            "portfolio_run": False,
            "clustering_run": False,
        }
        atomic_write_json(staging / "p5_diagnostics.json", summary)
        atomic_write_json(staging / "resolved_config.json", config.model_dump(mode="json"))
        metadata = {
            "version": P5_VERSION,
            "primary_model": config.dynamics.primary_model,
            "frontier_threshold": config.dynamics.survival_probability_threshold,
            "frontier_margin": "survival log odds minus fixed-threshold log odds",
            "hazard_proxy": "-log(calibrated survival probability) / horizon years",
            "velocity": "annualized within-outer-fold change in survival log odds",
            "acceleration": "annualized within-fold change in log-odds velocity",
            "migration_surprise": "observed log-odds velocity minus expanding prior-year ridge expectation",
            "migration_controls": [
                *NUMERIC_MIGRATION_CONTROLS,
                *CATEGORICAL_MIGRATION_CONTROLS,
            ],
            "synthetic_truth_read": False,
            "return_fields_read": False,
            "archetype_transition_deferred_to": "P9",
        }
        atomic_write_json(staging / "feature_metadata.json", metadata)
        model_inputs = tuple(
            p4_dir
            / "models"
            / f"h{horizon}_fold{definition['fold']}_{config.dynamics.primary_model}.joblib"
            for horizon in config.viability.horizons_years
            for definition in fold_definitions(config, horizon)
        )
        input_paths = (
            p2_dir / "accounting_availability.parquet",
            p2_dir / "firm_month_panel.parquet",
            p3_dir / "axis_scores.parquet",
            p4_dir / "model_matrix.parquet",
            p4_dir / "oos_predictions.parquet",
            raw_dir / "factor_returns.parquet",
            *model_inputs,
        )
        manifest = {
            "schema_version": 1,
            "phase": "P5",
            "scenario": scenario,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "seed": config.project.seed,
            "inputs": [
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                    "rows": (
                        int(pq.ParquetFile(path).metadata.num_rows)
                        if path.suffix == ".parquet"
                        else None
                    ),
                }
                for path in input_paths
            ],
            "files": _inventory(staging),
            "synthetic_truth_read": False,
            "return_test_run": False,
        }
        atomic_write_json(staging / "p5_manifest.json", manifest)
        if output_dir.exists():
            output_dir.rmdir()
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def validate_p5_directory(
    output_dir: Path,
    p2_dir: Path,
    p3_dir: Path,
    p4_dir: Path,
    raw_dir: Path,
    config: HypercubeConfig,
    *,
    synthetic_truth_path: Path | None = None,
) -> dict[str, Any]:
    """Independently validate P5 timing, arithmetic, hashes, and recovery."""

    required = (
        "frontier_dynamics.parquet",
        "migration_model_diagnostics.csv",
        "p5_diagnostics.json",
        "resolved_config.json",
        "feature_metadata.json",
        "p5_manifest.json",
    )
    missing = [name for name in required if not (output_dir / name).is_file()]
    if missing:
        return {"status": "FAIL", "errors": [f"Missing P5 outputs: {missing}"], "warnings": []}
    errors: list[str] = []
    warnings: list[str] = []
    frame = pd.read_parquet(output_dir / "frontier_dynamics.parquet")
    model_diagnostics = pd.read_csv(output_dir / "migration_model_diagnostics.csv")
    metadata = json.loads((output_dir / "feature_metadata.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((output_dir / "p5_diagnostics.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "p5_manifest.json").read_text(encoding="utf-8"))

    if frame.duplicated(list(EVENT_KEY)).any():
        errors.append("Duplicate P5 event keys.")
    if set(frame["model"].unique()) != {config.dynamics.primary_model}:
        errors.append("P5 uses a model other than the frozen P4 primary model.")
    numeric = frame.select_dtypes(include=[np.number])
    if np.isinf(numeric.to_numpy(float)).any():
        errors.append("P5 contains infinite numeric values.")
    if not frame["viability_level"].between(0.0, 1.0).all():
        errors.append("A viability level lies outside [0, 1].")
    expected_log_odds = _logit(
        frame["viability_level"], config.dynamics.probability_clip
    )
    if not np.allclose(frame["viability_log_odds"], expected_log_odds):
        errors.append("Saved viability log odds do not recompute.")
    threshold_log_odds = _logit(
        np.array([config.dynamics.survival_probability_threshold]),
        config.dynamics.probability_clip,
    )[0]
    if not np.allclose(
        frame["viability_margin_log_odds"], expected_log_odds - threshold_log_odds
    ):
        errors.append("Saved frontier margins do not recompute.")
    expected_hazard = -np.log(
        np.clip(frame["viability_level"], config.dynamics.probability_clip, 1.0)
    ) / frame["horizon_years"]
    if not np.allclose(frame["annualized_constant_hazard_proxy"], expected_hazard):
        errors.append("Saved annualized hazard proxies do not recompute.")

    selected = select_oos_surface(p4_dir, config)[list(EVENT_KEY) + ["viability_level", "fold"]]
    joined = frame.merge(
        selected,
        on=list(EVENT_KEY),
        suffixes=("_saved", "_p4"),
        validate="one_to_one",
    )
    if len(joined) != len(frame) or not np.allclose(
        joined["viability_level_saved"], joined["viability_level_p4"]
    ):
        errors.append("P5 levels do not exactly trace to P4 OOS probabilities.")
    if not joined["fold_saved"].equals(joined["fold_p4"]):
        errors.append("P5 fold provenance differs from P4.")
    saved_p4 = pd.read_parquet(p4_dir / "oos_predictions.parquet")
    saved_p4 = saved_p4.loc[
        saved_p4["model"].eq(config.dynamics.primary_model),
        [*EVENT_KEY, "calibrated_survival_probability"],
    ]
    observed_trace = saved_p4.merge(
        frame[[*EVENT_KEY, "viability_level"]],
        on=list(EVENT_KEY),
        how="left",
        validate="one_to_one",
    )
    if observed_trace["viability_level"].isna().any() or not np.allclose(
        observed_trace["calibrated_survival_probability"],
        observed_trace["viability_level"],
    ):
        errors.append("P5 does not reproduce P4's saved observed-label predictions.")

    ordered = frame.sort_values(["gvkey", "horizon_years", "feature_date"]).copy()
    group = ordered.groupby(["gvkey", "horizon_years"], sort=False)
    previous_level = group["viability_log_odds"].shift(1)
    previous_fold = group["fold"].shift(1)
    elapsed = (
        pd.to_datetime(ordered["feature_date"])
        - pd.to_datetime(group["feature_date"].shift(1))
    ).dt.days / 365.25
    comparable = previous_fold.eq(ordered["fold"]) & elapsed.gt(0.0)
    expected_velocity = (ordered["viability_log_odds"] - previous_level) / elapsed
    saved_velocity = ordered["velocity_log_odds"]
    if not np.allclose(
        saved_velocity.loc[comparable], expected_velocity.loc[comparable], equal_nan=True
    ) or saved_velocity.loc[~comparable].notna().any():
        errors.append("Saved log-odds velocity violates fold-safe recomputation.")
    migration_rows = frame["migration_surprise"].notna()
    if not np.allclose(
        frame.loc[migration_rows, "migration_surprise"],
        frame.loc[migration_rows, "velocity_log_odds"]
        - frame.loc[migration_rows, "expected_velocity_log_odds"],
    ):
        errors.append("Saved migration surprise does not equal observed minus expected.")
    training_dates = pd.to_datetime(frame["migration_train_max_feature_date"])
    prediction_year_start = pd.to_datetime(
        frame["feature_year"].astype(str) + "-01-01"
    )
    if (training_dates.notna() & (training_dates >= prediction_year_start)).any():
        errors.append("Migration expectation uses the prediction year or future.")
    successful_cutoff = pd.to_datetime(frame["successful_reference_cutoff"])
    if (
        successful_cutoff.notna()
        & (successful_cutoff >= prediction_year_start)
    ).any():
        errors.append("Historical-success density uses a nonhistorical reference.")
    fitted_diagnostics = model_diagnostics.loc[model_diagnostics["status"].eq("fitted")]
    if not fitted_diagnostics.empty:
        diagnostic_dates = pd.to_datetime(fitted_diagnostics["train_max_feature_date"])
        diagnostic_start = pd.to_datetime(
            fitted_diagnostics["prediction_year"].astype(int).astype(str) + "-01-01"
        )
        if (diagnostic_dates >= diagnostic_start).any():
            errors.append("A saved migration model diagnostic includes future data.")

    if metadata.get("synthetic_truth_read") is not False or diagnostics.get(
        "synthetic_truth_read"
    ) is not False:
        errors.append("P5 construction does not certify truth exclusion.")
    if manifest.get("synthetic_truth_read") is not False:
        errors.append("P5 manifest does not certify truth exclusion.")
    if diagnostics.get("return_test_run") is not False or diagnostics.get(
        "portfolio_run"
    ) is not False:
        errors.append("A return or portfolio test is incorrectly marked as run.")
    forbidden = {"ret", "ret_total", "injected_return_alpha", "future_return"}
    if forbidden.intersection(frame.columns):
        errors.append("A return or injected-alpha field leaked into P5 output.")
    inventory = {item["name"]: item for item in manifest.get("files", [])}
    for name, record in inventory.items():
        path = output_dir / name
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            errors.append(f"Manifest hash mismatch: {name}.")
        if path.suffix == ".parquet" and int(
            pq.ParquetFile(path).metadata.num_rows
        ) != int(record.get("rows", -1)):
            errors.append(f"Manifest row-count mismatch: {name}.")
    if any(
        (output_dir / name).exists()
        for name in ("return_tests.csv", "portfolio.csv", "clusters.parquet")
    ):
        errors.append("A downstream P6-P9 output exists inside P5.")

    recovery: dict[str, Any] | None = None
    if synthetic_truth_path is not None:
        recovery = _truth_recovery(frame, synthetic_truth_path, config)
        if recovery["status"] != "PASS":
            errors.append(f"Synthetic directional recovery failed: {recovery['gates']}")
    else:
        warnings.append("Synthetic recovery not evaluated for a real-data bundle.")
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "warnings": warnings,
        "rows": {
            "frontier_dynamics": int(len(frame)),
            "velocity": int(frame["velocity_log_odds"].notna().sum()),
            "acceleration": int(frame["acceleration_log_odds"].notna().sum()),
            "migration_surprise": int(frame["migration_surprise"].notna().sum()),
        },
        "recovery": recovery,
        "models_fitted": ["expanding_migration_ridge"],
        "return_test_run": False,
        "portfolio_run": False,
        "clustering_run": False,
    }
