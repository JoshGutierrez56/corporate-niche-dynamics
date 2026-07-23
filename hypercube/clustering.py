"""Training-only descriptive archetypes and out-of-sample transitions for P9."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.cluster import HDBSCAN
from sklearn.metrics import adjusted_rand_score

from hypercube.config import HypercubeConfig
from hypercube.data import Scenario, atomic_write_json, sha256_file
from hypercube.universe import scenario_processed_dir


class ClusteringError(ValueError):
    """Raised when the P9 descriptive clustering contract fails."""


P9_VERSION = "hypercube-descriptive-archetypes-v1"
AXIS_COLUMNS = (
    "anchored_demand_strength_pricing_power",
    "anchored_competitive_defensibility",
    "anchored_innovation_intensity",
    "anchored_go_to_market_efficiency",
    "anchored_unit_economics_profit_quality",
    "anchored_scalability_capital_efficiency",
)
KEY_COLUMNS = ("gvkey", "permno", "datadate", "feature_date")


def scenario_p9_dir(config: HypercubeConfig, scenario: Scenario) -> Path:
    """Resolve one scenario's P9 output directory."""

    return scenario_processed_dir(config, scenario) / "p9"


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.to_parquet(name, index=False, compression="zstd")
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


def _atomic_joblib(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        joblib.dump(payload, name, compress=3)
        os.replace(name, path)
    except Exception:
        Path(name).unlink(missing_ok=True)
        raise


def _file_record(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    if path.suffix == ".parquet":
        record["rows"] = pq.ParquetFile(path).metadata.num_rows
    elif path.suffix == ".csv":
        record["rows"] = max(
            0, sum(1 for _ in path.open("r", encoding="utf-8")) - 1
        )
    return record


def _excel_label(index: int) -> str:
    value = index + 1
    letters = ""
    while value:
        value, remainder = divmod(value - 1, 26)
        letters = chr(65 + remainder) + letters
    return f"Archetype {letters}"


def _fit_transform(
    frame: pd.DataFrame,
    config: HypercubeConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    train = frame.loc[frame["fyear"].le(config.clustering.training_end_year)]
    if len(train) < config.clustering.minimum_cluster_size * 2:
        raise ClusteringError("P9 has too few training observations.")
    median = train[list(AXIS_COLUMNS)].median()
    mad = (train[list(AXIS_COLUMNS)] - median).abs().median()
    scale = (1.4826 * mad).replace(0.0, 1.0).fillna(1.0)
    filled = frame[list(AXIS_COLUMNS)].fillna(median)
    transformed = ((filled - median) / scale).clip(-8.0, 8.0).to_numpy(float)
    if not np.isfinite(transformed).all():
        raise ClusteringError("P9 transformed axes contain non-finite values.")
    return transformed, {
        "median": median.to_dict(),
        "scale": scale.to_dict(),
        "axis_columns": list(AXIS_COLUMNS),
    }


def _sample_training_indices(
    frame: pd.DataFrame,
    config: HypercubeConfig,
    *,
    seed_offset: int = 0,
    fraction: float = 1.0,
) -> np.ndarray:
    eligible = np.flatnonzero(
        frame["fyear"].to_numpy(int) <= config.clustering.training_end_year
    )
    desired = min(
        len(eligible),
        config.clustering.maximum_training_rows,
        max(
            config.clustering.minimum_cluster_size * 2,
            int(math.floor(len(eligible) * fraction)),
        ),
    )
    if desired == len(eligible):
        return eligible
    rng = np.random.default_rng(config.project.seed + seed_offset)
    return np.sort(rng.choice(eligible, size=desired, replace=False))


def _fit_density_model(
    values: np.ndarray,
    sample_indices: np.ndarray,
    config: HypercubeConfig,
) -> tuple[HDBSCAN, dict[int, np.ndarray], dict[int, float], np.ndarray]:
    model = HDBSCAN(
        min_cluster_size=config.clustering.minimum_cluster_size,
        min_samples=config.clustering.minimum_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        allow_single_cluster=False,
        n_jobs=1,
        copy=True,
    )
    model.fit(values[sample_indices])
    labels = model.labels_.astype(int)
    clusters = sorted(int(value) for value in np.unique(labels) if value >= 0)
    if len(clusters) < 2:
        raise ClusteringError(
            "P9 density fit found fewer than two archetypes; the descriptive gate holds."
        )
    centroids: dict[int, np.ndarray] = {}
    radii: dict[int, float] = {}
    for label in clusters:
        points = values[sample_indices][labels == label]
        centroid = np.median(points, axis=0)
        distance = np.linalg.norm(points - centroid, axis=1)
        centroids[label] = centroid
        radii[label] = max(
            float(np.quantile(distance, config.clustering.assignment_radius_quantile)),
            1e-9,
        )
    return model, centroids, radii, labels


def _canonical_mapping(
    labels: np.ndarray,
    centroids: dict[int, np.ndarray],
) -> dict[int, str]:
    ordered = sorted(
        centroids,
        key=lambda label: (
            -int(np.sum(labels == label)),
            *np.round(centroids[label], 8).tolist(),
        ),
    )
    return {label: _excel_label(index) for index, label in enumerate(ordered)}


def _assign_nearest(
    values: np.ndarray,
    centroids: dict[int, np.ndarray],
    radii: dict[int, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = np.array(sorted(centroids), dtype=int)
    matrix = np.stack([centroids[label] for label in labels])
    distances = np.linalg.norm(values[:, None, :] - matrix[None, :, :], axis=2)
    nearest_index = np.argmin(distances, axis=1)
    nearest_label = labels[nearest_index]
    nearest_distance = distances[np.arange(len(values)), nearest_index]
    nearest_radius = np.array([radii[int(label)] for label in nearest_label])
    accepted = nearest_distance <= nearest_radius
    assigned = np.where(accepted, nearest_label, -1)
    confidence = np.where(
        accepted,
        np.clip(1.0 - nearest_distance / nearest_radius, 0.0, 1.0),
        0.0,
    )
    return assigned.astype(int), confidence.astype(float), nearest_distance


def build_archetype_assignments(
    axes: pd.DataFrame,
    config: HypercubeConfig,
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame]:
    """Fit HDBSCAN on frozen training rows and assign later observations."""

    required = {*KEY_COLUMNS, "fyear", *AXIS_COLUMNS}
    missing = sorted(required - set(axes.columns))
    if missing:
        raise ClusteringError(f"P9 axis input is missing columns: {missing}")
    frame = axes.copy()
    for column in ("datadate", "feature_date"):
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame = frame.sort_values(["feature_date", "gvkey"], kind="stable").reset_index(
        drop=True
    )
    values, transform = _fit_transform(frame, config)
    sample_indices = _sample_training_indices(frame, config)
    model, centroids, radii, sampled_labels = _fit_density_model(
        values,
        sample_indices,
        config,
    )
    assigned, confidence, distance = _assign_nearest(values, centroids, radii)
    probabilities = getattr(
        model,
        "probabilities_",
        np.ones(len(sample_indices), dtype=float),
    )
    assigned[sample_indices] = sampled_labels
    confidence[sample_indices] = probabilities
    distance[sample_indices] = np.nan
    mapping = _canonical_mapping(sampled_labels, centroids)
    result = frame[
        [
            "gvkey",
            "permno",
            "datadate",
            "fyear",
            "availability_date",
            "formation_date",
            "feature_date",
            "sic2",
            "market_cap_millions",
            *AXIS_COLUMNS,
        ]
    ].copy()
    result["cluster_id"] = assigned
    result["archetype"] = [
        mapping.get(int(label), "Noise / Unassigned") for label in assigned
    ]
    result["assignment_confidence"] = confidence
    result["distance_to_training_centroid"] = distance
    result["sample_role"] = np.where(
        result["fyear"].le(config.clustering.training_end_year),
        "training_period",
        "out_of_sample",
    )
    sampled_flag = np.zeros(len(result), dtype=bool)
    sampled_flag[sample_indices] = True
    result["used_to_fit_clustering"] = sampled_flag
    result["assignment_source"] = np.where(
        sampled_flag,
        "hdbscan_training_fit",
        "nearest_training_centroid_with_radius",
    )
    result["is_noise_or_unassigned"] = result["cluster_id"].eq(-1)
    centroid_rows = []
    for label, centroid in centroids.items():
        row: dict[str, Any] = {
            "cluster_id": label,
            "archetype": mapping[label],
            "training_radius": radii[label],
            "training_members": int(np.sum(sampled_labels == label)),
        }
        for column, value in zip(AXIS_COLUMNS, centroid, strict=True):
            row[f"scaled_centroid_{column}"] = float(value)
        centroid_rows.append(row)
    fit = {
        "version": P9_VERSION,
        "model": model,
        "transform": transform,
        "sample_indices": sample_indices,
        "centroids": centroids,
        "radii": radii,
        "canonical_mapping": mapping,
        "training_end_year": config.clustering.training_end_year,
    }
    return result, fit, pd.DataFrame(centroid_rows)


def cluster_stability(
    axes: pd.DataFrame,
    assignments: pd.DataFrame,
    fit: dict[str, Any],
    config: HypercubeConfig,
) -> pd.DataFrame:
    """Refit bounded subsamples and report label-invariant agreement."""

    median = pd.Series(fit["transform"]["median"])
    scale = pd.Series(fit["transform"]["scale"])
    values = (
        (axes[list(AXIS_COLUMNS)].fillna(median) - median) / scale
    ).clip(-8.0, 8.0).to_numpy(float)
    base = assignments["cluster_id"].to_numpy(int)
    evaluation = np.flatnonzero(
        axes["fyear"].to_numpy(int) <= config.clustering.training_end_year
    )
    rows = []
    for repetition in range(config.clustering.stability_repetitions):
        indices = _sample_training_indices(
            axes,
            config,
            seed_offset=10_000 + repetition,
            fraction=config.clustering.stability_sample_fraction,
        )
        try:
            _, centroids, radii, fitted_labels = _fit_density_model(
                values,
                indices,
                config,
            )
            alternative, _, _ = _assign_nearest(values[evaluation], centroids, radii)
            ari = float(adjusted_rand_score(base[evaluation], alternative))
            rows.append(
                {
                    "repetition": repetition,
                    "sample_rows": int(len(indices)),
                    "clusters": int(len(set(fitted_labels) - {-1})),
                    "noise_rate_evaluation": float(np.mean(alternative == -1)),
                    "adjusted_rand_index": ari,
                    "fit_status": "PASS",
                }
            )
        except ClusteringError:
            rows.append(
                {
                    "repetition": repetition,
                    "sample_rows": int(len(indices)),
                    "clusters": 0,
                    "noise_rate_evaluation": 1.0,
                    "adjusted_rand_index": np.nan,
                    "fit_status": "NO_STABLE_CLUSTERS",
                }
            )
    return pd.DataFrame(rows)


def descriptive_profiles(
    assignments: pd.DataFrame,
    p4_dir: Path,
    p5_dir: Path,
    p7_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Describe archetypes using saved outcomes without refitting clusters."""

    labels = pd.read_parquet(p4_dir / "viability_labels.parquet")
    labels = labels.loc[labels["horizon_years"].eq(5)].copy()
    dynamics = pd.read_parquet(
        p5_dir / "frontier_dynamics.parquet",
        columns=[
            "gvkey",
            "datadate",
            "feature_date",
            "horizon_years",
            "viability_level",
        ],
    )
    dynamics = dynamics.loc[dynamics["horizon_years"].eq(5)].drop(
        columns="horizon_years"
    )
    targets = pd.read_parquet(
        p7_dir / "forward_return_targets.parquet",
        columns=[
            "gvkey",
            "datadate",
            "feature_date",
            "horizon_months",
            "target_valid",
            "forward_excess_return",
        ],
    )
    targets = targets.loc[targets["horizon_months"].eq(6)].copy()
    frame = assignments.merge(
        labels[
            [
                "gvkey",
                "datadate",
                "feature_date",
                "failure_within_horizon",
                "label_status",
            ]
        ],
        on=["gvkey", "datadate", "feature_date"],
        how="left",
        validate="one_to_one",
    ).merge(
        dynamics,
        on=["gvkey", "datadate", "feature_date"],
        how="left",
        validate="one_to_one",
    ).merge(
        targets.drop(columns="horizon_months"),
        on=["gvkey", "datadate", "feature_date"],
        how="left",
        validate="one_to_one",
    )
    rows: list[dict[str, Any]] = []
    for (archetype, sample_role), group in frame.groupby(
        ["archetype", "sample_role"], observed=True, sort=True
    ):
        valid_return = group["target_valid"].eq(True)
        row: dict[str, Any] = {
            "archetype": archetype,
            "sample_role": sample_role,
            "observations": int(len(group)),
            "firms": int(group["gvkey"].nunique()),
            "noise_or_unassigned": bool(archetype == "Noise / Unassigned"),
            "mean_assignment_confidence": float(
                group["assignment_confidence"].mean()
            ),
            "observed_five_year_labels": int(
                group["label_status"].eq("observed").sum()
            ),
            "five_year_failure_rate": float(
                group.loc[
                    group["label_status"].eq("observed"),
                    "failure_within_horizon",
                ].mean()
            ),
            "viability_observations": int(group["viability_level"].notna().sum()),
            "mean_oos_viability_level": float(group["viability_level"].mean()),
            "valid_six_month_returns": int(valid_return.sum()),
            "mean_six_month_excess_return": float(
                group.loc[
                    valid_return,
                    "forward_excess_return",
                ].mean()
            ),
        }
        for column in AXIS_COLUMNS:
            row[f"mean_{column}"] = float(group[column].mean())
            row[f"median_{column}"] = float(group[column].median())
        rows.append(row)
    sizes = (
        assignments.groupby(
            ["archetype", "sample_role", "is_noise_or_unassigned"],
            observed=True,
        )
        .agg(
            observations=("gvkey", "size"),
            firms=("gvkey", "nunique"),
            first_year=("fyear", "min"),
            last_year=("fyear", "max"),
            mean_assignment_confidence=("assignment_confidence", "mean"),
        )
        .reset_index()
    )
    return pd.DataFrame(rows), sizes


def transition_outputs(
    assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute consecutive-firm archetype transitions and persistence."""

    ordered = assignments.sort_values(["gvkey", "feature_date"], kind="stable").copy()
    ordered["prior_archetype"] = ordered.groupby("gvkey", observed=True)[
        "archetype"
    ].shift()
    ordered["prior_feature_date"] = ordered.groupby("gvkey", observed=True)[
        "feature_date"
    ].shift()
    valid = ordered.loc[ordered["prior_archetype"].notna()].copy()
    transitions = (
        valid.groupby(
            ["sample_role", "prior_archetype", "archetype"], observed=True
        )
        .size()
        .rename("transitions")
        .reset_index()
    )
    totals = transitions.groupby(
        ["sample_role", "prior_archetype"], observed=True
    )["transitions"].transform("sum")
    transitions["transition_probability"] = transitions["transitions"] / totals
    persistence = (
        valid.assign(
            persisted=lambda value: value["archetype"].eq(value["prior_archetype"])
        )
        .groupby(["sample_role", "prior_archetype"], observed=True)
        .agg(
            transitions=("persisted", "size"),
            persistence_rate=("persisted", "mean"),
            median_gap_days=(
                "feature_date",
                lambda values: float(
                    np.median(
                        (
                            values
                            - valid.loc[values.index, "prior_feature_date"]
                        ).dt.days
                    )
                ),
            ),
        )
        .reset_index()
        .rename(columns={"prior_archetype": "archetype"})
    )
    return transitions, persistence


def build_clustering_bundle(
    p3_dir: Path,
    p4_dir: Path,
    p5_dir: Path,
    p7_dir: Path,
    output_dir: Path,
    config: HypercubeConfig,
    *,
    scenario: str,
) -> dict[str, Any]:
    """Build one atomic P9 descriptive bundle without changing return tests."""

    if config.project.phase != "P9":
        raise ClusteringError("Descriptive clustering requires a P9 config.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite completed P9 output: {output_dir}")
    axes = pd.read_parquet(p3_dir / "axis_scores.parquet")
    assignments, fit, centroids = build_archetype_assignments(axes, config)
    stability = cluster_stability(axes, assignments, fit, config)
    profiles, sizes = descriptive_profiles(assignments, p4_dir, p5_dir, p7_dir)
    transitions, persistence = transition_outputs(assignments)
    noise_rate = float(assignments["is_noise_or_unassigned"].mean())
    cluster_count = int(
        assignments.loc[
            assignments["used_to_fit_clustering"]
            & ~assignments["is_noise_or_unassigned"],
            "cluster_id",
        ].nunique()
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".p9-staging-", dir=output_dir.parent))
    try:
        _atomic_parquet(assignments, staging / "archetype_assignments.parquet")
        _atomic_csv(centroids, staging / "training_cluster_centroids.csv")
        _atomic_csv(sizes, staging / "cluster_sizes.csv")
        _atomic_csv(profiles, staging / "archetype_profiles.csv")
        _atomic_csv(transitions, staging / "transition_matrix.csv")
        _atomic_csv(persistence, staging / "cluster_persistence.csv")
        _atomic_csv(stability, staging / "cluster_stability.csv")
        _atomic_joblib(fit, staging / "training_cluster_model.joblib")
        metadata = {
            "schema_version": 1,
            "phase": "P9",
            "version": P9_VERSION,
            "scenario": scenario,
            "representation": config.clustering.representation,
            "training_end_year": config.clustering.training_end_year,
            "fit_rows": int(assignments["used_to_fit_clustering"].sum()),
            "assignment_rows": int(len(assignments)),
            "cluster_count": cluster_count,
            "noise_or_unassigned_rate": noise_rate,
            "descriptive_only": True,
            "return_models_refit": False,
            "synthetic_truth_read": False,
            "labels_are_neutral": True,
        }
        atomic_write_json(staging / "clustering_metadata.json", metadata)
        atomic_write_json(staging / "resolved_config.json", config.model_dump(mode="json"))
        input_paths = [
            p3_dir / "axis_scores.parquet",
            p4_dir / "viability_labels.parquet",
            p5_dir / "frontier_dynamics.parquet",
            p7_dir / "forward_return_targets.parquet",
        ]
        output_paths = [
            path
            for path in staging.iterdir()
            if path.is_file() and path.name != "p9_manifest.json"
        ]
        manifest = {
            "schema_version": 1,
            "phase": "P9",
            "scenario": scenario,
            "seed": config.project.seed,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": [_file_record(path) for path in input_paths],
            "outputs": [_file_record(path) for path in output_paths],
            "rows": {
                "archetype_assignments": int(len(assignments)),
                "archetype_profiles": int(len(profiles)),
                "transition_cells": int(len(transitions)),
                "stability_repetitions": int(len(stability)),
            },
            "descriptive_only": True,
            "return_models_refit": False,
            "synthetic_truth_read": False,
        }
        atomic_write_json(staging / "p9_manifest.json", manifest)
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return metadata


def validate_p9_directory(
    output_dir: Path,
    p3_dir: Path,
    config: HypercubeConfig,
    *,
    scenario: str,
) -> dict[str, Any]:
    """Independently validate P9 hashes, training boundary, and transitions."""

    required = (
        "archetype_assignments.parquet",
        "training_cluster_centroids.csv",
        "cluster_sizes.csv",
        "archetype_profiles.csv",
        "transition_matrix.csv",
        "cluster_persistence.csv",
        "cluster_stability.csv",
        "training_cluster_model.joblib",
        "clustering_metadata.json",
        "resolved_config.json",
        "p9_manifest.json",
    )
    errors = [
        f"Missing P9 file: {name}"
        for name in required
        if not (output_dir / name).is_file()
    ]
    if errors:
        return {"status": "FAIL", "errors": errors, "warnings": []}
    manifest = json.loads((output_dir / "p9_manifest.json").read_text(encoding="utf-8"))
    for record in manifest["inputs"]:
        path = Path(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            errors.append(f"P9 input hash mismatch: {path.name}")
    for record in manifest["outputs"]:
        path = output_dir / Path(record["path"]).name
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            errors.append(f"P9 output hash mismatch: {path.name}")
            continue
        if path.suffix == ".parquet" and pq.ParquetFile(path).metadata.num_rows != record["rows"]:
            errors.append(f"P9 parquet row-count mismatch: {path.name}")
        if path.suffix == ".csv":
            rows = max(0, sum(1 for _ in path.open("r", encoding="utf-8")) - 1)
            if rows != record["rows"]:
                errors.append(f"P9 CSV row-count mismatch: {path.name}")
    metadata = json.loads(
        (output_dir / "clustering_metadata.json").read_text(encoding="utf-8")
    )
    if metadata.get("descriptive_only") is not True:
        errors.append("P9 is not marked descriptive-only.")
    if metadata.get("return_models_refit") is not False:
        errors.append("P9 refit a return model.")
    if metadata.get("synthetic_truth_read") is not False:
        errors.append("P9 read synthetic truth.")
    assignments = pd.read_parquet(output_dir / "archetype_assignments.parquet")
    axes = pd.read_parquet(p3_dir / "axis_scores.parquet")
    if len(assignments) != len(axes):
        errors.append("P9 assignment rows do not equal frozen P3 axis rows.")
    if assignments.duplicated(list(KEY_COLUMNS)).any():
        errors.append("P9 archetype assignment keys are duplicated.")
    fit_rows = assignments.loc[assignments["used_to_fit_clustering"].astype(bool)]
    if fit_rows["fyear"].gt(config.clustering.training_end_year).any():
        errors.append("P9 fitted clustering on a post-training observation.")
    if assignments["assignment_confidence"].notna().all() is False:
        errors.append("P9 assignment confidence is missing.")
    if not assignments["assignment_confidence"].between(0.0, 1.0).all():
        errors.append("P9 assignment confidence is outside [0, 1].")
    if not assignments.loc[
        assignments["cluster_id"].eq(-1), "archetype"
    ].eq("Noise / Unassigned").all():
        errors.append("P9 noise rows have a substantive label.")
    transitions = pd.read_csv(output_dir / "transition_matrix.csv")
    totals = transitions.groupby(
        ["sample_role", "prior_archetype"], observed=True
    )["transition_probability"].sum()
    if not np.allclose(totals, 1.0, rtol=1e-10, atol=1e-12):
        errors.append("P9 transition probabilities do not sum to one.")
    stability = pd.read_csv(output_dir / "cluster_stability.csv")
    if len(stability) != config.clustering.stability_repetitions:
        errors.append("P9 stability repetitions are incomplete.")
    model_payload = joblib.load(output_dir / "training_cluster_model.joblib")
    if model_payload["training_end_year"] != config.clustering.training_end_year:
        errors.append("P9 saved model has the wrong training cutoff.")
    warnings = []
    mean_ari = float(stability["adjusted_rand_index"].mean())
    if np.isnan(mean_ari):
        warnings.append(
            "Every bounded stability refit failed to recover two clusters."
        )
    elif mean_ari < 0.50:
        warnings.append("Archetype stability is weak; interpret clusters cautiously.")
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "warnings": warnings,
        "scenario": scenario,
        "rows": {
            "assignments": int(len(assignments)),
            "fit_rows": int(len(fit_rows)),
            "clusters": int(
                assignments.loc[
                    fit_rows.index, "cluster_id"
                ].loc[lambda value: value.ge(0)].nunique()
            ),
            "transition_cells": int(len(transitions)),
        },
        "noise_or_unassigned_rate": float(
            assignments["is_noise_or_unassigned"].mean()
        ),
        "mean_stability_ari": mean_ari,
        "descriptive_only": True,
        "synthetic_truth_read": False,
    }
