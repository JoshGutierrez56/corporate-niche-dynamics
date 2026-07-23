"""Versioned exploratory calibration for a detectable synthetic return injection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t as student_t


P12E_VERSION = "hypercube-alpha-power-calibration-v1"
P15E_VERSION = "hypercube-locked-proxy-power-calibration-v1"
EVENT_KEY = ("gvkey", "datadate", "fyear")
DEFAULT_MULTIPLIERS = (1.0, 2.0, 4.0, 6.0, 8.0, 10.0)


class PowerCalibrationError(ValueError):
    """Raised when the frozen P7/truth inputs cannot support calibration."""


def _load_primary(
    targets_path: Path,
    truth_path: Path,
    primary_horizon_months: int,
) -> pd.DataFrame:
    targets = pd.read_parquet(
        targets_path,
        columns=[
            *EVENT_KEY,
            "horizon_months",
            "target_valid",
            "migration_surprise",
            "forward_excess_return",
        ],
    )
    primary = targets.loc[
        targets["horizon_months"].eq(primary_horizon_months)
        & targets["target_valid"].astype(bool)
        & targets["migration_surprise"].notna()
    ].copy()
    truth = pd.read_parquet(
        truth_path,
        columns=[*EVENT_KEY, "injected_return_alpha"],
    )
    joined = primary.merge(
        truth,
        on=list(EVENT_KEY),
        how="left",
        validate="one_to_one",
    )
    if joined.empty:
        raise PowerCalibrationError("No primary migration-alpha targets exist.")
    if joined[
        ["migration_surprise", "forward_excess_return", "injected_return_alpha"]
    ].isna().any().any():
        raise PowerCalibrationError("P7 targets and synthetic truth do not reconcile.")
    if joined["gvkey"].nunique() < 2:
        raise PowerCalibrationError("At least two issuer clusters are required.")
    return joined


def _clustered_regression(
    outcome: np.ndarray,
    predictor: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float]:
    design = np.column_stack([np.ones(len(predictor)), predictor])
    cross = design.T @ design
    if np.linalg.matrix_rank(cross) < 2:
        raise PowerCalibrationError("The bootstrap predictor has zero variance.")
    inverse = np.linalg.inv(cross)
    coefficients = inverse @ design.T @ outcome
    residual = outcome - design @ coefficients
    codes, labels = pd.factorize(groups, sort=False)
    cluster_count = len(labels)
    if cluster_count < 2:
        raise PowerCalibrationError("A bootstrap draw has fewer than two clusters.")
    score_intercept = np.bincount(codes, weights=residual, minlength=cluster_count)
    score_slope = np.bincount(
        codes,
        weights=residual * predictor,
        minlength=cluster_count,
    )
    scores = np.column_stack([score_intercept, score_slope])
    correction = (cluster_count / (cluster_count - 1.0)) * (
        (len(outcome) - 1.0) / (len(outcome) - design.shape[1])
    )
    covariance = correction * inverse @ (scores.T @ scores) @ inverse
    standard_error = float(np.sqrt(max(covariance[1, 1], 0.0)))
    slope = float(coefficients[1])
    t_stat = slope / standard_error if standard_error > 0.0 else np.inf
    p_value = float(2.0 * student_t.sf(abs(t_stat), df=cluster_count - 1))
    return {
        "slope": slope,
        "standard_error": standard_error,
        "t_stat": float(t_stat),
        "p_value": p_value,
    }


def _bootstrap_indices(
    groups: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    unique = pd.unique(groups)
    locations = {group: np.flatnonzero(groups == group) for group in unique}
    draws = rng.choice(unique, size=len(unique), replace=True)
    index_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    for draw_id, group in enumerate(draws):
        indices = locations[group]
        index_parts.append(indices)
        group_parts.append(np.full(len(indices), draw_id, dtype=np.int64))
    return np.concatenate(index_parts), np.concatenate(group_parts)


def _bootstrap_metrics(
    frame: pd.DataFrame,
    multipliers: Sequence[float],
    *,
    replicates: int,
    seed_start: int,
    expected_horizon_fraction: float,
) -> tuple[pd.DataFrame, float]:
    observable = frame["migration_surprise"].to_numpy(float)
    injected = frame["injected_return_alpha"].to_numpy(float)
    observed_return = frame["forward_excess_return"].to_numpy(float)
    groups = frame["gvkey"].astype(str).to_numpy()
    residual = observed_return - expected_horizon_fraction * injected
    metrics: dict[float, dict[str, list[float]]] = {
        float(multiplier): {
            "oracle_detected": [],
            "observable_detected": [],
            "oracle_ic": [],
            "observable_ic": [],
            "oracle_slope": [],
            "observable_slope": [],
        }
        for multiplier in multipliers
    }
    null_false_positives: list[float] = []
    for offset in range(replicates):
        rng = np.random.default_rng(seed_start + offset)
        indices, bootstrap_groups = _bootstrap_indices(groups, rng)
        boot_observable = observable[indices]
        boot_injected = injected[indices]
        boot_residual = residual[indices]
        null_fit = _clustered_regression(
            boot_residual,
            boot_observable,
            bootstrap_groups,
        )
        null_false_positives.append(
            float(null_fit["slope"] > 0.0 and null_fit["p_value"] < 0.05)
        )
        for multiplier in multipliers:
            key = float(multiplier)
            scaled_injected = key * boot_injected
            simulated_return = (
                boot_residual + expected_horizon_fraction * scaled_injected
            )
            oracle_fit = _clustered_regression(
                simulated_return,
                scaled_injected,
                bootstrap_groups,
            )
            observable_fit = _clustered_regression(
                simulated_return,
                boot_observable,
                bootstrap_groups,
            )
            metrics[key]["oracle_detected"].append(
                float(oracle_fit["slope"] > 0.0 and oracle_fit["p_value"] < 0.05)
            )
            metrics[key]["observable_detected"].append(
                float(
                    observable_fit["slope"] > 0.0
                    and observable_fit["p_value"] < 0.05
                )
            )
            metrics[key]["oracle_ic"].append(
                float(spearmanr(scaled_injected, simulated_return).statistic)
            )
            metrics[key]["observable_ic"].append(
                float(spearmanr(boot_observable, simulated_return).statistic)
            )
            metrics[key]["oracle_slope"].append(oracle_fit["slope"])
            metrics[key]["observable_slope"].append(observable_fit["slope"])

    rows = []
    null_rate = float(np.mean(null_false_positives))
    for multiplier in multipliers:
        key = float(multiplier)
        values = metrics[key]
        rows.append(
            {
                "multiplier": key,
                "replicates": int(replicates),
                "seed_start": int(seed_start),
                "oracle_detection_rate": float(np.mean(values["oracle_detected"])),
                "observable_detection_rate": float(
                    np.mean(values["observable_detected"])
                ),
                "null_observable_false_positive_rate": null_rate,
                "median_oracle_rank_ic": float(np.median(values["oracle_ic"])),
                "median_observable_rank_ic": float(
                    np.median(values["observable_ic"])
                ),
                "median_oracle_slope": float(np.median(values["oracle_slope"])),
                "median_observable_slope": float(
                    np.median(values["observable_slope"])
                ),
            }
        )
    return pd.DataFrame(rows), null_rate


def calibrate_alpha_injection_power(
    targets_path: Path,
    truth_path: Path,
    *,
    multipliers: Sequence[float] = DEFAULT_MULTIPLIERS,
    calibration_replicates: int = 100,
    evaluation_replicates: int = 50,
    calibration_seed_start: int = 12000,
    evaluation_seed_start: int = 22000,
    primary_horizon_months: int = 6,
    migration_decay_months: int = 12,
    base_monthly_alpha: float = 0.004,
    minimum_oracle_detection_rate: float = 0.80,
    minimum_observable_detection_rate: float = 0.80,
    maximum_null_false_positive_rate: float = 0.10,
    minimum_median_oracle_rank_ic: float = 0.04,
    minimum_median_observable_rank_ic: float = 0.01,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Select the smallest power-qualified injection multiplier, then hold it out."""

    if not multipliers or any(float(item) <= 0.0 for item in multipliers):
        raise PowerCalibrationError("All candidate multipliers must be positive.")
    if calibration_replicates < 2 or evaluation_replicates < 2:
        raise PowerCalibrationError("At least two calibration/evaluation draws are required.")
    frame = _load_primary(targets_path, truth_path, primary_horizon_months)
    expected_fraction = float(
        sum(np.exp(-offset / 4.0) for offset in range(primary_horizon_months))
        / sum(np.exp(-offset / 4.0) for offset in range(migration_decay_months))
    )
    calibration, null_rate = _bootstrap_metrics(
        frame,
        multipliers,
        replicates=calibration_replicates,
        seed_start=calibration_seed_start,
        expected_horizon_fraction=expected_fraction,
    )
    calibration["passes_oracle_detection"] = calibration[
        "oracle_detection_rate"
    ].ge(minimum_oracle_detection_rate)
    calibration["passes_observable_detection"] = calibration[
        "observable_detection_rate"
    ].ge(minimum_observable_detection_rate)
    calibration["passes_null_control"] = calibration[
        "null_observable_false_positive_rate"
    ].le(maximum_null_false_positive_rate)
    calibration["passes_oracle_ic"] = calibration["median_oracle_rank_ic"].ge(
        minimum_median_oracle_rank_ic
    )
    calibration["passes_observable_ic"] = calibration[
        "median_observable_rank_ic"
    ].ge(minimum_median_observable_rank_ic)
    gate_columns = [
        "passes_oracle_detection",
        "passes_observable_detection",
        "passes_null_control",
        "passes_oracle_ic",
        "passes_observable_ic",
    ]
    calibration["passes_all_calibration_gates"] = calibration[gate_columns].all(axis=1)
    passing = calibration.loc[calibration["passes_all_calibration_gates"]]
    selected = float(passing["multiplier"].min()) if not passing.empty else None
    evaluation_row: dict[str, Any] | None = None
    evaluation_pass = False
    if selected is not None:
        evaluation, _ = _bootstrap_metrics(
            frame,
            [selected],
            replicates=evaluation_replicates,
            seed_start=evaluation_seed_start,
            expected_horizon_fraction=expected_fraction,
        )
        evaluation_row = evaluation.iloc[0].to_dict()
        evaluation_pass = bool(
            evaluation_row["oracle_detection_rate"]
            >= minimum_oracle_detection_rate
            and evaluation_row["observable_detection_rate"]
            >= minimum_observable_detection_rate
            and evaluation_row["null_observable_false_positive_rate"]
            <= maximum_null_false_positive_rate
            and evaluation_row["median_oracle_rank_ic"]
            >= minimum_median_oracle_rank_ic
            and evaluation_row["median_observable_rank_ic"]
            >= minimum_median_observable_rank_ic
        )

    summary: dict[str, Any] = {
        "schema_version": 1,
        "phase": "P12E",
        "version": P12E_VERSION,
        "status": (
            "PASS" if selected is not None and evaluation_pass else "NO_GO"
        ),
        "interpretation": (
            "POWER_CALIBRATED_FOR_NEW_SYNTHETIC_CANARY"
            if selected is not None and evaluation_pass
            else "NO_CANDIDATE_PASSED_HELD_OUT_POWER_GATES"
        ),
        "confirmatory_gate": False,
        "linearized_existing_paths_only": True,
        "new_synthetic_scenario_generated": False,
        "frozen_p0_p10_outputs_modified": False,
        "real_data_run": False,
        "rows": int(len(frame)),
        "firms": int(frame["gvkey"].nunique()),
        "candidate_multipliers": [float(item) for item in multipliers],
        "selected_multiplier": selected,
        "proposed_migration_alpha_monthly": (
            float(base_monthly_alpha * selected)
            if selected is not None and evaluation_pass
            else None
        ),
        "base_migration_alpha_monthly": float(base_monthly_alpha),
        "expected_six_month_fraction_of_total_injection": expected_fraction,
        "calibration_replicates": int(calibration_replicates),
        "evaluation_replicates": int(evaluation_replicates),
        "calibration_seed_start": int(calibration_seed_start),
        "evaluation_seed_start": int(evaluation_seed_start),
        "thresholds": {
            "minimum_oracle_detection_rate": minimum_oracle_detection_rate,
            "minimum_observable_detection_rate": minimum_observable_detection_rate,
            "maximum_null_false_positive_rate": maximum_null_false_positive_rate,
            "minimum_median_oracle_rank_ic": minimum_median_oracle_rank_ic,
            "minimum_median_observable_rank_ic": minimum_median_observable_rank_ic,
        },
        "calibration_null_observable_false_positive_rate": null_rate,
        "evaluation": evaluation_row,
        "held_out_evaluation_pass": evaluation_pass,
        "stop_before_new_synthetic_generation": True,
    }
    return summary, calibration


def validate_power_calibration_outputs(
    targets_path: Path,
    truth_path: Path,
    summary_path: Path,
    calibration_path: Path,
    *,
    calibration_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute the deterministic P12E bootstrap protocol and compare outputs."""

    kwargs = calibration_kwargs or {}
    expected_summary, expected_calibration = calibrate_alpha_injection_power(
        targets_path,
        truth_path,
        **kwargs,
    )
    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    saved_calibration = pd.read_csv(calibration_path)
    errors: list[str] = []
    for key, expected in expected_summary.items():
        if key == "generated_at_utc":
            continue
        saved = saved_summary.get(key)
        if isinstance(expected, float):
            if saved is None or not np.isclose(
                float(saved), expected, rtol=1e-12, atol=1e-12
            ):
                errors.append(f"Summary mismatch: {key}")
        elif saved != expected:
            errors.append(f"Summary mismatch: {key}")
    try:
        pd.testing.assert_frame_equal(
            saved_calibration,
            expected_calibration,
            check_dtype=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as exc:
        errors.append(f"Calibration table mismatch: {exc}")
    return {
        "schema_version": 1,
        "phase": "P12E",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "interpretation": expected_summary["interpretation"],
        "selected_multiplier": expected_summary["selected_multiplier"],
        "held_out_evaluation_pass": expected_summary["held_out_evaluation_pass"],
        "new_synthetic_scenario_generated": False,
        "frozen_p0_p10_outputs_modified": False,
        "real_data_run": False,
    }


def _load_locked_proxy_primary(
    targets_path: Path,
    truth_path: Path,
    candidates_path: Path,
    *,
    selected_candidate: str,
    primary_horizon_months: int,
    minimum_year: int,
    maximum_year: int,
) -> pd.DataFrame:
    targets = pd.read_parquet(
        targets_path,
        columns=[
            *EVENT_KEY,
            "horizon_months",
            "target_valid",
            "migration_surprise",
            "forward_excess_return",
        ],
    )
    primary = targets.loc[
        targets["horizon_months"].eq(primary_horizon_months)
        & targets["target_valid"].astype(bool)
        & targets["migration_surprise"].notna()
        & targets["fyear"].between(minimum_year, maximum_year)
    ].copy()
    candidates = pd.read_parquet(
        candidates_path,
        columns=[*EVENT_KEY, selected_candidate],
    )
    truth = pd.read_parquet(
        truth_path,
        columns=[*EVENT_KEY, "injected_return_alpha"],
    )
    joined = (
        primary.merge(
            candidates,
            on=list(EVENT_KEY),
            how="left",
            validate="one_to_one",
        )
        .merge(
            truth,
            on=list(EVENT_KEY),
            how="left",
            validate="one_to_one",
        )
        .dropna(
            subset=[
                selected_candidate,
                "forward_excess_return",
                "injected_return_alpha",
            ]
        )
        .rename(columns={selected_candidate: "locked_proxy"})
    )
    if joined.empty or joined["gvkey"].nunique() < 2:
        raise PowerCalibrationError("No valid locked-proxy calibration rows exist.")
    return joined


def calibrate_locked_proxy_power(
    targets_path: Path,
    truth_path: Path,
    candidates_path: Path,
    *,
    selected_candidate: str = "anchored_axis_innovation",
    multipliers: Sequence[float] = DEFAULT_MULTIPLIERS,
    calibration_replicates: int = 100,
    evaluation_replicates: int = 50,
    calibration_seed_start: int = 32000,
    evaluation_seed_start: int = 42000,
    primary_horizon_months: int = 6,
    minimum_year: int = 2002,
    maximum_year: int = 2018,
    migration_decay_months: int = 12,
    base_monthly_alpha: float = 0.004,
    minimum_oracle_detection_rate: float = 0.80,
    minimum_observable_detection_rate: float = 0.80,
    maximum_null_false_positive_rate: float = 0.10,
    minimum_median_oracle_rank_ic: float = 0.04,
    minimum_median_observable_rank_ic: float = 0.01,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Calibrate injection power for the locked P14F proxy on existing paths."""

    if not multipliers or any(float(item) <= 0.0 for item in multipliers):
        raise PowerCalibrationError("All candidate multipliers must be positive.")
    if calibration_replicates < 2 or evaluation_replicates < 2:
        raise PowerCalibrationError("At least two calibration/evaluation draws are required.")
    frame = _load_locked_proxy_primary(
        targets_path,
        truth_path,
        candidates_path,
        selected_candidate=selected_candidate,
        primary_horizon_months=primary_horizon_months,
        minimum_year=minimum_year,
        maximum_year=maximum_year,
    )
    frame = frame.drop(columns=["migration_surprise"]).rename(
        columns={"locked_proxy": "migration_surprise"}
    )
    expected_fraction = float(
        sum(np.exp(-offset / 4.0) for offset in range(primary_horizon_months))
        / sum(np.exp(-offset / 4.0) for offset in range(migration_decay_months))
    )
    calibration, null_rate = _bootstrap_metrics(
        frame,
        multipliers,
        replicates=calibration_replicates,
        seed_start=calibration_seed_start,
        expected_horizon_fraction=expected_fraction,
    )
    calibration["passes_oracle_detection"] = calibration[
        "oracle_detection_rate"
    ].ge(minimum_oracle_detection_rate)
    calibration["passes_observable_detection"] = calibration[
        "observable_detection_rate"
    ].ge(minimum_observable_detection_rate)
    calibration["passes_null_control"] = calibration[
        "null_observable_false_positive_rate"
    ].le(maximum_null_false_positive_rate)
    calibration["passes_oracle_ic"] = calibration["median_oracle_rank_ic"].ge(
        minimum_median_oracle_rank_ic
    )
    calibration["passes_observable_ic"] = calibration[
        "median_observable_rank_ic"
    ].ge(minimum_median_observable_rank_ic)
    gate_columns = [
        "passes_oracle_detection",
        "passes_observable_detection",
        "passes_null_control",
        "passes_oracle_ic",
        "passes_observable_ic",
    ]
    calibration["passes_all_calibration_gates"] = calibration[gate_columns].all(axis=1)
    passing = calibration.loc[calibration["passes_all_calibration_gates"]]
    selected = float(passing["multiplier"].min()) if not passing.empty else None
    evaluation_row: dict[str, Any] | None = None
    evaluation_pass = False
    if selected is not None:
        evaluation, _ = _bootstrap_metrics(
            frame,
            [selected],
            replicates=evaluation_replicates,
            seed_start=evaluation_seed_start,
            expected_horizon_fraction=expected_fraction,
        )
        evaluation_row = evaluation.iloc[0].to_dict()
        evaluation_pass = bool(
            evaluation_row["oracle_detection_rate"]
            >= minimum_oracle_detection_rate
            and evaluation_row["observable_detection_rate"]
            >= minimum_observable_detection_rate
            and evaluation_row["null_observable_false_positive_rate"]
            <= maximum_null_false_positive_rate
            and evaluation_row["median_oracle_rank_ic"]
            >= minimum_median_oracle_rank_ic
            and evaluation_row["median_observable_rank_ic"]
            >= minimum_median_observable_rank_ic
        )

    summary: dict[str, Any] = {
        "schema_version": 1,
        "phase": "P15E",
        "version": P15E_VERSION,
        "status": "GO" if selected is not None and evaluation_pass else "NO_GO",
        "interpretation": (
            "LOCKED_PROXY_POWER_CALIBRATED_FOR_NEW_SYNTHETIC_CANARY"
            if selected is not None and evaluation_pass
            else "NO_MULTIPLIER_PASSED_LOCKED_PROXY_POWER_GATES"
        ),
        "selected_candidate": selected_candidate,
        "rows": int(len(frame)),
        "firms": int(frame["gvkey"].nunique()),
        "analysis_years": [minimum_year, maximum_year],
        "candidate_multipliers": [float(item) for item in multipliers],
        "selected_multiplier": selected,
        "proposed_migration_alpha_monthly": (
            float(base_monthly_alpha * selected)
            if selected is not None and evaluation_pass
            else None
        ),
        "base_migration_alpha_monthly": base_monthly_alpha,
        "expected_six_month_fraction_of_total_injection": expected_fraction,
        "calibration_replicates": calibration_replicates,
        "evaluation_replicates": evaluation_replicates,
        "calibration_seed_start": calibration_seed_start,
        "evaluation_seed_start": evaluation_seed_start,
        "thresholds": {
            "minimum_oracle_detection_rate": minimum_oracle_detection_rate,
            "minimum_observable_detection_rate": minimum_observable_detection_rate,
            "maximum_null_false_positive_rate": maximum_null_false_positive_rate,
            "minimum_median_oracle_rank_ic": minimum_median_oracle_rank_ic,
            "minimum_median_observable_rank_ic": minimum_median_observable_rank_ic,
        },
        "calibration_null_observable_false_positive_rate": null_rate,
        "evaluation": evaluation_row,
        "held_out_evaluation_pass": evaluation_pass,
        "existing_return_paths_only": True,
        "new_synthetic_scenario_generated": False,
        "frozen_p0_p10_outputs_modified": False,
        "real_data_run": False,
        "stop_before_new_synthetic_generation": True,
    }
    return summary, calibration


def validate_locked_proxy_power_outputs(
    targets_path: Path,
    truth_path: Path,
    candidates_path: Path,
    summary_path: Path,
    calibration_path: Path,
    *,
    calibration_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute the deterministic P15E protocol and compare saved outputs."""

    expected_summary, expected_calibration = calibrate_locked_proxy_power(
        targets_path,
        truth_path,
        candidates_path,
        **(calibration_kwargs or {}),
    )
    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    saved_calibration = pd.read_csv(calibration_path)
    errors: list[str] = []
    for key, expected in expected_summary.items():
        if key == "generated_at_utc":
            continue
        if saved_summary.get(key) != expected:
            errors.append(f"Summary mismatch: {key}")
    try:
        pd.testing.assert_frame_equal(
            saved_calibration,
            expected_calibration,
            check_dtype=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as exc:
        errors.append(f"Calibration table mismatch: {exc}")
    return {
        "schema_version": 1,
        "phase": "P15E",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "gate_result": expected_summary["status"],
        "selected_candidate": expected_summary["selected_candidate"],
        "selected_multiplier": expected_summary["selected_multiplier"],
        "held_out_evaluation_pass": expected_summary["held_out_evaluation_pass"],
        "new_synthetic_scenario_generated": False,
        "frozen_p0_p10_outputs_modified": False,
        "real_data_run": False,
    }
