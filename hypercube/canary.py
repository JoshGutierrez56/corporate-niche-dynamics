"""P16E isolated-canary construction and realized-return gates."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from hypercube.power_calibration import _clustered_regression
from hypercube.proxy_redesign import (
    BENCHMARK,
    EVENT_KEY,
    build_proxy_candidates,
)


P16E_VERSION = "hypercube-isolated-six-x-canary-v1"
LOCKED_PROXY = "anchored_axis_innovation"


class CanaryError(ValueError):
    """Raised when isolated P16E inputs or outputs violate the frozen contract."""


def _spearman(frame: pd.DataFrame, left: str, right: str) -> float:
    complete = frame[[left, right]].dropna()
    if len(complete) < 3:
        raise CanaryError(f"Insufficient complete rows for {left}/{right}.")
    return float(spearmanr(complete[left], complete[right]).statistic)


def _json_safe_row(row: pd.Series) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in row.to_dict().items():
        if pd.isna(value):
            output[key] = None
        elif isinstance(value, (int, np.integer)):
            output[key] = int(value)
        elif isinstance(value, (float, np.floating)):
            output[key] = float(value)
        else:
            output[key] = value
    return output


def evaluate_realized_canary(
    migration_targets_path: Path,
    migration_truth_path: Path,
    migration_candidates_path: Path,
    null_targets_path: Path,
    null_candidates_path: Path,
    *,
    selected_candidate: str = LOCKED_PROXY,
    primary_horizon_months: int = 6,
    minimum_year: int = 2002,
    maximum_year: int = 2018,
    minimum_truth_spearman: float = 0.35,
    minimum_coverage_ratio: float = 0.95,
    minimum_oracle_slope: float = 0.40,
    maximum_oracle_slope: float = 1.30,
    maximum_detection_p_value: float = 0.05,
    minimum_proxy_return_spearman: float = 0.01,
    maximum_null_abs_spearman: float = 0.03,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Evaluate actual regenerated migration and null return paths."""

    target_columns = [
        *EVENT_KEY,
        "horizon_months",
        "target_valid",
        BENCHMARK,
        "forward_excess_return",
    ]
    migration_targets = pd.read_parquet(
        migration_targets_path, columns=target_columns
    )
    migration_population = migration_targets.loc[
        migration_targets["horizon_months"].eq(primary_horizon_months)
        & migration_targets["target_valid"].astype(bool)
        & migration_targets[BENCHMARK].notna()
        & migration_targets["fyear"].between(minimum_year, maximum_year)
    ].copy()
    migration_candidates = pd.read_parquet(
        migration_candidates_path,
        columns=[*EVENT_KEY, selected_candidate],
    )
    migration_truth = pd.read_parquet(
        migration_truth_path,
        columns=[*EVENT_KEY, "migration_surprise", "injected_return_alpha"],
    ).rename(columns={"migration_surprise": "truth_migration_surprise"})
    migration = (
        migration_population.merge(
            migration_candidates,
            on=list(EVENT_KEY),
            how="left",
            validate="one_to_one",
        )
        .merge(
            migration_truth,
            on=list(EVENT_KEY),
            how="left",
            validate="one_to_one",
        )
        .dropna(
            subset=[
                selected_candidate,
                "truth_migration_surprise",
                "injected_return_alpha",
                "forward_excess_return",
            ]
        )
    )
    if migration.empty:
        raise CanaryError("No complete migration-alpha canary rows exist.")
    migration_coverage = float(len(migration) / len(migration_population))
    groups = migration["gvkey"].astype(str).to_numpy()
    migration_return = migration["forward_excess_return"].to_numpy(float)
    oracle_fit = _clustered_regression(
        migration_return,
        migration["injected_return_alpha"].to_numpy(float),
        groups,
    )
    proxy_fit = _clustered_regression(
        migration_return,
        migration[selected_candidate].to_numpy(float),
        groups,
    )
    truth_spearman = _spearman(
        migration, selected_candidate, "truth_migration_surprise"
    )
    migration_return_spearman = _spearman(
        migration, selected_candidate, "forward_excess_return"
    )

    null_targets = pd.read_parquet(null_targets_path, columns=target_columns)
    null_population = null_targets.loc[
        null_targets["horizon_months"].eq(primary_horizon_months)
        & null_targets["target_valid"].astype(bool)
        & null_targets[BENCHMARK].notna()
        & null_targets["fyear"].between(minimum_year, maximum_year)
    ].copy()
    null_candidates = pd.read_parquet(
        null_candidates_path,
        columns=[*EVENT_KEY, selected_candidate],
    )
    null_frame = null_population.merge(
        null_candidates,
        on=list(EVENT_KEY),
        how="left",
        validate="one_to_one",
    ).dropna(subset=[selected_candidate, "forward_excess_return"])
    if null_frame.empty:
        raise CanaryError("No complete null-alpha canary rows exist.")
    null_fit = _clustered_regression(
        null_frame["forward_excess_return"].to_numpy(float),
        null_frame[selected_candidate].to_numpy(float),
        null_frame["gvkey"].astype(str).to_numpy(),
    )
    null_spearman = _spearman(
        null_frame, selected_candidate, "forward_excess_return"
    )

    gates = {
        "candidate_truth_alignment": truth_spearman >= minimum_truth_spearman,
        "candidate_coverage": migration_coverage >= minimum_coverage_ratio,
        "oracle_slope_range": (
            oracle_fit["slope"] >= minimum_oracle_slope
            and oracle_fit["slope"] <= maximum_oracle_slope
        ),
        "oracle_detected": (
            oracle_fit["slope"] > 0.0
            and oracle_fit["p_value"] < maximum_detection_p_value
        ),
        "proxy_return_rank_ic": (
            migration_return_spearman >= minimum_proxy_return_spearman
        ),
        "proxy_detected": (
            proxy_fit["slope"] > 0.0
            and proxy_fit["p_value"] < maximum_detection_p_value
        ),
        "null_rank_ic": abs(null_spearman) <= maximum_null_abs_spearman,
        "null_not_detected": not (
            null_fit["slope"] > 0.0
            and null_fit["p_value"] < maximum_detection_p_value
        ),
    }
    passed = all(gates.values())
    rows = pd.DataFrame(
        [
            {
                "scenario": "migration_alpha",
                "rows": int(len(migration)),
                "firms": int(migration["gvkey"].nunique()),
                "coverage_vs_benchmark": migration_coverage,
                "candidate_truth_spearman": truth_spearman,
                "candidate_return_spearman": migration_return_spearman,
                "candidate_return_slope": proxy_fit["slope"],
                "candidate_return_clustered_se": proxy_fit["standard_error"],
                "candidate_return_t_stat": proxy_fit["t_stat"],
                "candidate_return_p_value": proxy_fit["p_value"],
                "oracle_return_slope": oracle_fit["slope"],
                "oracle_return_clustered_se": oracle_fit["standard_error"],
                "oracle_return_t_stat": oracle_fit["t_stat"],
                "oracle_return_p_value": oracle_fit["p_value"],
            },
            {
                "scenario": "null_alpha",
                "rows": int(len(null_frame)),
                "firms": int(null_frame["gvkey"].nunique()),
                "coverage_vs_benchmark": float(
                    len(null_frame) / len(null_population)
                ),
                "candidate_truth_spearman": np.nan,
                "candidate_return_spearman": null_spearman,
                "candidate_return_slope": null_fit["slope"],
                "candidate_return_clustered_se": null_fit["standard_error"],
                "candidate_return_t_stat": null_fit["t_stat"],
                "candidate_return_p_value": null_fit["p_value"],
                "oracle_return_slope": np.nan,
                "oracle_return_clustered_se": np.nan,
                "oracle_return_t_stat": np.nan,
                "oracle_return_p_value": np.nan,
            },
        ]
    )
    summary = {
        "schema_version": 1,
        "phase": "P16E",
        "version": P16E_VERSION,
        "status": "PASS" if passed else "FAIL",
        "interpretation": (
            "ISOLATED_REGENERATED_CANARY_RECOVERED_SIGNAL_AND_NULL"
            if passed
            else "ISOLATED_REGENERATED_CANARY_FAILED_ONE_OR_MORE_GATES"
        ),
        "selected_candidate": selected_candidate,
        "migration_alpha_monthly": 0.024,
        "injection_multiplier_vs_original": 6.0,
        "analysis_years": [minimum_year, maximum_year],
        "primary_horizon_months": primary_horizon_months,
        "gates": gates,
        "thresholds": {
            "minimum_truth_spearman": minimum_truth_spearman,
            "minimum_coverage_ratio": minimum_coverage_ratio,
            "minimum_oracle_slope": minimum_oracle_slope,
            "maximum_oracle_slope": maximum_oracle_slope,
            "maximum_detection_p_value": maximum_detection_p_value,
            "minimum_proxy_return_spearman": minimum_proxy_return_spearman,
            "maximum_null_abs_spearman": maximum_null_abs_spearman,
        },
        "migration_metrics": _json_safe_row(rows.iloc[0]),
        "null_metrics": _json_safe_row(rows.iloc[1]),
        "isolated_canary": True,
        "p8_p10_run": False,
        "real_data_run": False,
        "frozen_p0_p10_outputs_modified": False,
    }
    return summary, rows


def validate_realized_canary_outputs(
    migration_targets_path: Path,
    migration_truth_path: Path,
    migration_candidates_path: Path,
    null_targets_path: Path,
    null_candidates_path: Path,
    summary_path: Path,
    metrics_path: Path,
    *,
    evaluation_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute P16E realized-return gates and compare saved outputs."""

    expected_summary, expected_metrics = evaluate_realized_canary(
        migration_targets_path,
        migration_truth_path,
        migration_candidates_path,
        null_targets_path,
        null_candidates_path,
        **(evaluation_kwargs or {}),
    )
    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    saved_metrics = pd.read_csv(metrics_path)
    errors: list[str] = []
    for key, expected in expected_summary.items():
        if key == "generated_at_utc":
            continue
        if saved_summary.get(key) != expected:
            errors.append(f"Summary mismatch: {key}")
    try:
        pd.testing.assert_frame_equal(
            saved_metrics,
            expected_metrics,
            check_dtype=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as exc:
        errors.append(f"Metrics mismatch: {exc}")
    return {
        "schema_version": 1,
        "phase": "P16E",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "gate_result": expected_summary["status"],
        "selected_candidate": expected_summary["selected_candidate"],
        "isolated_canary": True,
        "frozen_p0_p10_outputs_modified": False,
        "real_data_run": False,
    }
