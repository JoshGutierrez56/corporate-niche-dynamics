"""Outcome-blind P13F migration-proxy construction and truth evaluation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


P13F_VERSION = "hypercube-migration-proxy-redesign-v2"
P14F_VERSION = "hypercube-p7-eligibility-proxy-audit-v2"
EVENT_KEY = ("gvkey", "datadate", "fyear")
RELATIVE_AXES = (
    "relative_demand_strength_pricing_power",
    "relative_competitive_defensibility",
    "relative_innovation_intensity",
    "relative_go_to_market_efficiency",
    "relative_unit_economics_profit_quality",
    "relative_scalability_capital_efficiency",
)
ANCHORED_AXES = tuple(item.replace("relative_", "anchored_") for item in RELATIVE_AXES)
NEW_CANDIDATES = (
    "anchored_axis_innovation",
    "blended_axis_innovation",
    "relative_axis_innovation",
)
BENCHMARK = "migration_surprise"


class ProxyRedesignError(ValueError):
    """Raised when P13F inputs or frozen gates cannot be reconciled."""


def atomic_write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Atomically publish a parquet table."""

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


def _fit_expanding_ar(
    frame: pd.DataFrame,
    level_column: str,
    *,
    minimum_train_transitions: int,
    minimum_prior_years: int,
) -> tuple[pd.Series, list[dict[str, Any]]]:
    prior_column = f"previous_{level_column}"
    output = pd.Series(np.nan, index=frame.index, dtype=float)
    diagnostics: list[dict[str, Any]] = []
    transition = (
        frame[level_column].notna()
        & frame[prior_column].notna()
        & frame["consecutive_fiscal_year"].astype(bool)
    )
    for year in sorted(frame["fyear"].unique()):
        train = frame.loc[transition & frame["fyear"].lt(year)]
        predict = frame.loc[transition & frame["fyear"].eq(year)]
        row: dict[str, Any] = {
            "level": level_column,
            "prediction_year": int(year),
            "train_rows": int(len(train)),
            "train_years": int(train["fyear"].nunique()),
            "prediction_rows": int(len(predict)),
            "status": "insufficient_history",
            "intercept": None,
            "persistence": None,
            "train_max_year": None,
        }
        if (
            predict.empty
            or len(train) < minimum_train_transitions
            or train["fyear"].nunique() < minimum_prior_years
        ):
            diagnostics.append(row)
            continue
        previous = train[prior_column].to_numpy(float)
        current = train[level_column].to_numpy(float)
        variance = float(np.square(previous - previous.mean()).sum())
        if variance <= 0.0:
            raise ProxyRedesignError(f"{level_column} has zero prior-level variance.")
        persistence = float(
            np.sum((previous - previous.mean()) * (current - current.mean()))
            / variance
        )
        persistence = float(np.clip(persistence, 0.0, 1.0))
        intercept = float(current.mean() - persistence * previous.mean())
        expected = intercept + persistence * predict[prior_column].to_numpy(float)
        output.loc[predict.index] = predict[level_column].to_numpy(float) - expected
        row.update(
            {
                "status": "fitted",
                "intercept": intercept,
                "persistence": persistence,
                "train_max_year": int(train["fyear"].max()),
            }
        )
        if row["train_max_year"] >= int(year):
            raise ProxyRedesignError("An expanding AR fit includes its prediction year.")
        diagnostics.append(row)
    return output, diagnostics


def build_proxy_candidates(
    p5_path: Path,
    *,
    primary_horizon_years: int = 5,
    minimum_axis_count: int = 4,
    minimum_train_transitions: int = 1000,
    minimum_prior_years: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Construct frozen P13F candidates without truth, returns, or injected alpha."""

    required = [
        *EVENT_KEY,
        "feature_date",
        "horizon_years",
        BENCHMARK,
        *RELATIVE_AXES,
        *ANCHORED_AXES,
    ]
    frame = pd.read_parquet(p5_path, columns=required)
    frame = frame.loc[frame["horizon_years"].eq(primary_horizon_years)].copy()
    frame["datadate"] = pd.to_datetime(frame["datadate"])
    frame["feature_date"] = pd.to_datetime(frame["feature_date"])
    frame = frame.sort_values(["gvkey", "datadate", "fyear"]).reset_index(drop=True)
    if frame.empty or frame.duplicated(list(EVENT_KEY)).any():
        raise ProxyRedesignError("P5 primary surface is empty or has duplicate events.")

    frame["relative_axis_level"] = frame[list(RELATIVE_AXES)].mean(axis=1).where(
        frame[list(RELATIVE_AXES)].notna().sum(axis=1).ge(minimum_axis_count)
    )
    frame["anchored_axis_level"] = frame[list(ANCHORED_AXES)].mean(axis=1).where(
        frame[list(ANCHORED_AXES)].notna().sum(axis=1).ge(minimum_axis_count)
    )
    frame["blended_axis_level"] = frame[
        ["relative_axis_level", "anchored_axis_level"]
    ].mean(axis=1)
    grouped = frame.groupby("gvkey", sort=False)
    frame["previous_fyear"] = grouped["fyear"].shift(1)
    frame["consecutive_fiscal_year"] = (
        frame["fyear"] - frame["previous_fyear"]
    ).eq(1)

    diagnostics: list[dict[str, Any]] = []
    for prefix in ("relative", "anchored", "blended"):
        level = f"{prefix}_axis_level"
        frame[f"previous_{level}"] = grouped[level].shift(1)
        innovation, rows = _fit_expanding_ar(
            frame,
            level,
            minimum_train_transitions=minimum_train_transitions,
            minimum_prior_years=minimum_prior_years,
        )
        frame[f"{prefix}_axis_innovation"] = innovation
        diagnostics.extend(rows)

    output_columns = [
        *EVENT_KEY,
        "feature_date",
        BENCHMARK,
        "relative_axis_level",
        "anchored_axis_level",
        "blended_axis_level",
        *NEW_CANDIDATES,
    ]
    output = frame[output_columns].copy()
    coefficient_table = pd.DataFrame(diagnostics).sort_values(
        ["level", "prediction_year"]
    ).reset_index(drop=True)
    receipt = {
        "schema_version": 1,
        "phase": "P13F",
        "version": P13F_VERSION,
        "status": "BUILT",
        "rows": int(len(output)),
        "firms": int(output["gvkey"].nunique()),
        "minimum_year": int(output["fyear"].min()),
        "maximum_year": int(output["fyear"].max()),
        "primary_horizon_years": int(primary_horizon_years),
        "minimum_axis_count": int(minimum_axis_count),
        "minimum_train_transitions": int(minimum_train_transitions),
        "minimum_prior_years": int(minimum_prior_years),
        "candidate_columns": list(NEW_CANDIDATES),
        "benchmark_column": BENCHMARK,
        "synthetic_truth_read": False,
        "return_outcomes_read": False,
        "injected_alpha_read": False,
        "real_data_run": False,
        "new_synthetic_scenario_generated": False,
        "frozen_p0_p10_outputs_modified": False,
    }
    return output, coefficient_table, receipt


def _rank_correlation(frame: pd.DataFrame, left: str, right: str) -> float:
    complete = frame[[left, right]].dropna()
    if len(complete) < 3:
        return float("nan")
    return float(spearmanr(complete[left], complete[right]).statistic)


def evaluate_proxy_candidates(
    candidates_path: Path,
    truth_path: Path,
    *,
    calibration_start_year: int = 2002,
    calibration_end_year: int = 2012,
    holdout_start_year: int = 2013,
    holdout_end_year: int = 2018,
    minimum_holdout_spearman: float = 0.35,
    minimum_block_spearman: float = 0.20,
    minimum_coverage_ratio: float = 0.95,
    minimum_improvement: float = 0.10,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Open synthetic migration truth only after candidates have been published."""

    candidates = pd.read_parquet(candidates_path)
    truth = pd.read_parquet(
        truth_path,
        columns=[*EVENT_KEY, "migration_surprise"],
    ).rename(columns={"migration_surprise": "truth_migration_surprise"})
    joined = candidates.merge(
        truth,
        on=list(EVENT_KEY),
        how="left",
        validate="one_to_one",
    )
    if joined["truth_migration_surprise"].isna().any():
        raise ProxyRedesignError("Synthetic truth does not match every P13F event.")
    calibration = joined.loc[
        joined["fyear"].between(calibration_start_year, calibration_end_year)
    ]
    holdout = joined.loc[joined["fyear"].between(holdout_start_year, holdout_end_year)]
    if calibration.empty or holdout.empty:
        raise ProxyRedesignError("Frozen calibration or holdout years are empty.")

    benchmark_holdout_rows = int(holdout[BENCHMARK].notna().sum())
    benchmark_holdout_ic = _rank_correlation(
        holdout, BENCHMARK, "truth_migration_surprise"
    )
    metrics: list[dict[str, Any]] = []
    for candidate in (*NEW_CANDIDATES, BENCHMARK):
        calibration_ic = _rank_correlation(
            calibration, candidate, "truth_migration_surprise"
        )
        holdout_ic = _rank_correlation(holdout, candidate, "truth_migration_surprise")
        block_one = holdout.loc[holdout["fyear"].between(2013, 2015)]
        block_two = holdout.loc[holdout["fyear"].between(2016, 2018)]
        holdout_rows = int(holdout[candidate].notna().sum())
        coverage_ratio = (
            float(holdout_rows / benchmark_holdout_rows)
            if benchmark_holdout_rows
            else float("nan")
        )
        metrics.append(
            {
                "candidate": candidate,
                "calibration_spearman": calibration_ic,
                "held_out_spearman": holdout_ic,
                "held_out_2013_2015_spearman": _rank_correlation(
                    block_one, candidate, "truth_migration_surprise"
                ),
                "held_out_2016_2018_spearman": _rank_correlation(
                    block_two, candidate, "truth_migration_surprise"
                ),
                "calibration_rows": int(calibration[candidate].notna().sum()),
                "held_out_rows": holdout_rows,
                "held_out_coverage_vs_benchmark": coverage_ratio,
                "held_out_improvement_vs_benchmark": float(
                    holdout_ic - benchmark_holdout_ic
                ),
                "is_benchmark": candidate == BENCHMARK,
            }
        )
    table = pd.DataFrame(metrics).sort_values("candidate").reset_index(drop=True)
    selectable = table.loc[~table["is_benchmark"]].sort_values(
        ["calibration_spearman", "candidate"],
        ascending=[False, True],
    )
    selected = str(selectable.iloc[0]["candidate"])
    selected_row = selectable.iloc[0]
    gates = {
        "held_out_spearman": bool(
            selected_row["held_out_spearman"] >= minimum_holdout_spearman
        ),
        "block_stability": bool(
            selected_row["held_out_2013_2015_spearman"]
            >= minimum_block_spearman
            and selected_row["held_out_2016_2018_spearman"]
            >= minimum_block_spearman
        ),
        "coverage": bool(
            selected_row["held_out_coverage_vs_benchmark"] >= minimum_coverage_ratio
        ),
        "improvement": bool(
            selected_row["held_out_improvement_vs_benchmark"]
            >= minimum_improvement
        ),
    }
    passed = all(gates.values())
    summary = {
        "schema_version": 1,
        "phase": "P13F",
        "version": P13F_VERSION,
        "status": "GO" if passed else "NO_GO",
        "interpretation": (
            "PROXY_QUALIFIED_FOR_HELD_OUT_POWER_CALIBRATION"
            if passed
            else "NO_PROXY_PASSED_ALL_HELD_OUT_GATES"
        ),
        "selected_candidate": selected,
        "selection_rule": "highest calibration Spearman; alphabetical tie break",
        "calibration_years": [calibration_start_year, calibration_end_year],
        "held_out_years": [holdout_start_year, holdout_end_year],
        "held_out_blocks": [[2013, 2015], [2016, 2018]],
        "thresholds": {
            "minimum_holdout_spearman": minimum_holdout_spearman,
            "minimum_block_spearman": minimum_block_spearman,
            "minimum_coverage_ratio": minimum_coverage_ratio,
            "minimum_improvement": minimum_improvement,
        },
        "selected_metrics": {
            key: (
                bool(value)
                if isinstance(value, (bool, np.bool_))
                else int(value)
                if isinstance(value, (int, np.integer))
                else float(value)
                if isinstance(value, (float, np.floating))
                else value
            )
            for key, value in selected_row.to_dict().items()
        },
        "benchmark_held_out_spearman": benchmark_holdout_ic,
        "gates": gates,
        "synthetic_truth_read_during_candidate_construction": False,
        "return_outcomes_read": False,
        "real_data_run": False,
        "new_synthetic_scenario_generated": False,
        "frozen_p0_p10_outputs_modified": False,
        "stop_before_new_synthetic_generation": True,
    }
    return summary, table


def validate_proxy_redesign_outputs(
    p5_path: Path,
    truth_path: Path,
    candidates_path: Path,
    coefficients_path: Path,
    build_receipt_path: Path,
    evaluation_path: Path,
    metrics_path: Path,
    *,
    build_kwargs: dict[str, Any] | None = None,
    evaluation_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reconstruct both P13F stages and compare every saved result."""

    expected_candidates, expected_coefficients, expected_receipt = (
        build_proxy_candidates(p5_path, **(build_kwargs or {}))
    )
    saved_candidates = pd.read_parquet(candidates_path)
    saved_coefficients = pd.read_csv(coefficients_path)
    saved_receipt = json.loads(build_receipt_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    try:
        pd.testing.assert_frame_equal(
            saved_candidates,
            expected_candidates,
            check_dtype=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as exc:
        errors.append(f"Candidate table mismatch: {exc}")
    try:
        pd.testing.assert_frame_equal(
            saved_coefficients,
            expected_coefficients,
            check_dtype=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as exc:
        errors.append(f"Coefficient table mismatch: {exc}")
    for key, expected in expected_receipt.items():
        if saved_receipt.get(key) != expected:
            errors.append(f"Build receipt mismatch: {key}")

    expected_summary, expected_metrics = evaluate_proxy_candidates(
        candidates_path,
        truth_path,
        **(evaluation_kwargs or {}),
    )
    saved_summary = json.loads(evaluation_path.read_text(encoding="utf-8"))
    saved_metrics = pd.read_csv(metrics_path)
    for key, expected in expected_summary.items():
        if key == "generated_at_utc":
            continue
        if saved_summary.get(key) != expected:
            errors.append(f"Evaluation summary mismatch: {key}")
    try:
        pd.testing.assert_frame_equal(
            saved_metrics,
            expected_metrics,
            check_dtype=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as exc:
        errors.append(f"Evaluation table mismatch: {exc}")
    return {
        "schema_version": 1,
        "phase": "P13F",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "gate_result": expected_summary["status"],
        "selected_candidate": expected_summary["selected_candidate"],
        "new_synthetic_scenario_generated": False,
        "frozen_p0_p10_outputs_modified": False,
        "real_data_run": False,
    }


def audit_p7_eligibility(
    candidates_path: Path,
    targets_path: Path,
    truth_path: Path,
    *,
    selected_candidate: str = "anchored_axis_innovation",
    primary_horizon_months: int = 6,
    calibration_start_year: int = 2002,
    calibration_end_year: int = 2012,
    holdout_start_year: int = 2013,
    holdout_end_year: int = 2018,
    minimum_overall_spearman: float = 0.35,
    minimum_improvement: float = 0.10,
    minimum_period_spearman: float = 0.30,
    minimum_block_spearman: float = 0.20,
    minimum_coverage_ratio: float = 0.95,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Audit a locked proxy on P7 eligibility keys without reading return values."""

    candidates = pd.read_parquet(
        candidates_path,
        columns=[*EVENT_KEY, selected_candidate, BENCHMARK],
    )
    targets = pd.read_parquet(
        targets_path,
        columns=[*EVENT_KEY, "horizon_months", "target_valid"],
    )
    eligible = targets.loc[
        targets["horizon_months"].eq(primary_horizon_months)
        & targets["target_valid"].astype(bool)
    ].copy()
    if eligible.empty or eligible.duplicated(list(EVENT_KEY)).any():
        raise ProxyRedesignError("P7 eligibility keys are empty or duplicated.")
    truth = pd.read_parquet(
        truth_path,
        columns=[*EVENT_KEY, "migration_surprise"],
    ).rename(columns={"migration_surprise": "truth_migration_surprise"})
    joined = (
        eligible[list(EVENT_KEY)]
        .merge(candidates, on=list(EVENT_KEY), how="left", validate="one_to_one")
        .merge(truth, on=list(EVENT_KEY), how="left", validate="one_to_one")
    )
    if joined["truth_migration_surprise"].isna().any():
        raise ProxyRedesignError("P7 eligibility keys do not reconcile to P5/truth.")
    joined = joined.loc[joined[BENCHMARK].notna()].copy()
    if joined.empty:
        raise ProxyRedesignError("No P11E-comparable benchmark rows remain.")
    joined = joined.loc[
        joined["fyear"].between(calibration_start_year, holdout_end_year)
    ].copy()
    if joined.empty:
        raise ProxyRedesignError("No rows remain in the frozen analysis years.")

    calibration = joined.loc[
        joined["fyear"].between(calibration_start_year, calibration_end_year)
    ]
    holdout = joined.loc[joined["fyear"].between(holdout_start_year, holdout_end_year)]
    block_one = holdout.loc[holdout["fyear"].between(2013, 2015)]
    block_two = holdout.loc[holdout["fyear"].between(2016, 2018)]
    benchmark_rows = int(joined[BENCHMARK].notna().sum())
    benchmark_overall = _rank_correlation(
        joined, BENCHMARK, "truth_migration_surprise"
    )
    rows: list[dict[str, Any]] = []
    for name in (selected_candidate, BENCHMARK):
        nonmissing = int(joined[name].notna().sum())
        rows.append(
            {
                "candidate": name,
                "overall_spearman": _rank_correlation(
                    joined, name, "truth_migration_surprise"
                ),
                "calibration_spearman": _rank_correlation(
                    calibration, name, "truth_migration_surprise"
                ),
                "held_out_spearman": _rank_correlation(
                    holdout, name, "truth_migration_surprise"
                ),
                "held_out_2013_2015_spearman": _rank_correlation(
                    block_one, name, "truth_migration_surprise"
                ),
                "held_out_2016_2018_spearman": _rank_correlation(
                    block_two, name, "truth_migration_surprise"
                ),
                "eligible_rows": nonmissing,
                "coverage_vs_benchmark": float(nonmissing / benchmark_rows),
                "overall_improvement_vs_benchmark": float(
                    _rank_correlation(joined, name, "truth_migration_surprise")
                    - benchmark_overall
                ),
                "is_benchmark": name == BENCHMARK,
            }
        )
    table = pd.DataFrame(rows)
    selected = table.loc[table["candidate"].eq(selected_candidate)].iloc[0]
    gates = {
        "overall_spearman": bool(
            selected["overall_spearman"] >= minimum_overall_spearman
        ),
        "improvement": bool(
            selected["overall_improvement_vs_benchmark"] >= minimum_improvement
        ),
        "period_stability": bool(
            selected["calibration_spearman"] >= minimum_period_spearman
            and selected["held_out_spearman"] >= minimum_period_spearman
        ),
        "block_stability": bool(
            selected["held_out_2013_2015_spearman"] >= minimum_block_spearman
            and selected["held_out_2016_2018_spearman"]
            >= minimum_block_spearman
        ),
        "coverage": bool(
            selected["coverage_vs_benchmark"] >= minimum_coverage_ratio
        ),
    }
    passed = all(gates.values())
    summary = {
        "schema_version": 1,
        "phase": "P14F",
        "version": P14F_VERSION,
        "status": "GO" if passed else "NO_GO",
        "interpretation": (
            "LOCKED_PROXY_QUALIFIED_ON_P7_ELIGIBILITY_SAMPLE"
            if passed
            else "LOCKED_PROXY_FAILED_P7_ELIGIBILITY_GATES"
        ),
        "selected_candidate": selected_candidate,
        "primary_horizon_months": primary_horizon_months,
        "eligible_rows": int(len(joined)),
        "firms": int(joined["gvkey"].nunique()),
        "calibration_years": [calibration_start_year, calibration_end_year],
        "held_out_years": [holdout_start_year, holdout_end_year],
        "thresholds": {
            "minimum_overall_spearman": minimum_overall_spearman,
            "minimum_improvement": minimum_improvement,
            "minimum_period_spearman": minimum_period_spearman,
            "minimum_block_spearman": minimum_block_spearman,
            "minimum_coverage_ratio": minimum_coverage_ratio,
        },
        "selected_metrics": {
            key: (
                bool(value)
                if isinstance(value, (bool, np.bool_))
                else int(value)
                if isinstance(value, (int, np.integer))
                else float(value)
                if isinstance(value, (float, np.floating))
                else value
            )
            for key, value in selected.to_dict().items()
        },
        "benchmark_overall_spearman": benchmark_overall,
        "gates": gates,
        "target_columns_read": [
            *EVENT_KEY,
            "horizon_months",
            "target_valid",
        ],
        "return_values_read": False,
        "injected_alpha_read": False,
        "real_data_run": False,
        "new_synthetic_scenario_generated": False,
        "frozen_p0_p10_outputs_modified": False,
        "stop_before_new_synthetic_generation": True,
    }
    return summary, table


def validate_p7_eligibility_audit(
    candidates_path: Path,
    targets_path: Path,
    truth_path: Path,
    summary_path: Path,
    table_path: Path,
    *,
    audit_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute the target-key-only P14F audit."""

    expected_summary, expected_table = audit_p7_eligibility(
        candidates_path,
        targets_path,
        truth_path,
        **(audit_kwargs or {}),
    )
    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    saved_table = pd.read_csv(table_path)
    errors: list[str] = []
    for key, expected in expected_summary.items():
        if key == "generated_at_utc":
            continue
        if saved_summary.get(key) != expected:
            errors.append(f"Summary mismatch: {key}")
    try:
        pd.testing.assert_frame_equal(
            saved_table,
            expected_table,
            check_dtype=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as exc:
        errors.append(f"Audit table mismatch: {exc}")
    return {
        "schema_version": 1,
        "phase": "P14F",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "gate_result": expected_summary["status"],
        "selected_candidate": expected_summary["selected_candidate"],
        "return_values_read": False,
        "new_synthetic_scenario_generated": False,
        "frozen_p0_p10_outputs_modified": False,
        "real_data_run": False,
    }
