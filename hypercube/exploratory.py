"""Post-closeout exploratory diagnostics that never alter frozen P0-P10 outputs."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm


P11E_VERSION = "hypercube-alpha-power-diagnostic-v1"
EVENT_KEY = ("gvkey", "datadate", "fyear")


class AlphaPowerError(ValueError):
    """Raised when the saved P7/truth contract cannot support the diagnostic."""


def _correlations(left: pd.Series, right: pd.Series) -> dict[str, float]:
    frame = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(frame) < 3:
        raise AlphaPowerError("At least three complete observations are required.")
    return {
        "spearman": float(spearmanr(frame["left"], frame["right"]).statistic),
        "pearson": float(pearsonr(frame["left"], frame["right"]).statistic),
    }


def _linear_r_squared(left: np.ndarray, right: np.ndarray) -> float:
    design = np.column_stack([np.ones(len(left)), left])
    fitted = design @ np.linalg.lstsq(design, right, rcond=None)[0]
    denominator = float(np.square(right - right.mean()).sum())
    if denominator <= 0.0:
        raise AlphaPowerError("The diagnostic target has zero variance.")
    return 1.0 - float(np.square(right - fitted).sum()) / denominator


def diagnose_alpha_recovery_power(
    targets_path: Path,
    truth_path: Path,
    *,
    primary_horizon_months: int = 6,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Measure whether the frozen injected-alpha oracle was actually detectable."""

    targets = pd.read_parquet(
        targets_path,
        columns=[
            *EVENT_KEY,
            "horizon_months",
            "target_valid",
            "migration_surprise",
            "forward_excess_return",
            "fold",
        ],
    )
    primary = targets.loc[
        targets["horizon_months"].eq(primary_horizon_months)
        & targets["target_valid"].astype(bool)
        & targets["migration_surprise"].notna()
    ].copy()
    if primary.empty:
        raise AlphaPowerError("No valid primary P7 migration-surprise targets exist.")
    truth = pd.read_parquet(
        truth_path,
        columns=[
            *EVENT_KEY,
            "migration_surprise",
            "injected_return_alpha",
        ],
    )
    joined = primary.merge(
        truth,
        on=list(EVENT_KEY),
        how="left",
        validate="one_to_one",
        suffixes=("_observable", "_truth"),
    )
    required = [
        "migration_surprise_truth",
        "injected_return_alpha",
        "forward_excess_return",
    ]
    if joined[required].isna().any().any():
        raise AlphaPowerError("Synthetic truth does not fully match the P7 event keys.")

    observable = joined["migration_surprise_observable"].to_numpy(float)
    truth_signal = joined["migration_surprise_truth"].to_numpy(float)
    injected = joined["injected_return_alpha"].to_numpy(float)
    returns = joined["forward_excess_return"].to_numpy(float)
    design = sm.add_constant(injected, has_constant="add")
    ols = sm.OLS(returns, design).fit()
    clustered = ols.get_robustcov_results(
        cov_type="cluster",
        groups=joined["gvkey"],
    )

    observable_truth = _correlations(
        joined["migration_surprise_observable"],
        joined["migration_surprise_truth"],
    )
    oracle_return = _correlations(
        joined["injected_return_alpha"],
        joined["forward_excess_return"],
    )
    observable_return = _correlations(
        joined["migration_surprise_observable"],
        joined["forward_excess_return"],
    )
    clustered_p_value = float(clustered.pvalues[1])
    summary: dict[str, Any] = {
        "schema_version": 1,
        "version": P11E_VERSION,
        "phase": "P11E",
        "status": "DESCRIPTIVE_EXPLORATORY",
        "interpretation": (
            "ORACLE_DETECTABLE"
            if clustered_p_value < 0.05
            else "ORACLE_NOT_DETECTABLE"
        ),
        "confirmatory_gate": False,
        "frozen_p0_p10_outputs_modified": False,
        "real_data_run": False,
        "primary_horizon_months": int(primary_horizon_months),
        "rows": int(len(joined)),
        "firms": int(joined["gvkey"].nunique()),
        "observable_truth_signal_spearman": observable_truth["spearman"],
        "observable_truth_signal_pearson": observable_truth["pearson"],
        "observable_to_injected_alpha_linear_r_squared": _linear_r_squared(
            observable, injected
        ),
        "oracle_return_spearman": oracle_return["spearman"],
        "oracle_return_pearson": oracle_return["pearson"],
        "oracle_ols_slope": float(ols.params[1]),
        "oracle_ols_standard_error": float(ols.bse[1]),
        "oracle_ols_t_stat": float(ols.tvalues[1]),
        "oracle_ols_p_value": float(ols.pvalues[1]),
        "oracle_ols_r_squared": float(ols.rsquared),
        "oracle_clustered_standard_error": float(clustered.bse[1]),
        "oracle_clustered_t_stat": float(clustered.tvalues[1]),
        "oracle_clustered_p_value": clustered_p_value,
        "oracle_clustered_ci95_lower": float(clustered.conf_int()[1][0]),
        "oracle_clustered_ci95_upper": float(clustered.conf_int()[1][1]),
        "oracle_detectable_at_5pct": bool(clustered_p_value < 0.05),
        "observable_return_spearman": observable_return["spearman"],
        "observable_return_pearson": observable_return["pearson"],
        "expected_proxy_ic_from_correlation_product": float(
            observable_truth["spearman"] * oracle_return["spearman"]
        ),
        "observable_standard_deviation": float(np.std(observable, ddof=1)),
        "truth_signal_standard_deviation": float(np.std(truth_signal, ddof=1)),
        "injected_alpha_standard_deviation": float(np.std(injected, ddof=1)),
        "forward_return_standard_deviation": float(np.std(returns, ddof=1)),
    }

    fold_rows: list[dict[str, Any]] = []
    for fold, frame in joined.groupby("fold", observed=True):
        fold_injected = frame["injected_return_alpha"].to_numpy(float)
        fold_returns = frame["forward_excess_return"].to_numpy(float)
        fitted = sm.OLS(
            fold_returns,
            sm.add_constant(fold_injected, has_constant="add"),
        ).fit()
        fold_rows.append(
            {
                "fold": fold,
                "rows": int(len(frame)),
                "firms": int(frame["gvkey"].nunique()),
                "oracle_slope": float(fitted.params[1]),
                "oracle_standard_error": float(fitted.bse[1]),
                "oracle_t_stat": float(fitted.tvalues[1]),
                "oracle_p_value": float(fitted.pvalues[1]),
                "oracle_r_squared": float(fitted.rsquared),
            }
        )
    return summary, pd.DataFrame(fold_rows).sort_values("fold").reset_index(drop=True)


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Publish a CSV without exposing a partial diagnostic."""

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


def validate_alpha_power_outputs(
    targets_path: Path,
    truth_path: Path,
    summary_path: Path,
    folds_path: Path,
) -> dict[str, Any]:
    """Independently recompute and compare the saved P11E diagnostic."""

    import json

    errors: list[str] = []
    expected_summary, expected_folds = diagnose_alpha_recovery_power(
        targets_path,
        truth_path,
    )
    saved_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    saved_folds = pd.read_csv(folds_path)
    ignored = {"generated_at_utc"}
    for key, expected in expected_summary.items():
        if key in ignored:
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
            saved_folds,
            expected_folds,
            check_dtype=False,
            rtol=1e-12,
            atol=1e-12,
        )
    except AssertionError as exc:
        errors.append(f"Fold diagnostic mismatch: {exc}")
    if saved_summary.get("frozen_p0_p10_outputs_modified") is not False:
        errors.append("P11E must not claim modification of frozen outputs.")
    if saved_summary.get("real_data_run") is not False:
        errors.append("P11E must not claim a real-data run.")
    return {
        "schema_version": 1,
        "phase": "P11E",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "rows": int(expected_summary["rows"]),
        "firms": int(expected_summary["firms"]),
        "interpretation": expected_summary["interpretation"],
        "frozen_p0_p10_outputs_modified": False,
        "real_data_run": False,
    }
