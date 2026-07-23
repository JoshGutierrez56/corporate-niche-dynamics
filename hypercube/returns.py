"""Point-in-time return tests and gross portfolio diagnostics for phase P7."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from hypercube.config import HypercubeConfig
from hypercube.data import Scenario, atomic_write_json, sha256_file
from hypercube.dynamics import scenario_p5_dir, validate_p5_directory
from hypercube.universe import (
    combine_delisting_returns,
    map_valid_ccm_links,
    scenario_processed_dir,
)


class ReturnTestError(ValueError):
    """Raised when a P7 timing, target, inference, or portfolio contract fails."""


P7_VERSION = "hypercube-returns-v1"
EVENT_KEY = ("gvkey", "datadate", "fyear")
SIGNAL_LABELS = {
    "viability_log_odds": "static_viability",
    "velocity_log_odds": "raw_migration",
    "migration_surprise": "migration_surprise",
    "anchored_axis_innovation": "anchored_axis_innovation",
}


def scenario_p7_dir(config: HypercubeConfig, scenario: Scenario) -> Path:
    """Resolve one scenario's P7 output directory."""

    return scenario_processed_dir(config, scenario) / "p7"


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


def _event_identifier(frame: pd.DataFrame) -> pd.Series:
    return (
        frame["gvkey"].astype("string")
        + "|"
        + pd.to_datetime(frame["datadate"]).dt.strftime("%Y-%m-%d")
        + "|"
        + frame["fyear"].astype(int).astype(str)
    )


def prepare_return_events(
    p5_dir: Path, config: HypercubeConfig
) -> pd.DataFrame:
    """Select the frozen five-year P5 surface without opening return outcomes."""

    required = [
        "permno",
        *EVENT_KEY,
        "horizon_years",
        "feature_date",
        "availability_date",
        "formation_date",
        "sic2",
        "fold",
        *config.returns.signals,
        *config.returns.controls,
    ]
    surface = pd.read_parquet(p5_dir / "frontier_dynamics.parquet", columns=required)
    events = surface.loc[
        surface["horizon_years"].eq(config.returns.primary_surface_horizon_years)
    ].copy()
    if events.empty:
        raise ReturnTestError("P7 found no events on the primary P5 surface.")
    for column in ("feature_date", "availability_date", "formation_date", "datadate"):
        events[column] = pd.to_datetime(events[column], errors="coerce")
    if events[list(EVENT_KEY)].isna().any(axis=None):
        raise ReturnTestError("P7 event keys contain missing values.")
    if events.duplicated(list(EVENT_KEY)).any():
        raise ReturnTestError("P7 primary-surface event keys are not unique.")
    if (events["feature_date"] < events["availability_date"]).any():
        raise ReturnTestError("P7 event precedes public availability.")
    if (events["feature_date"] < events["formation_date"]).any():
        raise ReturnTestError("P7 feature date precedes the frozen accounting formation.")
    events["event_id"] = _event_identifier(events)
    events["sic1"] = (pd.to_numeric(events["sic2"], errors="coerce") // 10).astype(
        "Int64"
    )
    events["market_cap_millions"] = np.exp(events["log_market_cap"])
    events["feature_year"] = events["feature_date"].dt.year.astype(int)
    if events["event_id"].duplicated().any():
        raise ReturnTestError("P7 event identifiers are not unique.")
    numeric = events[
        [*config.returns.signals, *config.returns.controls, "market_cap_millions"]
    ].to_numpy(float)
    if np.isinf(numeric).any():
        raise ReturnTestError("P7 event predictors contain infinities.")
    return events.sort_values(["feature_date", "gvkey"], kind="stable").reset_index(
        drop=True
    )


def build_dated_return_history(
    raw_dir: Path, config: HypercubeConfig
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Map raw security returns to dated issuers without applying future filters."""

    crsp = pd.read_parquet(raw_dir / "crsp_monthly.parquet")
    delist = pd.read_parquet(raw_dir / "crsp_delist.parquet")
    links = pd.read_parquet(raw_dir / "ccm_link.parquet")
    factors = pd.read_parquet(raw_dir / "factor_returns.parquet")
    security = combine_delisting_returns(crsp, delist)
    security = security.loc[
        security["shrcd"].isin(config.universe.share_codes)
        & security["exchcd"].isin(config.universe.exchange_codes)
    ].copy()
    security["market_cap_millions"] = (
        security["prc"].abs() * security["shrout"] / 1000.0
    )
    mapped, diagnostics = map_valid_ccm_links(security, links, config)
    mapped["gvkey"] = mapped["gvkey"].astype("string")
    duplicate_candidates = int(mapped.duplicated(["gvkey", "date"], keep=False).sum())
    mapped = mapped.sort_values(
        ["gvkey", "date", "market_cap_millions", "permno"],
        ascending=[True, True, False, True],
        kind="stable",
    ).drop_duplicates(["gvkey", "date"], keep="first")
    factors["date"] = pd.to_datetime(factors["date"], errors="coerce")
    if factors["date"].isna().any() or factors["date"].duplicated().any():
        raise ReturnTestError("Factor dates must be valid and unique.")
    factor_columns = ["date", "rf", *config.returns.factor_columns]
    factors = factors[factor_columns].sort_values("date").reset_index(drop=True)
    history_columns = [
        "gvkey",
        "date",
        "permno",
        "ret_total",
        "has_delist_event",
        "delist_return_missing",
        "exit_category",
        "market_cap_millions",
    ]
    history = mapped[history_columns].sort_values(
        ["gvkey", "date"], kind="stable"
    ).reset_index(drop=True)
    diagnostics = {
        **diagnostics,
        "duplicate_issuer_month_candidates": duplicate_candidates,
        "selected_issuer_months": int(len(history)),
        "missing_delisting_returns": int(history["delist_return_missing"].sum()),
    }
    return history, factors, diagnostics


def construct_forward_paths(
    events: pd.DataFrame,
    history: pd.DataFrame,
    factors: pd.DataFrame,
    config: HypercubeConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create strictly post-formation monthly paths and forward-return targets."""

    maximum = max(config.returns.horizons_months)
    pieces: list[pd.DataFrame] = []
    base_columns = [
        "event_id",
        "gvkey",
        "permno",
        "feature_date",
        "fold",
        *EVENT_KEY[1:],
    ]
    for offset in range(config.returns.entry_lag_months, maximum + 1):
        piece = events[base_columns].copy()
        piece["holding_offset"] = offset
        piece["holding_date"] = piece["feature_date"] + pd.offsets.MonthEnd(offset)
        pieces.append(piece)
    paths = pd.concat(pieces, ignore_index=True)
    paths = paths.merge(
        history,
        left_on=["gvkey", "holding_date"],
        right_on=["gvkey", "date"],
        how="left",
        validate="many_to_one",
        suffixes=("_formation", "_return"),
    ).drop(columns="date")
    paths = paths.merge(
        factors,
        left_on="holding_date",
        right_on="date",
        how="left",
        validate="many_to_one",
    ).drop(columns="date")
    if paths[["rf", *config.returns.factor_columns]].isna().any(axis=None):
        raise ReturnTestError("Factor coverage is incomplete over P7 holding paths.")
    paths["has_delist_event"] = paths["has_delist_event"].eq(True)
    paths["delist_return_missing"] = paths["delist_return_missing"].eq(True)
    paths = paths.sort_values(["event_id", "holding_offset"], kind="stable")
    valid_delist = (
        paths["has_delist_event"]
        & ~paths["delist_return_missing"]
        & paths["ret_total"].notna()
    )
    paths["valid_delist_event"] = valid_delist
    paths["after_valid_delist"] = (
        valid_delist.groupby(paths["event_id"], observed=True)
        .cummax()
        .groupby(paths["event_id"], observed=True)
        .shift(1, fill_value=False)
    )
    observed_security_return = (
        paths["ret_total"].notna() & ~paths["delist_return_missing"]
    )
    paths["period_valid"] = paths["after_valid_delist"] | observed_security_return
    paths["path_valid_through_month"] = (
        paths["period_valid"]
        .groupby(paths["event_id"], observed=True)
        .cummin()
        .astype(bool)
    )
    paths["return_used"] = paths["ret_total"]
    paths.loc[paths["after_valid_delist"], "return_used"] = paths.loc[
        paths["after_valid_delist"], "rf"
    ]
    paths["cash_after_delist"] = paths["after_valid_delist"]
    paths["delisted_by_month"] = valid_delist.groupby(
        paths["event_id"], observed=True
    ).cummax()
    if (paths["holding_date"] <= paths["feature_date"]).any():
        raise ReturnTestError("P7 holding return is not strictly after formation.")

    target_frames: list[pd.DataFrame] = []
    for horizon in config.returns.horizons_months:
        subset = paths.loc[paths["holding_offset"].le(horizon)].copy()
        grouped = subset.groupby("event_id", observed=True, sort=False)
        records = grouped.agg(
            path_months=("holding_offset", "size"),
            target_valid=("path_valid_through_month", "all"),
            missing_delisting_return=("delist_return_missing", "any"),
            delisted_within_horizon=("valid_delist_event", "any"),
            observed_security_months=("ret_total", "count"),
            first_holding_date=("holding_date", "min"),
            last_holding_date=("holding_date", "max"),
        )
        gross = grouped["return_used"].apply(
            lambda values: float(np.prod(1.0 + values.to_numpy(float)) - 1.0)
            if values.notna().all()
            else np.nan
        )
        risk_free = grouped["rf"].apply(
            lambda values: float(np.prod(1.0 + values.to_numpy(float)) - 1.0)
        )
        records["forward_total_return"] = gross
        records["forward_risk_free_return"] = risk_free
        records["forward_excess_return"] = gross - risk_free
        records["horizon_months"] = horizon
        records["target_status"] = "complete"
        records.loc[
            records["delisted_within_horizon"] & records["target_valid"],
            "target_status",
        ] = "complete_after_delist"
        records.loc[~records["target_valid"], "target_status"] = "missing_return_path"
        records.loc[
            records["missing_delisting_return"], "target_status"
        ] = "missing_delisting_return"
        records.loc[~records["target_valid"], [
            "forward_total_return",
            "forward_excess_return",
        ]] = np.nan
        target_frames.append(records.reset_index())
    targets = pd.concat(target_frames, ignore_index=True)
    event_columns = [
        "event_id",
        *EVENT_KEY,
        "permno",
        "feature_date",
        "availability_date",
        "formation_date",
        "feature_year",
        "fold",
        "sic2",
        "sic1",
        "market_cap_millions",
        *config.returns.signals,
        *config.returns.controls,
    ]
    targets = targets.merge(
        events[event_columns], on="event_id", how="left", validate="many_to_one"
    )
    targets["entry_timing_valid"] = (
        targets["first_holding_date"] > targets["feature_date"]
    )
    if (~targets["entry_timing_valid"]).any():
        raise ReturnTestError("P7 target begins before its permitted entry month.")
    return (
        paths.sort_values(["event_id", "holding_offset"]).reset_index(drop=True),
        targets.sort_values(["horizon_months", "feature_date", "event_id"]).reset_index(
            drop=True
        ),
    )


def _winsorize_standardize(
    frame: pd.DataFrame, columns: Iterable[str], config: HypercubeConfig
) -> pd.DataFrame:
    output = pd.DataFrame(index=frame.index)
    for column in columns:
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.notna().sum() == 0:
            output[column] = 0.0
            continue
        lower = values.quantile(config.returns.winsor_lower)
        upper = values.quantile(config.returns.winsor_upper)
        values = values.clip(lower=lower, upper=upper)
        values = values.fillna(values.median())
        scale = float(values.std(ddof=0))
        output[column] = (
            (values - float(values.mean())) / scale if scale > 1e-12 else 0.0
        )
    return output


def _hac_mean(values: pd.Series, maxlags: int) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna().to_numpy(float)
    if len(clean) < 2:
        return {
            "mean": np.nan,
            "standard_error": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
        }
    fitted = sm.OLS(clean, np.ones((len(clean), 1))).fit(
        cov_type="HAC", cov_kwds={"maxlags": maxlags}
    )
    return {
        "mean": float(fitted.params[0]),
        "standard_error": float(fitted.bse[0]),
        "t_stat": float(fitted.tvalues[0]),
        "p_value": float(fitted.pvalues[0]),
    }


def run_fama_macbeth(
    targets: pd.DataFrame, config: HypercubeConfig
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run frozen monthly cross-sectional regressions and HAC summaries."""

    monthly_rows: list[dict[str, Any]] = []
    controls = list(config.returns.controls)
    for horizon in config.returns.horizons_months:
        horizon_frame = targets.loc[
            targets["horizon_months"].eq(horizon)
            & targets["target_valid"]
            & targets["forward_excess_return"].notna()
        ]
        for signal in config.returns.signals:
            for formation_date, group in horizon_frame.groupby(
                "feature_date", observed=True, sort=True
            ):
                usable = group.loc[
                    group[signal].notna() & group["sic1"].notna()
                ].copy()
                if len(usable) < config.returns.minimum_cross_section:
                    continue
                numeric = _winsorize_standardize(
                    usable, [signal, *controls], config
                )
                dummies = pd.get_dummies(
                    usable["sic1"].astype("string"),
                    prefix="sic1",
                    drop_first=True,
                    dtype=float,
                )
                design = pd.concat([numeric, dummies], axis=1)
                design.insert(0, "intercept", 1.0)
                y = usable["forward_excess_return"].to_numpy(float)
                x = design.to_numpy(float)
                if len(usable) <= x.shape[1] + 5 or np.linalg.matrix_rank(x) < x.shape[1]:
                    continue
                params, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
                fitted = x @ params
                residual = y - fitted
                total = float(np.sum((y - y.mean()) ** 2))
                r_squared = 1.0 - float(np.sum(residual**2)) / total if total > 0 else np.nan
                rank_ic = float(
                    usable[signal].corr(
                        usable["forward_excess_return"], method="spearman"
                    )
                )
                for term, coefficient in zip(design.columns, params, strict=True):
                    monthly_rows.append(
                        {
                            "horizon_months": horizon,
                            "signal": signal,
                            "signal_label": SIGNAL_LABELS[signal],
                            "formation_date": formation_date,
                            "term": term if term not in controls else f"control:{term}",
                            "coefficient": float(coefficient),
                            "rank_ic": rank_ic if term == signal else np.nan,
                            "cross_section_rows": int(len(usable)),
                            "r_squared": r_squared,
                            "fold": int(usable["fold"].mode().iloc[0]),
                        }
                    )
    monthly = pd.DataFrame(monthly_rows)
    if monthly.empty:
        raise ReturnTestError("P7 produced no Fama-MacBeth cross-sections.")
    signal_months = monthly.loc[
        monthly.apply(lambda row: row["term"] == row["signal"], axis=1)
    ].copy()
    summary_rows: list[dict[str, Any]] = []
    for (horizon, signal), group in signal_months.groupby(
        ["horizon_months", "signal"], observed=True, sort=True
    ):
        if len(group) < config.returns.minimum_fmb_months:
            raise ReturnTestError(
                f"P7 has only {len(group)} Fama-MacBeth months for {signal}, h={horizon}."
            )
        inference = _hac_mean(group["coefficient"], max(0, int(horizon) - 1))
        ic_inference = _hac_mean(group["rank_ic"], max(0, int(horizon) - 1))
        summary_rows.append(
            {
                "horizon_months": int(horizon),
                "signal": signal,
                "signal_label": SIGNAL_LABELS[signal],
                "months": int(len(group)),
                "observations": int(group["cross_section_rows"].sum()),
                "mean_coefficient": inference["mean"],
                "hac_standard_error": inference["standard_error"],
                "t_stat": inference["t_stat"],
                "p_value": inference["p_value"],
                "mean_rank_ic": ic_inference["mean"],
                "rank_ic_hac_standard_error": ic_inference["standard_error"],
                "rank_ic_t_stat": ic_inference["t_stat"],
                "rank_ic_p_value": ic_inference["p_value"],
                "positive_fold_share": float(
                    group.groupby("fold", observed=True)["coefficient"].mean().gt(0).mean()
                ),
                "is_primary": bool(
                    horizon == config.returns.primary_horizon_months
                    and signal == config.returns.primary_signal
                ),
            }
        )
    summary = pd.DataFrame(summary_rows)
    reject, adjusted, _, _ = multipletests(
        summary["p_value"].to_numpy(float),
        alpha=0.05,
        method=config.returns.multiple_testing_method,
    )
    summary["holm_p_value"] = adjusted
    summary["holm_reject_5pct"] = reject
    return monthly, summary


def _quantiles(values: pd.Series, bins: int) -> pd.Series:
    ranks = values.rank(method="first")
    return pd.qcut(ranks, bins, labels=range(1, bins + 1)).astype(int)


def _leg_weights(
    frame: pd.DataFrame,
    scheme: str,
    industry_neutral: bool,
) -> pd.Series:
    base = (
        pd.Series(1.0, index=frame.index)
        if scheme == "equal"
        else frame["market_cap_millions"].clip(lower=0.0)
    )
    if not industry_neutral:
        total = float(base.sum())
        return base / total if total > 0 else pd.Series(np.nan, index=frame.index)
    weights = pd.Series(0.0, index=frame.index)
    industries = list(frame["sic1"].dropna().unique())
    if not industries:
        return pd.Series(np.nan, index=frame.index)
    budget = 1.0 / len(industries)
    for industry in industries:
        mask = frame["sic1"].eq(industry)
        subtotal = float(base.loc[mask].sum())
        if subtotal > 0:
            weights.loc[mask] = budget * base.loc[mask] / subtotal
    return weights


def construct_portfolio_sorts(
    targets: pd.DataFrame,
    config: HypercubeConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Form frozen gross quintile portfolios and top/bottom assignments."""

    quantile_rows: list[dict[str, Any]] = []
    assignment_frames: list[pd.DataFrame] = []
    attrition_rows: list[dict[str, Any]] = []
    for horizon in config.returns.horizons_months:
        all_horizon = targets.loc[targets["horizon_months"].eq(horizon)]
        for signal in config.returns.signals:
            for formation_date, raw_group in all_horizon.groupby(
                "feature_date", observed=True, sort=True
            ):
                candidates = raw_group.loc[
                    raw_group["target_valid"]
                    & raw_group["forward_excess_return"].notna()
                    & raw_group[signal].notna()
                    & raw_group["market_cap_millions"].gt(0.0)
                    & raw_group["sic1"].notna()
                ].copy()
                attrition_rows.append(
                    {
                        "horizon_months": horizon,
                        "signal": signal,
                        "formation_date": formation_date,
                        "events_total": int(len(raw_group)),
                        "events_eligible": int(len(candidates)),
                    }
                )
                if len(candidates) < config.returns.minimum_cross_section:
                    continue
                for neutral in config.returns.industry_neutral_modes:
                    ranked = candidates.copy()
                    if neutral:
                        ranked["quantile"] = np.nan
                        for _, industry in ranked.groupby("sic1", observed=True):
                            if len(industry) >= config.returns.minimum_industry_group:
                                ranked.loc[industry.index, "quantile"] = _quantiles(
                                    industry[signal], config.returns.quantiles
                                )
                        ranked = ranked.dropna(subset=["quantile"])
                        ranked["quantile"] = ranked["quantile"].astype(int)
                    else:
                        ranked["quantile"] = _quantiles(
                            ranked[signal], config.returns.quantiles
                        )
                    if ranked.empty:
                        continue
                    for scheme in config.returns.weighting_schemes:
                        for quantile, quantile_group in ranked.groupby(
                            "quantile", observed=True, sort=True
                        ):
                            weights = _leg_weights(
                                quantile_group, scheme, bool(neutral)
                            )
                            if weights.isna().any():
                                continue
                            quantile_rows.append(
                                {
                                    "horizon_months": horizon,
                                    "signal": signal,
                                    "signal_label": SIGNAL_LABELS[signal],
                                    "formation_date": formation_date,
                                    "fold": int(quantile_group["fold"].mode().iloc[0]),
                                    "weighting": scheme,
                                    "industry_neutral": bool(neutral),
                                    "quantile": int(quantile),
                                    "events": int(len(quantile_group)),
                                    "forward_total_return": float(
                                        np.dot(
                                            weights,
                                            quantile_group["forward_total_return"],
                                        )
                                    ),
                                    "forward_excess_return": float(
                                        np.dot(
                                            weights,
                                            quantile_group["forward_excess_return"],
                                        )
                                    ),
                                }
                            )
                            if quantile in (1, config.returns.quantiles):
                                assignment = quantile_group[
                                    [
                                        "event_id",
                                        "gvkey",
                                        "feature_date",
                                        "fold",
                                        "sic1",
                                        *config.returns.controls,
                                    ]
                                ].copy()
                                assignment["horizon_months"] = horizon
                                assignment["signal"] = signal
                                assignment["signal_label"] = SIGNAL_LABELS[signal]
                                assignment["weighting"] = scheme
                                assignment["industry_neutral"] = bool(neutral)
                                assignment["leg"] = (
                                    "long"
                                    if quantile == config.returns.quantiles
                                    else "short"
                                )
                                assignment["weight"] = weights.to_numpy(float)
                                assignment_frames.append(assignment)
    quantiles = pd.DataFrame(quantile_rows)
    assignments = pd.concat(assignment_frames, ignore_index=True)
    attrition = pd.DataFrame(attrition_rows)
    if quantiles.empty or assignments.empty:
        raise ReturnTestError("P7 produced no portfolio sorts.")
    return quantiles, assignments, attrition


def _weight_turnover(
    joined: pd.DataFrame,
    monthly_cohorts: pd.DataFrame,
) -> pd.DataFrame:
    positions = joined.copy()
    positions["signed_weight"] = np.where(
        positions["leg"].eq("long"), positions["weight"], -positions["weight"]
    )
    active = monthly_cohorts.set_index("holding_date")["active_cohorts"]
    aggregate = (
        positions.groupby(["holding_date", "gvkey"], observed=True)["signed_weight"]
        .sum()
        .reset_index()
    )
    aggregate["signed_weight"] /= aggregate["holding_date"].map(active)
    months = sorted(aggregate["holding_date"].unique())
    previous: dict[str, float] = {}
    rows = []
    for month in months:
        current_frame = aggregate.loc[aggregate["holding_date"].eq(month)]
        current = dict(zip(current_frame["gvkey"], current_frame["signed_weight"]))
        names = set(previous) | set(current)
        turnover = 0.5 * sum(
            abs(current.get(name, 0.0) - previous.get(name, 0.0)) for name in names
        )
        rows.append({"holding_date": month, "turnover": turnover})
        previous = current
    return pd.DataFrame(rows)


def build_monthly_portfolios(
    assignments: pd.DataFrame,
    paths: pd.DataFrame,
    factors: pd.DataFrame,
    config: HypercubeConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Aggregate overlapping gross cohorts, turnover, and characteristic exposures."""

    monthly_frames: list[pd.DataFrame] = []
    exposure_rows: list[dict[str, Any]] = []
    strategy_columns = [
        "horizon_months",
        "signal",
        "weighting",
        "industry_neutral",
    ]
    path_columns = [
        "event_id",
        "holding_offset",
        "holding_date",
        "return_used",
        "path_valid_through_month",
    ]
    for keys, strategy in assignments.groupby(strategy_columns, observed=True, sort=True):
        horizon, signal, weighting, neutral = keys
        eligible_paths = paths.loc[
            paths["holding_offset"].le(int(horizon)), path_columns
        ]
        joined = strategy.merge(
            eligible_paths, on="event_id", how="left", validate="many_to_many"
        )
        if (
            joined["return_used"].isna().any()
            or (~joined["path_valid_through_month"]).any()
        ):
            raise ReturnTestError("A P7 portfolio assignment has an invalid return path.")
        joined["weighted_return"] = joined["weight"] * joined["return_used"]
        cohorts = (
            joined.groupby(
                ["feature_date", "holding_date", "leg"], observed=True
            )["weighted_return"]
            .sum()
            .unstack("leg")
            .reset_index()
        )
        if not {"long", "short"}.issubset(cohorts.columns):
            continue
        cohorts["long_short_return"] = cohorts["long"] - cohorts["short"]
        calendar = (
            cohorts.groupby("holding_date", observed=True)
            .agg(
                long_return=("long", "mean"),
                short_return=("short", "mean"),
                long_short_return=("long_short_return", "mean"),
                active_cohorts=("feature_date", "nunique"),
            )
            .reset_index()
        )
        turnover = _weight_turnover(joined, calendar)
        calendar = calendar.merge(
            turnover, on="holding_date", how="left", validate="one_to_one"
        )
        calendar["horizon_months"] = int(horizon)
        calendar["signal"] = signal
        calendar["signal_label"] = SIGNAL_LABELS[signal]
        calendar["weighting"] = weighting
        calendar["industry_neutral"] = bool(neutral)
        monthly_frames.append(calendar)

        signed = np.where(strategy["leg"].eq("long"), strategy["weight"], -strategy["weight"])
        for control in config.returns.controls:
            values = pd.to_numeric(strategy[control], errors="coerce")
            valid = values.notna()
            exposure_rows.append(
                {
                    "horizon_months": int(horizon),
                    "signal": signal,
                    "signal_label": SIGNAL_LABELS[signal],
                    "weighting": weighting,
                    "industry_neutral": bool(neutral),
                    "characteristic": control,
                    "weighted_long_short_exposure": float(
                        np.sum(signed[valid] * values.loc[valid])
                        / max(1, strategy.loc[valid, "feature_date"].nunique())
                    ),
                    "observations": int(valid.sum()),
                }
            )
    monthly = pd.concat(monthly_frames, ignore_index=True)
    monthly = monthly.merge(
        factors,
        left_on="holding_date",
        right_on="date",
        how="left",
        validate="many_to_one",
    ).drop(columns="date")
    if monthly[list(config.returns.factor_columns)].isna().any(axis=None):
        raise ReturnTestError("P7 monthly portfolios lack factor coverage.")
    summaries: list[dict[str, Any]] = []
    for keys, group in monthly.groupby(strategy_columns, observed=True, sort=True):
        horizon, signal, weighting, neutral = keys
        group = group.sort_values("holding_date")
        y = group["long_short_return"].to_numpy(float)
        x = sm.add_constant(group[list(config.returns.factor_columns)].to_numpy(float))
        fitted = sm.OLS(y, x).fit(
            cov_type="HAC", cov_kwds={"maxlags": max(0, int(horizon) - 1)}
        )
        wealth = np.cumprod(1.0 + y)
        peak = np.maximum.accumulate(wealth)
        drawdown = wealth / peak - 1.0
        volatility = float(np.std(y, ddof=1))
        row: dict[str, Any] = {
            "horizon_months": int(horizon),
            "signal": signal,
            "signal_label": SIGNAL_LABELS[signal],
            "weighting": weighting,
            "industry_neutral": bool(neutral),
            "months": int(len(group)),
            "first_month": group["holding_date"].min(),
            "last_month": group["holding_date"].max(),
            "mean_monthly_gross_return": float(np.mean(y)),
            "annualized_arithmetic_gross_return": float(12.0 * np.mean(y)),
            "annualized_gross_sharpe": (
                float(math.sqrt(12.0) * np.mean(y) / volatility)
                if volatility > 0
                else np.nan
            ),
            "maximum_drawdown": float(np.min(drawdown)),
            "hit_rate": float(np.mean(y > 0.0)),
            "average_turnover": float(group["turnover"].mean()),
            "average_active_cohorts": float(group["active_cohorts"].mean()),
            "factor_alpha_monthly": float(fitted.params[0]),
            "factor_alpha_hac_standard_error": float(fitted.bse[0]),
            "factor_alpha_t_stat": float(fitted.tvalues[0]),
            "factor_alpha_p_value": float(fitted.pvalues[0]),
            "factor_r_squared": float(fitted.rsquared),
            "net_return_computed": False,
        }
        for index, factor in enumerate(config.returns.factor_columns, start=1):
            row[f"beta_{factor}"] = float(fitted.params[index])
        summaries.append(row)
    return monthly, pd.DataFrame(summaries), pd.DataFrame(exposure_rows)


def portfolio_cohort_diagnostics(
    quantiles: pd.DataFrame,
    config: HypercubeConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Report monotonicity, long-short spreads, and frozen-fold stability."""

    keys = [
        "horizon_months",
        "signal",
        "signal_label",
        "formation_date",
        "fold",
        "weighting",
        "industry_neutral",
    ]
    spreads: list[dict[str, Any]] = []
    for group_keys, group in quantiles.groupby(keys, observed=True, sort=True):
        indexed = group.set_index("quantile")
        if not {1, config.returns.quantiles}.issubset(indexed.index):
            continue
        values = group.sort_values("quantile")
        monotonicity = values["quantile"].corr(
            values["forward_excess_return"], method="spearman"
        )
        row = dict(zip(keys, group_keys, strict=True))
        row.update(
            {
                "long_short_total_return": float(
                    indexed.loc[config.returns.quantiles, "forward_total_return"]
                    - indexed.loc[1, "forward_total_return"]
                ),
                "long_short_excess_return": float(
                    indexed.loc[config.returns.quantiles, "forward_excess_return"]
                    - indexed.loc[1, "forward_excess_return"]
                ),
                "quantile_monotonicity": float(monotonicity),
            }
        )
        spreads.append(row)
    cohort = pd.DataFrame(spreads)
    subgroup_rows: list[dict[str, Any]] = []
    grouping = [
        "horizon_months",
        "signal",
        "signal_label",
        "weighting",
        "industry_neutral",
        "fold",
    ]
    for group_keys, group in cohort.groupby(grouping, observed=True, sort=True):
        horizon = int(group_keys[0])
        inference = _hac_mean(
            group["long_short_excess_return"], max(0, horizon - 1)
        )
        row = dict(zip(grouping, group_keys, strict=True))
        row.update(
            {
                "formation_cohorts": int(len(group)),
                "mean_long_short_excess_return": inference["mean"],
                "hac_standard_error": inference["standard_error"],
                "t_stat": inference["t_stat"],
                "p_value": inference["p_value"],
                "mean_monotonicity": float(group["quantile_monotonicity"].mean()),
            }
        )
        subgroup_rows.append(row)
    return cohort, pd.DataFrame(subgroup_rows)


def _target_attrition(targets: pd.DataFrame) -> pd.DataFrame:
    return (
        targets.groupby(["horizon_months", "target_status"], observed=True)
        .size()
        .rename("events")
        .reset_index()
    )


def build_return_bundle(
    p2_dir: Path,
    p5_dir: Path,
    raw_dir: Path,
    output_dir: Path,
    config: HypercubeConfig,
    *,
    scenario: str,
) -> dict[str, Any]:
    """Build one atomic P7 bundle without reading synthetic truth."""

    if config.project.phase != "P7":
        raise ReturnTestError("Return testing requires a P7 config.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite completed P7 output: {output_dir}")
    p5_validation = validate_p5_directory(
        p5_dir,
        p2_dir,
        p2_dir / "p3",
        p2_dir / "p4",
        raw_dir,
        config,
        synthetic_truth_path=None,
    )
    if p5_validation["status"] != "PASS":
        raise ReturnTestError(f"P5 validation failed before P7: {p5_validation['errors']}")
    events = prepare_return_events(p5_dir, config)
    history, factors, link_diagnostics = build_dated_return_history(raw_dir, config)
    paths, targets = construct_forward_paths(events, history, factors, config)
    fmb_monthly, fmb_summary = run_fama_macbeth(targets, config)
    quantiles, assignments, sort_attrition = construct_portfolio_sorts(targets, config)
    monthly, factor_results, exposures = build_monthly_portfolios(
        assignments, paths, factors, config
    )
    cohort, subsamples = portfolio_cohort_diagnostics(quantiles, config)
    target_attrition = _target_attrition(targets)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".p7-staging-", dir=output_dir.parent))
    try:
        _atomic_parquet(events, staging / "return_events.parquet")
        _atomic_parquet(paths, staging / "event_month_paths.parquet")
        _atomic_parquet(targets, staging / "forward_return_targets.parquet")
        _atomic_csv(fmb_monthly, staging / "fmb_monthly_coefficients.csv")
        _atomic_csv(fmb_summary, staging / "fmb_summary.csv")
        _atomic_csv(quantiles, staging / "portfolio_quantile_returns.csv")
        _atomic_parquet(assignments, staging / "portfolio_assignments.parquet")
        _atomic_csv(monthly, staging / "portfolio_monthly_returns.csv")
        _atomic_csv(factor_results, staging / "portfolio_factor_results.csv")
        _atomic_csv(exposures, staging / "portfolio_exposures.csv")
        _atomic_csv(cohort, staging / "portfolio_cohort_spreads.csv")
        _atomic_csv(subsamples, staging / "subsample_results.csv")
        _atomic_csv(target_attrition, staging / "target_attrition.csv")
        _atomic_csv(sort_attrition, staging / "portfolio_sort_attrition.csv")
        atomic_write_json(
            staging / "return_metadata.json",
            {
                "schema_version": 1,
                "phase": "P7",
                "version": P7_VERSION,
                "scenario": scenario,
                "primary_surface_horizon_years": config.returns.primary_surface_horizon_years,
                "primary_signal": config.returns.primary_signal,
                "primary_horizon_months": config.returns.primary_horizon_months,
                "horizons_months": list(config.returns.horizons_months),
                "entry_rule": "first monthly return strictly after feature_date",
                "target_rule": "complete path or valid delisting; risk-free after delisting",
                "fmb_hac_rule": config.returns.overlapping_hac_rule,
                "multiple_testing_method": config.returns.multiple_testing_method,
                "link_diagnostics": link_diagnostics,
                "synthetic_truth_read": False,
                "costs_applied": False,
                "borrow_model_applied": False,
                "capacity_model_applied": False,
                "return_test_run": True,
                "portfolio_run": True,
            },
        )
        atomic_write_json(staging / "resolved_config.json", config.model_dump(mode="json"))
        input_paths = [
            p2_dir / "p2_manifest.json",
            p5_dir / "p5_manifest.json",
            raw_dir / "crsp_monthly.parquet",
            raw_dir / "crsp_delist.parquet",
            raw_dir / "ccm_link.parquet",
            raw_dir / "factor_returns.parquet",
        ]
        output_paths = [
            path
            for path in staging.iterdir()
            if path.is_file() and path.name != "p7_manifest.json"
        ]
        manifest = {
            "schema_version": 1,
            "phase": "P7",
            "scenario": scenario,
            "seed": config.project.seed,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "inputs": [_file_record(path) for path in input_paths],
            "outputs": [_file_record(path) for path in output_paths],
            "rows": {
                "events": int(len(events)),
                "paths": int(len(paths)),
                "targets": int(len(targets)),
                "valid_targets": int(targets["target_valid"].sum()),
                "fmb_monthly_coefficients": int(len(fmb_monthly)),
                "portfolio_months": int(len(monthly)),
            },
            "synthetic_truth_read": False,
            "costs_applied": False,
        }
        atomic_write_json(staging / "p7_manifest.json", manifest)
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "scenario": scenario,
        "events": int(len(events)),
        "targets": int(len(targets)),
        "valid_targets": int(targets["target_valid"].sum()),
        "fmb_tests": int(len(fmb_summary)),
        "gross_portfolios": int(len(factor_results)),
        "return_test_run": True,
        "portfolio_run": True,
        "costs_applied": False,
        "synthetic_truth_read": False,
    }


def _recompute_forward_targets(
    paths: pd.DataFrame, targets: pd.DataFrame, config: HypercubeConfig
) -> list[str]:
    errors: list[str] = []
    valid = targets.loc[targets["target_valid"]]
    for horizon in config.returns.horizons_months:
        subset = paths.loc[paths["holding_offset"].le(horizon)]
        recomputed = (
            subset.groupby("event_id", observed=True)["return_used"]
            .apply(
                lambda values: float(np.prod(1.0 + values.to_numpy(float)) - 1.0)
                if values.notna().all()
                else np.nan
            )
            .rename("recomputed")
        )
        comparison = valid.loc[
            valid["horizon_months"].eq(horizon),
            ["event_id", "forward_total_return"],
        ].merge(recomputed, on="event_id", validate="one_to_one")
        if not np.allclose(
            comparison["forward_total_return"],
            comparison["recomputed"],
            rtol=1e-12,
            atol=1e-12,
        ):
            errors.append(f"P7 forward-return reconstruction failed at {horizon} months.")
    return errors


def _synthetic_recovery(
    output_dir: Path,
    truth_path: Path,
    fmb_summary: pd.DataFrame,
    config: HypercubeConfig,
    scenario: str,
) -> dict[str, Any]:
    targets = pd.read_parquet(output_dir / "forward_return_targets.parquet")
    primary = targets.loc[
        targets["horizon_months"].eq(config.returns.primary_horizon_months)
        & targets["target_valid"]
        & targets[config.returns.primary_signal].notna()
    ].copy()
    truth = pd.read_parquet(
        truth_path,
        columns=[*EVENT_KEY, "injected_return_alpha"],
    )
    joined = primary.merge(truth, on=list(EVENT_KEY), how="left", validate="one_to_one")
    if joined["injected_return_alpha"].isna().any():
        raise ReturnTestError("Synthetic P7 recovery could not match all truth rows.")
    primary_ic = float(
        joined[config.returns.primary_signal].corr(
            joined["forward_excess_return"], method="spearman"
        )
    )
    primary_fmb = fmb_summary.loc[
        fmb_summary["is_primary"].astype(bool)
    ].iloc[0]
    recovery: dict[str, Any] = {
        "scenario": scenario,
        "rows": int(len(joined)),
        "primary_signal": config.returns.primary_signal,
        "primary_horizon_months": config.returns.primary_horizon_months,
        "primary_spearman_ic": primary_ic,
        "primary_fmb_coefficient": float(primary_fmb["mean_coefficient"]),
        "primary_fmb_p_value": float(primary_fmb["p_value"]),
        "primary_fmb_holm_p_value": float(primary_fmb["holm_p_value"]),
        "truth_field_read": "injected_return_alpha",
    }
    if scenario == "null_alpha":
        gates = {
            "absolute_ic_below_frozen_bound": abs(primary_ic)
            <= config.returns.synthetic_null_max_abs_primary_ic,
            "primary_holm_not_significant": float(primary_fmb["holm_p_value"]) >= 0.05,
        }
    elif scenario == "migration_alpha":
        x = joined["injected_return_alpha"].to_numpy(float)
        y = joined["forward_excess_return"].to_numpy(float)
        design = np.column_stack([np.ones(len(x)), x])
        slope = float(np.linalg.lstsq(design, y, rcond=None)[0][1])
        expected_fraction = float(
            sum(
                math.exp(-offset / 4.0)
                for offset in range(config.returns.primary_horizon_months)
            )
            / sum(
                math.exp(-offset / 4.0)
                for offset in range(config.synthetic.migration_decay_months)
            )
        )
        recovery["oracle_injected_alpha_slope"] = slope
        recovery["expected_six_month_fraction_of_total_injection"] = expected_fraction
        gates = {
            "primary_ic_positive": primary_ic
            > config.returns.synthetic_migration_min_primary_ic,
            "primary_fmb_coefficient_positive": float(
                primary_fmb["mean_coefficient"]
            )
            > 0.0,
            "oracle_slope_within_frozen_bounds": (
                config.returns.synthetic_oracle_slope_lower
                <= slope
                <= config.returns.synthetic_oracle_slope_upper
            ),
        }
    else:
        gates = {}
    recovery["gates"] = gates
    recovery["scientific_status"] = (
        "PASS" if all(gates.values()) else "FAIL"
    ) if gates else "DESCRIPTIVE_ONLY"
    return recovery


def validate_p7_directory(
    output_dir: Path,
    p2_dir: Path,
    p5_dir: Path,
    raw_dir: Path,
    config: HypercubeConfig,
    *,
    scenario: str,
    synthetic_truth_path: Path | None = None,
) -> dict[str, Any]:
    """Independently validate saved P7 timing, targets, inference, and hashes."""

    required = (
        "return_events.parquet",
        "event_month_paths.parquet",
        "forward_return_targets.parquet",
        "fmb_monthly_coefficients.csv",
        "fmb_summary.csv",
        "portfolio_quantile_returns.csv",
        "portfolio_assignments.parquet",
        "portfolio_monthly_returns.csv",
        "portfolio_factor_results.csv",
        "portfolio_exposures.csv",
        "portfolio_cohort_spreads.csv",
        "subsample_results.csv",
        "target_attrition.csv",
        "portfolio_sort_attrition.csv",
        "return_metadata.json",
        "resolved_config.json",
        "p7_manifest.json",
    )
    errors = [f"Missing P7 file: {name}" for name in required if not (output_dir / name).is_file()]
    if errors:
        return {"status": "FAIL", "errors": errors, "warnings": []}
    manifest = json.loads((output_dir / "p7_manifest.json").read_text(encoding="utf-8"))
    for record in manifest["inputs"]:
        path = Path(record["path"])
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            errors.append(f"P7 input hash mismatch: {path.name}")
    for record in manifest["outputs"]:
        path = output_dir / Path(record["path"]).name
        if sha256_file(path) != record["sha256"]:
            errors.append(f"P7 output hash mismatch: {path.name}")
        if path.suffix == ".parquet" and pq.ParquetFile(path).metadata.num_rows != record["rows"]:
            errors.append(f"P7 parquet row-count mismatch: {path.name}")
        if path.suffix == ".csv":
            rows = max(0, sum(1 for _ in path.open("r", encoding="utf-8")) - 1)
            if rows != record["rows"]:
                errors.append(f"P7 CSV row-count mismatch: {path.name}")
    metadata = json.loads((output_dir / "return_metadata.json").read_text(encoding="utf-8"))
    if metadata.get("synthetic_truth_read") is not False:
        errors.append("P7 construction metadata does not exclude synthetic truth.")
    if metadata.get("costs_applied") is not False:
        errors.append("P7 applied costs before P8.")
    events = pd.read_parquet(output_dir / "return_events.parquet")
    paths = pd.read_parquet(output_dir / "event_month_paths.parquet")
    targets = pd.read_parquet(output_dir / "forward_return_targets.parquet")
    fmb_monthly = pd.read_csv(
        output_dir / "fmb_monthly_coefficients.csv",
        parse_dates=["formation_date"],
    )
    fmb_summary = pd.read_csv(output_dir / "fmb_summary.csv")
    factor_results = pd.read_csv(output_dir / "portfolio_factor_results.csv")
    monthly = pd.read_csv(
        output_dir / "portfolio_monthly_returns.csv",
        parse_dates=["holding_date"],
    )
    if events.duplicated("event_id").any():
        errors.append("P7 return event identifiers are duplicated.")
    if paths.duplicated(["event_id", "holding_offset"]).any():
        errors.append("P7 event-month path keys are duplicated.")
    if len(paths) != len(events) * max(config.returns.horizons_months):
        errors.append("P7 event-month path does not contain twelve rows per event.")
    if targets.duplicated(["event_id", "horizon_months"]).any():
        errors.append("P7 forward-target keys are duplicated.")
    if len(targets) != len(events) * len(config.returns.horizons_months):
        errors.append("P7 target ladder is incomplete.")
    if (paths["holding_date"] <= paths["feature_date"]).any():
        errors.append("P7 path contains a pre-formation holding return.")
    if targets.loc[~targets["target_valid"], "forward_total_return"].notna().any():
        errors.append("P7 converted an invalid/missing path into a return target.")
    after_delist = paths["cash_after_delist"].astype(bool)
    if not np.allclose(
        paths.loc[after_delist, "return_used"],
        paths.loc[after_delist, "rf"],
        rtol=0.0,
        atol=0.0,
    ):
        errors.append("P7 post-delisting cash returns do not equal the risk-free rate.")
    errors.extend(_recompute_forward_targets(paths, targets, config))
    signal_months = fmb_monthly.loc[
        fmb_monthly.apply(lambda row: row["term"] == row["signal"], axis=1)
    ]
    for _, saved in fmb_summary.iterrows():
        group = signal_months.loc[
            signal_months["horizon_months"].eq(saved["horizon_months"])
            & signal_months["signal"].eq(saved["signal"])
        ]
        recalculated = _hac_mean(
            group["coefficient"], max(0, int(saved["horizon_months"]) - 1)
        )
        if not np.isclose(
            saved["mean_coefficient"], recalculated["mean"], rtol=1e-10, atol=1e-12
        ):
            errors.append(
                f"P7 Fama-MacBeth summary mismatch: {saved['signal']}, {saved['horizon_months']}"
            )
    if len(fmb_summary) != len(config.returns.horizons_months) * len(config.returns.signals):
        errors.append("P7 Fama-MacBeth signal/horizon ladder is incomplete.")
    _, adjusted, _, _ = multipletests(
        fmb_summary["p_value"].to_numpy(float),
        alpha=0.05,
        method=config.returns.multiple_testing_method,
    )
    if not np.allclose(
        fmb_summary["holm_p_value"], adjusted, rtol=1e-12, atol=1e-12
    ):
        errors.append("P7 multiple-testing adjustment does not recompute.")
    expected_portfolios = (
        len(config.returns.horizons_months)
        * len(config.returns.signals)
        * len(config.returns.weighting_schemes)
        * len(config.returns.industry_neutral_modes)
    )
    if len(factor_results) != expected_portfolios:
        errors.append("P7 gross portfolio ladder is incomplete.")
    if factor_results["net_return_computed"].astype(str).str.lower().eq("true").any():
        errors.append("P7 portfolio output contains net returns.")
    if monthly["turnover"].isna().any() or (monthly["turnover"] < 0.0).any():
        errors.append("P7 turnover is missing or negative.")
    strategy_columns = [
        "horizon_months",
        "signal",
        "weighting",
        "industry_neutral",
    ]
    factor_copy = factor_results.copy()
    factor_copy["industry_neutral"] = (
        factor_copy["industry_neutral"].astype(str).str.lower().eq("true")
    )
    monthly_copy = monthly.copy()
    monthly_copy["industry_neutral"] = (
        monthly_copy["industry_neutral"].astype(str).str.lower().eq("true")
    )
    for _, saved in factor_copy.iterrows():
        group = monthly_copy.loc[
            monthly_copy["horizon_months"].eq(saved["horizon_months"])
            & monthly_copy["signal"].eq(saved["signal"])
            & monthly_copy["weighting"].eq(saved["weighting"])
            & monthly_copy["industry_neutral"].eq(saved["industry_neutral"])
        ].sort_values("holding_date")
        y = group["long_short_return"].to_numpy(float)
        x = sm.add_constant(
            group[list(config.returns.factor_columns)].to_numpy(float)
        )
        fitted = sm.OLS(y, x).fit(
            cov_type="HAC",
            cov_kwds={"maxlags": max(0, int(saved["horizon_months"]) - 1)},
        )
        if not np.isclose(
            saved["factor_alpha_monthly"], fitted.params[0], rtol=1e-10, atol=1e-12
        ):
            errors.append(
                f"P7 factor alpha does not recompute: {saved['signal']}, "
                f"{saved['horizon_months']}, {saved['weighting']}, "
                f"neutral={saved['industry_neutral']}"
            )
        if not np.isclose(
            saved["average_turnover"], group["turnover"].mean(), rtol=1e-10, atol=1e-12
        ):
            errors.append("P7 average turnover does not recompute.")
    recovery = None
    if synthetic_truth_path is not None:
        recovery = _synthetic_recovery(
            output_dir,
            synthetic_truth_path,
            fmb_summary,
            config,
            scenario,
        )
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "warnings": [],
        "scenario": scenario,
        "rows": {
            "events": int(len(events)),
            "paths": int(len(paths)),
            "targets": int(len(targets)),
            "valid_targets": int(targets["target_valid"].sum()),
            "portfolio_months": int(len(monthly)),
        },
        "fmb_tests": int(len(fmb_summary)),
        "gross_portfolios": int(len(factor_results)),
        "recovery": recovery,
        "costs_applied": False,
        "return_test_run": True,
        "portfolio_run": True,
    }
