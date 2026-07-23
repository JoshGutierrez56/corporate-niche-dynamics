"""Point-in-time robust relative and historically anchored transformations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from hypercube.config import HypercubeConfig


class StandardizationError(ValueError):
    """Raised when a P3 reference distribution violates its contract."""


def _group_statistics(
    frame: pd.DataFrame,
    value_column: str,
    keys: list[str],
    *,
    lower: float,
    upper: float,
) -> pd.DataFrame:
    values = frame[keys + [value_column]].dropna(subset=[value_column]).copy()
    if values.empty:
        return pd.DataFrame(
            columns=[*keys, "count", "median", "lower", "upper", "mad", "std"]
        )
    grouped = values.groupby(keys, observed=True, sort=False)[value_column]
    base = grouped.agg(count="count", median="median").reset_index()
    quantiles = grouped.quantile([lower, upper]).unstack().reset_index()
    quantiles = quantiles.rename(columns={lower: "lower", upper: "upper"})
    standard_deviation = grouped.std(ddof=0).rename("std").reset_index()
    deviations = values.merge(base[keys + ["median"]], on=keys, how="left")
    deviations["absolute_deviation"] = (
        deviations[value_column] - deviations["median"]
    ).abs()
    mad = (
        deviations.groupby(keys, observed=True, sort=False)["absolute_deviation"]
        .median()
        .rename("mad")
        .reset_index()
    )
    return base.merge(quantiles, on=keys, how="left").merge(
        standard_deviation, on=keys, how="left"
    ).merge(mad, on=keys, how="left")


def _robust_score(
    values: pd.Series,
    median: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
    mad: pd.Series,
    standard_deviation: pd.Series,
    clip: float,
) -> pd.Series:
    winsorized = values.clip(lower=lower, upper=upper)
    scale = 1.4826 * mad
    scale = scale.where(scale.gt(1e-12), standard_deviation)
    score = (winsorized - median).div(scale)
    constant = scale.fillna(0.0).le(1e-12) & values.notna()
    score.loc[constant] = 0.0
    return score.clip(-clip, clip)


def _event_statistics(
    peer: pd.DataFrame,
    events: pd.DataFrame,
    value_column: str,
    level: str,
    config: HypercubeConfig,
) -> pd.DataFrame:
    peer_keys = ["date"] if level == "market" else ["date", level]
    event_keys = ["feature_date"] if level == "market" else ["feature_date", level]
    stats = _group_statistics(
        peer,
        value_column,
        peer_keys,
        lower=config.standardization.winsor_lower,
        upper=config.standardization.winsor_upper,
    ).rename(columns={"date": "feature_date"})
    return events[["_row_id", *event_keys]].merge(
        stats,
        on=event_keys,
        how="left",
        validate="many_to_one",
    ).set_index("_row_id")


def contemporaneous_peer_median_impute(
    peer: pd.DataFrame,
    value_column: str,
    config: HypercubeConfig,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Impute missing values from current peers with the frozen fallback order."""

    work = peer.reset_index(drop=True).copy()
    work["_row_id"] = np.arange(len(work))
    events = work.rename(columns={"date": "feature_date"})
    result = work[value_column].copy()
    level_used = pd.Series("observed", index=work.index, dtype="string")
    count_used = pd.Series(np.nan, index=work.index, dtype="float64")
    unresolved = result.isna()
    for level in config.standardization.peer_fallback_hierarchy:
        stats = _event_statistics(work, events, value_column, level, config)
        eligible = unresolved & stats["count"].ge(
            config.standardization.minimum_peer_group_size
        )
        result.loc[eligible] = stats.loc[eligible, "median"]
        level_used.loc[eligible] = level
        count_used.loc[eligible] = stats.loc[eligible, "count"]
        unresolved &= result.isna()
    level_used.loc[unresolved] = "unavailable"
    return result, level_used, count_used


def relative_standardize_events(
    peer: pd.DataFrame,
    events: pd.DataFrame,
    component_columns: Sequence[str],
    peer_scopes: Mapping[str, str],
    config: HypercubeConfig,
) -> pd.DataFrame:
    """Score event values against only contemporaneously available peers."""

    work = events.reset_index(drop=True).copy()
    work["_row_id"] = np.arange(len(work))
    output = work.copy()
    for component in component_columns:
        levels = (
            ("market",)
            if peer_scopes.get(component, "industry") == "market"
            else config.standardization.peer_fallback_hierarchy
        )
        score = pd.Series(np.nan, index=work.index, dtype="float64")
        count_used = pd.Series(np.nan, index=work.index, dtype="float64")
        level_used = pd.Series("unavailable", index=work.index, dtype="string")
        unresolved = work[component].notna()
        for level in levels:
            stats = _event_statistics(peer, work, component, level, config)
            eligible = unresolved & stats["count"].ge(
                config.standardization.minimum_peer_group_size
            )
            candidate = _robust_score(
                work[component],
                stats["median"],
                stats["lower"],
                stats["upper"],
                stats["mad"],
                stats["std"],
                config.standardization.score_clip,
            )
            use = eligible & candidate.notna()
            score.loc[use] = candidate.loc[use]
            count_used.loc[use] = stats.loc[use, "count"]
            level_used.loc[use] = level
            unresolved &= score.isna()
        output[f"rel_{component}"] = score
        output[f"rel_n_{component}"] = count_used.astype("Int32")
        output[f"rel_level_{component}"] = level_used
    return output.drop(columns="_row_id")


def anchored_standardize_events(
    events: pd.DataFrame,
    component_columns: Sequence[str],
    config: HypercubeConfig,
) -> pd.DataFrame:
    """Score events against an expanding distribution ending the prior year."""

    output = events.reset_index(drop=True).copy()
    years = pd.to_datetime(output["feature_date"], errors="raise").dt.year.astype(int)
    output["anchor_reference_end_year"] = years - 1
    for component in component_columns:
        scores = pd.Series(np.nan, index=output.index, dtype="float64")
        counts = pd.Series(pd.NA, index=output.index, dtype="Int32")
        for year in sorted(years.unique()):
            current = years.eq(year)
            history = output.loc[years.lt(year), component].dropna()
            counts.loc[current] = len(history)
            if len(history) < config.standardization.minimum_anchor_observations:
                continue
            lower = float(history.quantile(config.standardization.winsor_lower))
            upper = float(history.quantile(config.standardization.winsor_upper))
            median = float(history.median())
            mad = float((history - median).abs().median())
            standard_deviation = float(history.std(ddof=0))
            scale = 1.4826 * mad if mad > 1e-12 else standard_deviation
            values = output.loc[current, component]
            if not np.isfinite(scale) or scale <= 1e-12:
                scores.loc[current & output[component].notna()] = 0.0
            else:
                scores.loc[current] = (
                    values.clip(lower, upper).sub(median).div(scale)
                ).clip(
                    -config.standardization.score_clip,
                    config.standardization.score_clip,
                )
        output[f"anch_{component}"] = scores
        output[f"anch_n_{component}"] = counts
    return output
