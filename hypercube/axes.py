"""Transparent, swappable accounting components and six-axis composites."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Literal, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from hypercube.config import HypercubeConfig
from hypercube.data import SCENARIOS, Scenario, atomic_write_json, sha256_file
from hypercube.diagnostics import (
    correlation_long,
    count_nonfinite,
    missingness_table,
    vif_table,
)
from hypercube.standardize import (
    anchored_standardize_events,
    contemporaneous_peer_median_impute,
    relative_standardize_events,
)


PeerScope = Literal["industry", "market"]


class AxisConstructionError(ValueError):
    """Raised when P3 component construction violates its data contract."""


@dataclass(frozen=True)
class ComponentSpec:
    """Economic and implementation contract for one primary component."""

    slug: str
    name: str
    axis_slug: str
    formula: str
    required_inputs: tuple[str, ...]
    availability_rule: str
    missing_policy: str
    expected_sign_with_viability: str
    orientation_sign: int
    peer_scope: PeerScope
    known_limitations: str
    ablation_enabled: bool = True
    included_in_baseline_composite: bool = True


COMPONENT_SPECS: tuple[ComponentSpec, ...] = (
    ComponentSpec(
        "gross_margin",
        "Gross margin",
        "demand_strength_pricing_power",
        "(sale - cogs) / sale, requiring sale > 0",
        ("sale", "cogs"),
        "current eligible accounting release",
        "missing if sale is nonpositive or an input is missing",
        "positive",
        1,
        "industry",
        "Accounting classification and business mix affect comparability.",
    ),
    ComponentSpec(
        "revenue_cagr_3y",
        "Three-year revenue CAGR",
        "demand_strength_pricing_power",
        "(sale_t / sale_t-3) ** (1/3) - 1 for consecutive fiscal years",
        ("sale", "fyear"),
        "current release and the release from three fiscal years earlier",
        "missing without four consecutive positive-sales observations",
        "positive",
        1,
        "industry",
        "CAGR can be distorted by acquisitions, disposals, and small bases.",
    ),
    ComponentSpec(
        "gross_margin_volatility_3y",
        "Three-year gross-margin volatility",
        "demand_strength_pricing_power",
        "population standard deviation of gross margin over t, t-1, t-2",
        ("sale", "cogs", "fyear"),
        "current and two prior eligible releases",
        "missing without three consecutive valid gross margins",
        "negative",
        -1,
        "industry",
        "Three observations provide a noisy stability estimate.",
    ),
    ComponentSpec(
        "industry_sales_hhi",
        "Contemporaneous industry sales HHI",
        "competitive_defensibility",
        "sum of squared positive-sales shares among current SIC2 peers",
        ("sale", "sich"),
        "current eligible peer snapshot at feature_date",
        "missing if peer sales are unavailable or total positive sales is zero",
        "ambiguous; baseline treats concentration as defensibility",
        1,
        "market",
        "Compustat-style HHI can mismeasure product markets and concentration can signal either power or threat.",
    ),
    ComponentSpec(
        "log_active_firm_count",
        "Log active-firm count",
        "competitive_defensibility",
        "log(1 + distinct current eligible firms in SIC2)",
        ("sich",),
        "current eligible peer snapshot at feature_date",
        "missing if industry is unavailable",
        "negative",
        -1,
        "market",
        "Public-firm counts omit private competitors and differ by industry definition.",
    ),
    ComponentSpec(
        "rd_intensity_zero",
        "R&D intensity, missing set to zero",
        "innovation_intensity",
        "max(coalesce(xrd, 0), 0) / sale, requiring sale > 0",
        ("xrd", "sale"),
        "current eligible accounting release",
        "missing R&D is zero in the primary variant and separately flagged",
        "positive",
        1,
        "industry",
        "Missing R&D may mean zero, immaterial, or undisclosed; alternatives are retained.",
    ),
    ComponentSpec(
        "capitalized_rnd_stock_assets",
        "Capitalized R&D stock to assets",
        "innovation_intensity",
        "R&D stock_t = max(xrd_t,0) + 0.8**year_gap * stock_t-1; divide by assets",
        ("xrd", "at", "fyear"),
        "current and prior eligible accounting releases",
        "missing R&D is zero; missing or nonpositive assets makes the ratio missing",
        "positive",
        1,
        "industry",
        "The 20% depreciation rate is a fixed accounting proxy, not an estimated useful life.",
    ),
    ComponentSpec(
        "rnd_positive_share_3y",
        "Three-year R&D persistence",
        "innovation_intensity",
        "share of t, t-1, t-2 with reported positive R&D after zero treatment",
        ("xrd", "fyear"),
        "current and two prior eligible releases",
        "missing without three consecutive reports",
        "positive",
        1,
        "industry",
        "Persistence measures reporting continuity, not innovation output or quality.",
    ),
    ComponentSpec(
        "gross_profit_lag_sga",
        "Gross profit to lagged SG&A",
        "go_to_market_efficiency",
        "(sale - cogs) / xsga_t-1 for consecutive fiscal years",
        ("sale", "cogs", "xsga", "fyear"),
        "current release and prior eligible SG&A",
        "missing if lagged SG&A is nonpositive or history is nonconsecutive",
        "positive",
        1,
        "industry",
        "SG&A mixes selling, administrative, and sometimes R&D-like expenses.",
    ),
    ComponentSpec(
        "revenue_change_lag_sga",
        "Revenue change to lagged SG&A",
        "go_to_market_efficiency",
        "(sale_t - sale_t-1) / xsga_t-1 for consecutive fiscal years",
        ("sale", "xsga", "fyear"),
        "current and prior eligible releases",
        "missing if lagged SG&A is nonpositive or history is nonconsecutive",
        "positive",
        1,
        "industry",
        "Organic growth cannot be separated from acquisitions with these fields.",
    ),
    ComponentSpec(
        "advertising_intensity",
        "Advertising intensity",
        "go_to_market_efficiency",
        "xad / sale, requiring sale > 0",
        ("xad", "sale"),
        "current eligible accounting release",
        "left missing when advertising is not reported; no zero imputation",
        "positive but ambiguous",
        1,
        "industry",
        "Reported advertising is sparse and spending intensity is not efficiency.",
    ),
    ComponentSpec(
        "operating_margin",
        "Operating margin",
        "unit_economics_profit_quality",
        "oiadp / sale, requiring sale > 0",
        ("oiadp", "sale"),
        "current eligible accounting release",
        "missing for invalid denominator or input",
        "positive",
        1,
        "industry",
        "Accounting policies and cyclicality affect margins.",
    ),
    ComponentSpec(
        "cash_adjusted_roic",
        "Fixed-tax NOPAT return on lagged invested capital",
        "unit_economics_profit_quality",
        "oiadp * 0.75 / lag(dltt + dlc + ceq - che)",
        ("oiadp", "dltt", "dlc", "ceq", "che", "fyear"),
        "current release and prior eligible balance sheet",
        "missing if lagged invested capital is nonpositive or history is nonconsecutive",
        "positive",
        1,
        "industry",
        "A fixed 25% tax rate and book invested capital are coarse proxies.",
    ),
    ComponentSpec(
        "gross_profitability_lag_assets",
        "Gross profitability on lagged assets",
        "unit_economics_profit_quality",
        "(sale - cogs) / at_t-1 for consecutive fiscal years",
        ("sale", "cogs", "at", "fyear"),
        "current release and prior eligible assets",
        "missing if lagged assets are nonpositive or history is nonconsecutive",
        "positive",
        1,
        "industry",
        "Expensed intangible investment can inflate apparent profitability.",
    ),
    ComponentSpec(
        "cash_conversion_lag_assets",
        "Internal-cash-after-capex proxy on lagged assets",
        "unit_economics_profit_quality",
        "(ib + dp - capx) / at_t-1 for consecutive fiscal years",
        ("ib", "dp", "capx", "at", "fyear"),
        "current release and prior eligible assets",
        "missing if lagged assets are nonpositive or history is nonconsecutive",
        "positive",
        1,
        "industry",
        "This is not operating cash flow and omits working-capital accruals.",
        included_in_baseline_composite=False,
    ),
    ComponentSpec(
        "asset_turnover",
        "Asset turnover",
        "scalability_capital_efficiency",
        "sale / at_t-1 for consecutive fiscal years",
        ("sale", "at", "fyear"),
        "current release and prior eligible assets",
        "missing if lagged assets are nonpositive or history is nonconsecutive",
        "positive",
        1,
        "industry",
        "Asset-light firms may expense economically important intangible assets.",
    ),
    ComponentSpec(
        "asset_lightness",
        "Asset lightness",
        "scalability_capital_efficiency",
        "1 - ppent / at, requiring assets > 0",
        ("ppent", "at"),
        "current eligible accounting release",
        "missing for invalid denominator or input",
        "positive but industry-dependent",
        1,
        "industry",
        "Expensed intangible assets can create mechanical apparent lightness.",
    ),
    ComponentSpec(
        "incremental_operating_margin",
        "Historical incremental operating margin",
        "scalability_capital_efficiency",
        "(oiadp_t - oiadp_t-1) / (sale_t - sale_t-1), requiring |delta sale| >= 1% of lagged sale",
        ("oiadp", "sale", "fyear"),
        "current and prior eligible releases",
        "missing for nonconsecutive history or economically tiny revenue change",
        "positive",
        1,
        "industry",
        "The ratio is unstable around small revenue changes and can reflect restructuring.",
    ),
    ComponentSpec(
        "capex_intensity",
        "Capital expenditure intensity",
        "scalability_capital_efficiency",
        "capx / sale, requiring sale > 0",
        ("capx", "sale"),
        "current eligible accounting release",
        "missing for invalid denominator or input",
        "negative",
        -1,
        "industry",
        "Capital intensity can reflect valuable growth investment rather than inefficiency.",
    ),
)

COMPONENT_BY_SLUG: Mapping[str, ComponentSpec] = {
    item.slug: item for item in COMPONENT_SPECS
}
AXIS_COMPONENTS: Mapping[str, tuple[str, ...]] = {
    axis: tuple(
        item.slug
        for item in COMPONENT_SPECS
        if item.axis_slug == axis and item.included_in_baseline_composite
    )
    for axis in (
        "demand_strength_pricing_power",
        "competitive_defensibility",
        "innovation_intensity",
        "go_to_market_efficiency",
        "unit_economics_profit_quality",
        "scalability_capital_efficiency",
    )
}
DIAGNOSTIC_COMPONENTS: tuple[str, ...] = (
    "rd_intensity_observed",
    "rd_missing_indicator",
    "rd_intensity_peer_imputed",
    "sga_intensity",
)


def component_catalog() -> list[dict[str, object]]:
    """Return the serializable predeclared component contract."""

    return [asdict(item) for item in COMPONENT_SPECS]


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    valid = numerator.notna() & denominator.notna() & denominator.gt(0.0)
    result = pd.Series(np.nan, index=numerator.index, dtype="float64")
    result.loc[valid] = numerator.loc[valid] / denominator.loc[valid]
    return result


def _require_columns(frame: pd.DataFrame, names: set[str]) -> None:
    missing = sorted(names.difference(frame.columns))
    if missing:
        raise AxisConstructionError(f"Accounting table is missing P3 inputs: {missing}")


def construct_accounting_components(
    accounting: pd.DataFrame, config: HypercubeConfig
) -> pd.DataFrame:
    """Construct raw accounting components using only current and prior releases."""

    required = {
        "gvkey",
        "datadate",
        "fyear",
        "formation_date",
        "availability_date",
        "sich",
        "sale",
        "cogs",
        "xrd",
        "xsga",
        "xad",
        "oiadp",
        "ib",
        "dp",
        "at",
        "ppent",
        "capx",
        "dltt",
        "dlc",
        "ceq",
        "che",
    }
    _require_columns(accounting, required)
    if accounting.duplicated(["gvkey", "datadate", "fyear"]).any():
        raise AxisConstructionError("Duplicate accounting keys are rejected in P3.")
    if accounting.duplicated(["gvkey", "fyear"]).any():
        raise AxisConstructionError(
            "Multiple annual observations for one gvkey-fyear are rejected in P3."
        )
    frame = accounting.copy()
    frame["gvkey"] = frame["gvkey"].astype("string")
    frame["fyear"] = pd.to_numeric(frame["fyear"], errors="raise").astype(int)
    frame = frame.sort_values(
        ["gvkey", "fyear", "formation_date", "datadate"], kind="stable"
    ).reset_index(drop=True)
    if (frame["formation_date"] <= frame["availability_date"]).any():
        raise AxisConstructionError("P3 received a pre-availability accounting row.")

    group = frame.groupby("gvkey", sort=False, observed=True)
    prior_fyear = group["fyear"].shift(1)
    consecutive1 = frame["fyear"].sub(prior_fyear).eq(1)
    prior_consecutive1 = consecutive1.groupby(frame["gvkey"], sort=False).shift(1)
    prior2_consecutive1 = consecutive1.groupby(frame["gvkey"], sort=False).shift(2)
    consecutive2 = consecutive1 & prior_consecutive1.eq(True)
    consecutive3 = consecutive2 & prior2_consecutive1.eq(True)

    gross_profit = frame["sale"] - frame["cogs"]
    frame["gross_margin"] = _safe_divide(gross_profit, frame["sale"])
    sale_lag3 = group["sale"].shift(3)
    cagr_valid = consecutive3 & frame["sale"].gt(0.0) & sale_lag3.gt(0.0)
    frame["revenue_cagr_3y"] = np.nan
    frame.loc[cagr_valid, "revenue_cagr_3y"] = (
        frame.loc[cagr_valid, "sale"].div(sale_lag3.loc[cagr_valid]).pow(1.0 / 3.0)
        - 1.0
    )
    margin_vol = group["gross_margin"].rolling(3, min_periods=3).std(ddof=0)
    margin_vol = margin_vol.reset_index(level=0, drop=True).sort_index()
    frame["gross_margin_volatility_3y"] = margin_vol.where(consecutive2)

    xrd_observed = frame["xrd"].where(frame["xrd"].ge(0.0))
    xrd_zero = xrd_observed.fillna(0.0)
    frame["rd_intensity_observed"] = _safe_divide(xrd_observed, frame["sale"])
    frame["rd_intensity_zero"] = _safe_divide(xrd_zero, frame["sale"])
    frame["rd_missing_indicator"] = frame["xrd"].isna().astype("int8")

    depreciation = config.innovation.stock_depreciation_rate
    stock = np.zeros(len(frame), dtype="float64")
    for _, indexes in frame.groupby("gvkey", sort=False, observed=True).groups.items():
        prior_stock = 0.0
        prior_year: int | None = None
        for index in indexes:
            year = int(frame.at[index, "fyear"])
            gap = 1 if prior_year is None else max(1, year - prior_year)
            prior_stock = float(xrd_zero.iat[index]) + prior_stock * (
                (1.0 - depreciation) ** gap
            )
            stock[index] = prior_stock
            prior_year = year
    frame["capitalized_rnd_stock_assets"] = _safe_divide(
        pd.Series(stock, index=frame.index), frame["at"]
    )
    positive_rnd = xrd_zero.gt(0.0).astype(float)
    persistence = positive_rnd.groupby(frame["gvkey"], sort=False).rolling(
        config.innovation.persistence_years,
        min_periods=config.innovation.persistence_years,
    ).mean()
    persistence = persistence.reset_index(level=0, drop=True).sort_index()
    frame["rnd_positive_share_3y"] = persistence.where(consecutive2)

    prior_sale = group["sale"].shift(1)
    prior_sga = group["xsga"].shift(1)
    prior_assets = group["at"].shift(1)
    prior_oiadp = group["oiadp"].shift(1)
    invested_capital = frame["dltt"] + frame["dlc"] + frame["ceq"] - frame["che"]
    prior_invested = invested_capital.groupby(frame["gvkey"], sort=False).shift(1)

    frame["gross_profit_lag_sga"] = _safe_divide(gross_profit, prior_sga).where(
        consecutive1
    )
    frame["revenue_change_lag_sga"] = _safe_divide(
        frame["sale"] - prior_sale, prior_sga
    ).where(consecutive1)
    frame["advertising_intensity"] = _safe_divide(frame["xad"], frame["sale"])
    frame["sga_intensity"] = _safe_divide(frame["xsga"], frame["sale"])
    frame["operating_margin"] = _safe_divide(frame["oiadp"], frame["sale"])
    nopat = frame["oiadp"] * (1.0 - config.innovation.fixed_nopat_tax_rate)
    frame["cash_adjusted_roic"] = _safe_divide(nopat, prior_invested).where(
        consecutive1
    )
    frame["gross_profitability_lag_assets"] = _safe_divide(
        gross_profit, prior_assets
    ).where(consecutive1)
    frame["cash_conversion_lag_assets"] = _safe_divide(
        frame["ib"] + frame["dp"] - frame["capx"], prior_assets
    ).where(consecutive1)
    frame["asset_turnover"] = _safe_divide(frame["sale"], prior_assets).where(
        consecutive1
    )
    frame["asset_lightness"] = 1.0 - _safe_divide(frame["ppent"], frame["at"])
    delta_sale = frame["sale"] - prior_sale
    delta_oiadp = frame["oiadp"] - prior_oiadp
    material_change = delta_sale.abs().ge(prior_sale.abs() * 0.01)
    incremental = pd.Series(np.nan, index=frame.index, dtype="float64")
    valid_incremental = consecutive1 & material_change & delta_sale.ne(0.0)
    incremental.loc[valid_incremental] = (
        delta_oiadp.loc[valid_incremental] / delta_sale.loc[valid_incremental]
    )
    frame["incremental_operating_margin"] = incremental
    frame["capex_intensity"] = _safe_divide(frame["capx"], frame["sale"])

    value_columns = [
        item.slug for item in COMPONENT_SPECS if item.slug in frame.columns
    ] + [
        "rd_intensity_observed",
        "sga_intensity",
    ]
    frame[value_columns] = frame[value_columns].replace([np.inf, -np.inf], np.nan)
    return frame


def construct_axis_scores(
    transformed: pd.DataFrame, config: HypercubeConfig
) -> pd.DataFrame:
    """Average signed component scores subject to per-axis coverage gates."""

    output = transformed.loc[
        :,
        [
            "permno",
            "gvkey",
            "datadate",
            "fyear",
            "availability_date",
            "formation_date",
            "feature_date",
            "sich",
            "sic2",
            "sic1",
            "market_cap_millions",
        ],
    ].copy()
    minimums = {item.slug: item.minimum_components for item in config.axes.components}
    for axis_slug, components in AXIS_COMPONENTS.items():
        signs = pd.Series(
            {slug: COMPONENT_BY_SLUG[slug].orientation_sign for slug in components}
        )
        for representation, prefix in (("relative", "rel_"), ("anchored", "anch_")):
            columns = [f"{prefix}{slug}" for slug in components]
            signed = transformed[columns].mul(signs.to_numpy(), axis=1)
            count = signed.notna().sum(axis=1)
            score = signed.mean(axis=1, skipna=True).where(
                count.ge(minimums[axis_slug])
            )
            output[f"{representation}_{axis_slug}"] = score
            output[f"{representation}_{axis_slug}_component_count"] = count.astype(
                "int8"
            )

    for representation, prefix in (("relative", "rel_"), ("anchored", "anch_")):
        baseline_hhi = transformed[f"{prefix}industry_sales_hhi"]
        active_count = transformed[f"{prefix}log_active_firm_count"]
        output[f"{representation}_competitive_defensibility_hhi_reversed"] = pd.concat(
            [-baseline_hhi, -active_count], axis=1
        ).mean(axis=1).where(baseline_hhi.notna() & active_count.notna())
        imputed = transformed[f"{prefix}rd_intensity_peer_imputed"]
        innovation_other = transformed[
            [
                f"{prefix}capitalized_rnd_stock_assets",
                f"{prefix}rnd_positive_share_3y",
            ]
        ]
        alternative = pd.concat([imputed, innovation_other], axis=1)
        output[f"{representation}_innovation_intensity_peer_imputed"] = (
            alternative.mean(axis=1).where(alternative.notna().sum(axis=1).ge(2))
        )
    return output


def prepare_feature_population(
    panel: pd.DataFrame, accounting_components: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Create current peer snapshots and one first-eligible row per accounting event."""

    panel_required = {
        "permno",
        "date",
        "market_cap_millions",
        "gvkey",
        "datadate",
        "fyear",
    }
    _require_columns(panel, panel_required)
    keys = ["gvkey", "datadate", "fyear"]
    component_columns = [
        item.slug for item in COMPONENT_SPECS if item.slug in accounting_components
    ] + [
        "rd_intensity_observed",
        "rd_missing_indicator",
        "sga_intensity",
        "availability_date",
        "formation_date",
        "sich",
        "sale",
    ]
    right = accounting_components[keys + component_columns].copy()
    right["fyear"] = pd.to_numeric(right["fyear"], errors="raise").astype(int)
    left = panel[
        ["permno", "date", "market_cap_millions", *keys]
    ].copy()
    left["fyear"] = pd.to_numeric(left["fyear"], errors="raise").astype(int)
    peer = left.merge(right, on=keys, how="left", validate="many_to_one")
    if peer["formation_date"].isna().any():
        raise AxisConstructionError("A P2 panel row did not map to accounting components.")
    peer = peer.sort_values(
        ["gvkey", "date", "market_cap_millions", "permno"],
        ascending=[True, True, False, True],
        kind="stable",
    )
    duplicate_firm_months = int(peer.duplicated(["gvkey", "date"]).sum())
    peer = peer.drop_duplicates(["gvkey", "date"], keep="first").reset_index(drop=True)
    peer["sich"] = pd.to_numeric(peer["sich"], errors="coerce")
    peer["sic2"] = (peer["sich"] // 100).astype("Int64")
    peer["sic1"] = (peer["sic2"] // 10).astype("Int64")

    positive_sales = peer["sale"].where(peer["sale"].gt(0.0))
    group_keys = ["date", "sic2"]
    total_sales = positive_sales.groupby(
        [peer["date"], peer["sic2"]], observed=True
    ).transform("sum")
    share_squared = positive_sales.div(total_sales).pow(2)
    peer["industry_sales_hhi"] = share_squared.groupby(
        [peer["date"], peer["sic2"]], observed=True
    ).transform("sum")
    active_count = peer.groupby(group_keys, observed=True)["gvkey"].transform("nunique")
    peer["log_active_firm_count"] = np.log1p(active_count.astype(float))

    events = peer.sort_values([*keys, "date"], kind="stable").drop_duplicates(
        keys, keep="first"
    )
    events = events.rename(columns={"date": "feature_date"}).reset_index(drop=True)
    if (events["feature_date"] < events["formation_date"]).any():
        raise AxisConstructionError("An event entered the feature population before formation.")
    diagnostics = {
        "panel_rows": int(len(panel)),
        "peer_snapshot_rows": int(len(peer)),
        "duplicate_security_rows_removed": duplicate_firm_months,
        "feature_events": int(len(events)),
        "accounting_rows_not_in_feature_population": int(
            len(accounting_components) - len(events)
        ),
    }
    return peer, events, diagnostics


def scenario_p3_dir(config: HypercubeConfig, scenario: Scenario) -> Path:
    """Resolve the phase-specific output below one P2 scenario directory."""

    configured = Path(config.paths.processed_dir)
    scenario_root = (
        configured.parent / scenario if configured.name in SCENARIOS else configured / scenario
    )
    return scenario_root / "p3"


def _p2_scenario_dir(config: HypercubeConfig, scenario: Scenario) -> Path:
    configured = Path(config.paths.processed_dir)
    return configured.parent / scenario if configured.name in SCENARIOS else configured / scenario


def _ablation_catalog(config: HypercubeConfig) -> pd.DataFrame:
    axis_flags = {item.slug: item.ablation_enabled for item in config.axes.components}
    records = [
        {
            "level": "component",
            "axis_slug": item.axis_slug,
            "variable": item.slug,
            "enabled": item.ablation_enabled,
            "included_in_baseline_composite": item.included_in_baseline_composite,
        }
        for item in COMPONENT_SPECS
    ]
    records.extend(
        {
            "level": "axis",
            "axis_slug": axis,
            "variable": axis,
            "enabled": enabled,
            "included_in_baseline_composite": True,
        }
        for axis, enabled in axis_flags.items()
    )
    records.extend(
        (
            {
                "level": "sensitivity",
                "axis_slug": "competitive_defensibility",
                "variable": "competitive_defensibility_hhi_reversed",
                "enabled": True,
                "included_in_baseline_composite": False,
            },
            {
                "level": "sensitivity",
                "axis_slug": "innovation_intensity",
                "variable": "innovation_intensity_peer_imputed",
                "enabled": True,
                "included_in_baseline_composite": False,
            },
        )
    )
    return pd.DataFrame.from_records(records)


def _fallback_counts(frame: pd.DataFrame, columns: Sequence[str]) -> dict[str, dict[str, int]]:
    return {
        component: {
            str(key): int(value)
            for key, value in frame[f"rel_level_{component}"].value_counts().items()
        }
        for component in columns
    }


def _output_inventory(directory: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.name == "p3_manifest.json":
            continue
        record: dict[str, object] = {
            "name": path.name,
            "bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
        if path.suffix == ".parquet":
            record["rows"] = int(pq.ParquetFile(path).metadata.num_rows)
        records.append(record)
    return records


def _write_p3_frames(
    directory: Path,
    components: pd.DataFrame,
    axes: pd.DataFrame,
    missingness: pd.DataFrame,
    correlations: pd.DataFrame,
    axis_correlations: pd.DataFrame,
    vif: pd.DataFrame,
    ablations: pd.DataFrame,
) -> None:
    components.to_parquet(
        directory / "component_features.parquet", index=False, compression="zstd"
    )
    axes.to_parquet(directory / "axis_scores.parquet", index=False, compression="zstd")
    missingness.to_csv(directory / "component_missingness.csv", index=False, lineterminator="\n")
    correlations.to_csv(
        directory / "component_correlations.csv", index=False, lineterminator="\n"
    )
    axis_correlations.to_csv(
        directory / "axis_correlations.csv", index=False, lineterminator="\n"
    )
    vif.to_csv(directory / "component_vif.csv", index=False, lineterminator="\n")
    ablations.to_csv(directory / "ablation_catalog.csv", index=False, lineterminator="\n")


def build_axis_bundle(
    p2_dir: Path,
    output_dir: Path,
    config: HypercubeConfig,
    *,
    scenario: str,
) -> dict[str, object]:
    """Build and atomically publish one P3 component and axis bundle."""

    if config.project.phase != "P3":
        raise AxisConstructionError("Axis construction requires a P3 config.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite completed P3 output: {output_dir}")
    accounting_path = p2_dir / "accounting_availability.parquet"
    panel_path = p2_dir / "firm_month_panel.parquet"
    if not accounting_path.is_file() or not panel_path.is_file():
        raise FileNotFoundError(f"Missing P2 inputs below {p2_dir}")
    accounting = pd.read_parquet(accounting_path)
    panel = pd.read_parquet(panel_path)
    accounting_components = construct_accounting_components(accounting, config)
    peer, events, population = prepare_feature_population(panel, accounting_components)

    imputed, impute_level, impute_count = contemporaneous_peer_median_impute(
        peer, "rd_intensity_observed", config
    )
    peer["rd_intensity_peer_imputed"] = imputed
    peer["rd_impute_level"] = impute_level
    peer["rd_impute_peer_count"] = impute_count.astype("Int32")
    event_imputation = peer[
        [
            "gvkey",
            "datadate",
            "fyear",
            "date",
            "rd_intensity_peer_imputed",
            "rd_impute_level",
            "rd_impute_peer_count",
        ]
    ].rename(columns={"date": "feature_date"})
    events = events.drop(
        columns=[
            column
            for column in (
                "rd_intensity_peer_imputed",
                "rd_impute_level",
                "rd_impute_peer_count",
            )
            if column in events
        ]
    ).merge(
        event_imputation,
        on=["gvkey", "datadate", "fyear", "feature_date"],
        how="left",
        validate="one_to_one",
    )

    primary_components = [item.slug for item in COMPONENT_SPECS]
    transformed_components = [
        *primary_components,
        "rd_intensity_observed",
        "rd_intensity_peer_imputed",
        "rd_missing_indicator",
        "sga_intensity",
    ]
    peer_scopes = {
        item.slug: item.peer_scope for item in COMPONENT_SPECS
    } | {
        "rd_intensity_observed": "industry",
        "rd_intensity_peer_imputed": "industry",
        "rd_missing_indicator": "industry",
        "sga_intensity": "industry",
    }
    relative = relative_standardize_events(
        peer, events, transformed_components, peer_scopes, config
    )
    transformed = anchored_standardize_events(
        relative, transformed_components, config
    )
    axes = construct_axis_scores(transformed, config)

    raw_columns = [*primary_components, *DIAGNOSTIC_COMPONENTS]
    relative_columns = [f"rel_{item}" for item in transformed_components]
    anchored_columns = [f"anch_{item}" for item in transformed_components]
    axis_score_columns = [
        column
        for column in axes.columns
        if column.startswith("relative_") or column.startswith("anchored_")
    ]
    missingness = pd.concat(
        [
            missingness_table(transformed, raw_columns, section="raw_components"),
            missingness_table(
                transformed, relative_columns, section="relative_components"
            ),
            missingness_table(
                transformed, anchored_columns, section="anchored_components"
            ),
            missingness_table(axes, axis_score_columns, section="axis_scores"),
        ],
        ignore_index=True,
    )
    correlations = pd.concat(
        [
            correlation_long(
                transformed, primary_components, section="raw_components"
            ),
            correlation_long(
                transformed, relative_columns[: len(primary_components)], section="relative_components"
            ),
            correlation_long(
                transformed, anchored_columns[: len(primary_components)], section="anchored_components"
            ),
        ],
        ignore_index=True,
    )
    relative_axes = [
        f"relative_{item.slug}" for item in config.axes.components
    ]
    anchored_axes = [
        f"anchored_{item.slug}" for item in config.axes.components
    ]
    axis_correlations = pd.concat(
        [
            correlation_long(axes, relative_axes, section="relative_axes"),
            correlation_long(axes, anchored_axes, section="anchored_axes"),
        ],
        ignore_index=True,
    )
    unit_components = AXIS_COMPONENTS["unit_economics_profit_quality"]
    vif = pd.concat(
        [
            vif_table(
                transformed,
                [f"rel_{item}" for item in unit_components],
                section="relative_unit_economics",
            ),
            vif_table(
                transformed,
                [f"anch_{item}" for item in unit_components],
                section="anchored_unit_economics",
            ),
        ],
        ignore_index=True,
    )
    ablations = _ablation_catalog(config)
    numeric_feature_columns = [*raw_columns, *relative_columns, *anchored_columns]
    nonfinite_components = count_nonfinite(transformed, numeric_feature_columns)
    nonfinite_axes = count_nonfinite(axes, axis_score_columns)
    diagnostics: dict[str, object] = {
        "status": "PASS" if nonfinite_components + nonfinite_axes == 0 else "FAIL",
        "phase": "P3",
        "scenario": scenario,
        "models_fitted": [],
        "population": population,
        "rows": {
            "accounting_input": int(len(accounting)),
            "panel_input": int(len(panel)),
            "component_features": int(len(transformed)),
            "axis_scores": int(len(axes)),
        },
        "nonfinite_component_values": nonfinite_components,
        "nonfinite_axis_values": nonfinite_axes,
        "pre_availability_violations": int(
            (transformed["formation_date"] <= transformed["availability_date"]).sum()
            + (transformed["feature_date"] < transformed["formation_date"]).sum()
        ),
        "duplicate_event_keys": int(
            transformed.duplicated(["gvkey", "datadate", "fyear"]).sum()
        ),
        "relative_fallback_counts": _fallback_counts(
            transformed, transformed_components
        ),
        "rd_missing_treatment_counts": {
            str(key): int(value)
            for key, value in transformed["rd_impute_level"].value_counts().items()
        },
        "synthetic_truth_read": False,
    }
    if diagnostics["status"] != "PASS" or diagnostics["pre_availability_violations"]:
        raise AxisConstructionError(f"P3 construction gate failed: {diagnostics}")

    identity = [
        "permno",
        "gvkey",
        "datadate",
        "fyear",
        "availability_date",
        "formation_date",
        "feature_date",
        "sich",
        "sic2",
        "sic1",
        "market_cap_millions",
        "anchor_reference_end_year",
        "rd_impute_level",
        "rd_impute_peer_count",
    ]
    audit_columns = [
        column
        for component in transformed_components
        for column in (
            f"rel_n_{component}",
            f"rel_level_{component}",
            f"anch_n_{component}",
        )
    ]
    component_output = transformed[
        [*identity, *raw_columns, *relative_columns, *anchored_columns, *audit_columns]
    ].copy()

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.p3-", dir=output_dir.parent))
    try:
        _write_p3_frames(
            staging,
            component_output,
            axes,
            missingness,
            correlations,
            axis_correlations,
            vif,
            ablations,
        )
        transformation_metadata = {
            "phase": "P3",
            "scenario": scenario,
            "component_catalog": component_catalog(),
            "axis_components": {
                key: list(value) for key, value in AXIS_COMPONENTS.items()
            },
            "standardization": config.standardization.model_dump(mode="json"),
            "innovation": config.innovation.model_dump(mode="json"),
            "relative_reference": "current eligible firm snapshot at feature_date",
            "anchored_reference": "all feature events from strictly prior calendar years",
            "competitive_baseline": "higher HHI is treated as defensibility; reversed-HHI sensitivity retained",
            "rd_variants": [
                "zero_with_indicator",
                "observed_only",
                "contemporaneous_peer_median",
            ],
            "synthetic_truth_read": False,
            "models_fitted": [],
        }
        atomic_write_json(staging / "transformation_metadata.json", transformation_metadata)
        atomic_write_json(staging / "p3_diagnostics.json", diagnostics)
        atomic_write_json(staging / "resolved_config.json", config.model_dump(mode="json"))
        manifest: dict[str, object] = {
            "schema_version": 1,
            "phase": "P3",
            "scenario": scenario,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "p2_dir": str(p2_dir),
            "output_dir": str(output_dir),
            "seed": config.project.seed,
            "input_files": [
                {
                    "name": accounting_path.name,
                    "bytes": accounting_path.stat().st_size,
                    "sha256": sha256_file(accounting_path),
                    "rows": int(pq.ParquetFile(accounting_path).metadata.num_rows),
                },
                {
                    "name": panel_path.name,
                    "bytes": panel_path.stat().st_size,
                    "sha256": sha256_file(panel_path),
                    "rows": int(pq.ParquetFile(panel_path).metadata.num_rows),
                },
            ],
            "files": _output_inventory(staging),
            "models_fitted": [],
        }
        atomic_write_json(staging / "p3_manifest.json", manifest)
        if output_dir.exists():
            output_dir.rmdir()
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return diagnostics


def validate_p3_directory(
    output_dir: Path, config: HypercubeConfig
) -> dict[str, object]:
    """Independently reopen and validate one saved P3 feature bundle."""

    required = (
        "component_features.parquet",
        "axis_scores.parquet",
        "component_missingness.csv",
        "component_correlations.csv",
        "axis_correlations.csv",
        "component_vif.csv",
        "ablation_catalog.csv",
        "transformation_metadata.json",
        "p3_diagnostics.json",
        "resolved_config.json",
        "p3_manifest.json",
    )
    missing = [name for name in required if not (output_dir / name).is_file()]
    if missing:
        return {
            "status": "FAIL",
            "errors": [f"Missing P3 outputs: {missing}"],
            "warnings": [],
            "models_fitted": [],
        }
    components = pd.read_parquet(output_dir / "component_features.parquet")
    axes = pd.read_parquet(output_dir / "axis_scores.parquet")
    metadata = json.loads(
        (output_dir / "transformation_metadata.json").read_text(encoding="utf-8")
    )
    manifest = json.loads((output_dir / "p3_manifest.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    keys = ["gvkey", "datadate", "fyear"]
    if components.duplicated(keys).any() or axes.duplicated(keys).any():
        errors.append("Duplicate accounting-event keys in P3 outputs.")
    if len(components) != len(axes):
        errors.append("Component and axis row counts differ.")
    if (components["formation_date"] <= components["availability_date"]).any():
        errors.append("A P3 feature forms before public availability.")
    if (components["feature_date"] < components["formation_date"]).any():
        errors.append("A P3 feature enters before its eligible formation date.")
    if not (
        components["anchor_reference_end_year"]
        < pd.to_datetime(components["feature_date"]).dt.year
    ).all():
        errors.append("Anchored reference includes the current or a future year.")

    primary = [item.slug for item in COMPONENT_SPECS]
    transformed = [*primary, "rd_intensity_observed", "rd_intensity_peer_imputed", "rd_missing_indicator", "sga_intensity"]
    feature_numeric = [
        *[item for item in primary if item in components],
        *[f"rel_{item}" for item in transformed],
        *[f"anch_{item}" for item in transformed],
    ]
    if count_nonfinite(components, feature_numeric):
        errors.append("Component table contains positive or negative infinity.")
    axis_score_columns = [
        column
        for column in axes
        if (column.startswith("relative_") or column.startswith("anchored_"))
        and not column.endswith("_component_count")
    ]
    if count_nonfinite(axes, axis_score_columns):
        errors.append("Axis table contains positive or negative infinity.")
    clip = config.standardization.score_clip
    component_scores = components[
        [f"rel_{item}" for item in transformed] + [f"anch_{item}" for item in transformed]
    ]
    if component_scores.abs().max(skipna=True).max(skipna=True) > clip + 1e-12:
        errors.append("A standardized component exceeds the configured score clip.")
    for component in transformed:
        relative_present = components[f"rel_{component}"].notna()
        if not components.loc[relative_present, f"rel_n_{component}"].ge(
            config.standardization.minimum_peer_group_size
        ).all():
            errors.append(f"Relative reference count is too small: {component}.")
        anchored_present = components[f"anch_{component}"].notna()
        if not components.loc[anchored_present, f"anch_n_{component}"].ge(
            config.standardization.minimum_anchor_observations
        ).all():
            errors.append(f"Anchored reference count is too small: {component}.")
    recomputed = construct_axis_scores(components, config)
    for column in axis_score_columns:
        if column not in recomputed:
            errors.append(f"Unexpected axis score column: {column}.")
            continue
        if not np.allclose(
            axes[column].to_numpy(dtype=float, na_value=np.nan),
            recomputed[column].to_numpy(dtype=float, na_value=np.nan),
            equal_nan=True,
            rtol=1e-12,
            atol=1e-12,
        ):
            errors.append(f"Saved axis score does not recompute: {column}.")
    if metadata.get("synthetic_truth_read") is not False:
        errors.append("P3 metadata does not certify exclusion of synthetic truth.")
    inventory = {item["name"]: item for item in manifest.get("files", [])}
    for name, record in inventory.items():
        path = output_dir / name
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            errors.append(f"Manifest hash mismatch: {name}.")
        if path.suffix == ".parquet" and int(
            pq.ParquetFile(path).metadata.num_rows
        ) != int(record.get("rows", -1)):
            errors.append(f"Manifest row-count mismatch: {name}.")
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "warnings": [],
        "rows": {
            "component_features": int(len(components)),
            "axis_scores": int(len(axes)),
        },
        "nonmissing_axes": {
            column: int(axes[column].notna().sum()) for column in axis_score_columns
        },
        "models_fitted": [],
    }
