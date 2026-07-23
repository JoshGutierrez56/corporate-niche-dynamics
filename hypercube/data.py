"""P1 synthetic raw-data generation and schema validation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Literal, Mapping

import numpy as np
import pandas as pd

from hypercube.config import HypercubeConfig


Scenario = Literal["null_alpha", "migration_alpha", "regime_shift"]
SCENARIOS: tuple[Scenario, ...] = (
    "null_alpha",
    "migration_alpha",
    "regime_shift",
)
GENERATOR_VERSION = "hypercube-synthetic-v1"


@dataclass(frozen=True)
class TableSpec:
    """Required columns, key, and dates for one raw file."""

    columns: tuple[str, ...]
    primary_key: tuple[str, ...]
    date_columns: tuple[str, ...]


TABLE_SPECS: Mapping[str, TableSpec] = {
    "funda.parquet": TableSpec(
        columns=(
            "gvkey",
            "datadate",
            "fyear",
            "fyr",
            "rdq",
            "sich",
            "naics",
            "revt",
            "sale",
            "cogs",
            "oibdp",
            "oiadp",
            "xrd",
            "xsga",
            "xad",
            "ib",
            "ni",
            "dp",
            "at",
            "ppent",
            "capx",
            "dltt",
            "dlc",
            "ceq",
            "che",
            "act",
            "lct",
            "csho",
            "prcc_f",
        ),
        primary_key=("gvkey", "datadate", "fyear"),
        date_columns=("datadate", "rdq"),
    ),
    "crsp_monthly.parquet": TableSpec(
        columns=(
            "permno",
            "date",
            "ret",
            "retx",
            "prc",
            "shrout",
            "vol",
            "bid",
            "ask",
            "exchcd",
            "shrcd",
        ),
        primary_key=("permno", "date"),
        date_columns=("date",),
    ),
    "crsp_delist.parquet": TableSpec(
        columns=("permno", "dlstdt", "dlstcd", "dlret", "exit_category"),
        primary_key=("permno", "dlstdt"),
        date_columns=("dlstdt",),
    ),
    "ccm_link.parquet": TableSpec(
        columns=("gvkey", "lpermno", "linkdt", "linkenddt", "linktype", "linkprim"),
        primary_key=("gvkey", "lpermno", "linkdt"),
        date_columns=("linkdt", "linkenddt"),
    ),
    "factor_returns.parquet": TableSpec(
        columns=("date", "mkt_excess", "smb", "hml", "rmw", "cma", "mom", "rf"),
        primary_key=("date",),
        date_columns=("date",),
    ),
}

OPTIONAL_TABLE_SPECS: Mapping[str, TableSpec] = {
    "sec_filing_dates.parquet": TableSpec(
        columns=("gvkey", "datadate", "fyear", "filing_timestamp"),
        primary_key=("gvkey", "datadate", "fyear"),
        date_columns=("datadate", "filing_timestamp"),
    ),
    "synthetic_truth.parquet": TableSpec(
        columns=(
            "gvkey",
            "datadate",
            "fyear",
            "latent_axis_1",
            "latent_axis_2",
            "latent_axis_3",
            "latent_axis_4",
            "latent_axis_5",
            "latent_axis_6",
            "true_viability",
            "migration_surprise",
            "availability_date",
            "availability_source",
            "injected_return_alpha",
            "regime_id",
            "exit_category",
        ),
        primary_key=("gvkey", "datadate", "fyear"),
        date_columns=("datadate", "availability_date"),
    ),
}


class RawValidationError(ValueError):
    """Raised when one or more P1 raw-data contracts fail."""


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON to a same-volume temporary file and atomically replace."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _write_parquet(frame: pd.DataFrame, path: Path) -> None:
    """Write a compressed parquet atomically inside a staging directory."""

    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def scenario_output_dir(config: HypercubeConfig, scenario: Scenario) -> Path:
    """Resolve sibling synthetic scenario directories from one config."""

    configured = Path(config.paths.raw_dir)
    if configured.name in SCENARIOS:
        return configured.parent / scenario
    return configured / scenario


def _factor_frame(
    rng: np.random.Generator, start_year: int, end_year: int
) -> pd.DataFrame:
    months = pd.date_range(
        f"{start_year}-01-31", f"{end_year}-12-31", freq="ME"
    )
    means = np.array([0.0050, 0.0015, 0.0018, 0.0012, 0.0010, 0.0025])
    vols = np.array([0.043, 0.025, 0.027, 0.021, 0.020, 0.033])
    correlation = np.full((6, 6), 0.08)
    np.fill_diagonal(correlation, 1.0)
    covariance = np.outer(vols, vols) * correlation
    draws = rng.multivariate_normal(means, covariance, size=len(months))
    frame = pd.DataFrame(
        draws,
        columns=["mkt_excess", "smb", "hml", "rmw", "cma", "mom"],
    )
    frame.insert(0, "date", months)
    frame["rf"] = np.maximum(0.0, rng.normal(0.0020, 0.0008, len(months)))
    return frame


def _empty_columns(names: tuple[str, ...]) -> dict[str, list[Any]]:
    return {name: [] for name in names}


def _table_summary(path: Path, frame: pd.DataFrame, dates: tuple[str, ...]) -> dict[str, Any]:
    date_ranges: dict[str, dict[str, str] | None] = {}
    for column in dates:
        values = pd.to_datetime(frame[column], errors="coerce").dropna()
        date_ranges[column] = (
            {"min": values.min().isoformat(), "max": values.max().isoformat()}
            if not values.empty
            else None
        )
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "date_ranges": date_ranges,
    }


def generate_synthetic_bundle(
    config: HypercubeConfig,
    scenario: Scenario,
    output_dir: Path,
    *,
    n_firms: int | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Generate one deterministic WRDS-shaped synthetic scenario."""

    if scenario not in SCENARIOS:
        raise ValueError(f"Unsupported scenario: {scenario}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output_dir}")

    settings = config.synthetic
    firm_count = n_firms or settings.n_firms
    first_year = start_year or settings.start_year
    last_year = end_year or settings.end_year
    run_seed = config.project.seed if seed is None else seed
    if firm_count < 20:
        raise ValueError("At least 20 firms are required for exit/missingness coverage.")
    if last_year - first_year + 1 < settings.minimum_history_years:
        raise ValueError("Synthetic test window is too short.")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.", dir=output_dir.parent)
    )
    rng = np.random.default_rng(run_seed)
    try:
        factors = _factor_frame(rng, first_year, last_year)
        factor_lookup = factors.set_index(factors["date"].dt.to_period("M"))

        funda_columns = TABLE_SPECS["funda.parquet"].columns
        funda_data = _empty_columns(funda_columns)
        truth_data = _empty_columns(OPTIONAL_TABLE_SPECS["synthetic_truth.parquet"].columns)
        link_data = _empty_columns(TABLE_SPECS["ccm_link.parquet"].columns)
        delist_data = _empty_columns(TABLE_SPECS["crsp_delist.parquet"].columns)
        crsp_data = _empty_columns(TABLE_SPECS["crsp_monthly.parquet"].columns)

        years = np.arange(first_year, last_year + 1)
        industry_count = settings.n_industries
        axis_correlation = np.full((6, 6), 0.28)
        np.fill_diagonal(axis_correlation, 1.0)
        axis_cholesky = np.linalg.cholesky(axis_correlation)
        industry_shocks = np.zeros((len(years), industry_count, 6))
        for year_index in range(len(years)):
            innovations = rng.normal(size=(industry_count, 6)) @ axis_cholesky.T
            if year_index == 0:
                industry_shocks[year_index] = innovations * 0.45
            else:
                industry_shocks[year_index] = (
                    0.88 * industry_shocks[year_index - 1] + 0.18 * innovations
                )

        pre_weights = np.array([0.20, 0.14, 0.12, 0.16, 0.24, 0.14])
        post_weights = (
            np.array([0.12, 0.20, 0.23, 0.10, 0.17, 0.18])
            if scenario == "regime_shift"
            else pre_weights
        )

        firm_state_rows: dict[str, list[dict[str, Any]]] = {}
        firm_month_rules: dict[str, dict[str, Any]] = {}
        exit_counts = {name: 0 for name in ("performance_failure", "merger", "voluntary_administrative", "other_unknown", "censored")}

        latest_entry = max(first_year, last_year - settings.minimum_history_years + 1)
        entry_years = rng.integers(first_year, latest_entry + 1, size=firm_count)
        entry_years[:4] = first_year
        industries = rng.integers(0, industry_count, size=firm_count)

        for firm_index in range(firm_count):
            gvkey = f"{100001 + firm_index:06d}"
            base_permno = 10000 + firm_index
            replacement_permno = 500000 + firm_index
            entry_year = int(entry_years[firm_index])
            industry = int(industries[firm_index])
            state = rng.normal(size=6) @ axis_cholesky.T
            log_assets = rng.normal(5.0, 1.0)
            prior_viability: float | None = None
            prior_sales: float | None = None
            annual_rows: list[dict[str, Any]] = []
            exit_category = "censored"
            exit_date: pd.Timestamp | None = None

            potential_switch_year = entry_year + max(
                4, (last_year - entry_year + 1) // 2
            )
            has_permno_change = bool(
                rng.random() < 0.05 and potential_switch_year < last_year
            )
            switch_date = (
                pd.Timestamp(potential_switch_year, 1, 1)
                if has_permno_change
                else None
            )

            for fyear in range(entry_year, last_year + 1):
                year_index = fyear - first_year
                innovation = rng.normal(size=6) @ axis_cholesky.T
                state = (
                    settings.annual_state_persistence * state
                    + 0.15 * industry_shocks[year_index, industry]
                    + 0.28 * innovation
                )
                if scenario == "regime_shift" and fyear >= settings.regime_shift_year:
                    state = state + np.array([0.03, -0.02, 0.05, 0.0, -0.03, 0.04])
                weights = post_weights if fyear >= settings.regime_shift_year else pre_weights
                true_viability = float(state @ weights)
                migration_surprise = (
                    0.0
                    if prior_viability is None
                    else true_viability
                    - settings.annual_state_persistence * prior_viability
                )

                log_assets += float(0.035 + 0.018 * state[5] + rng.normal(0.0, 0.10))
                assets = float(max(5.0, math.exp(log_assets)))
                asset_turnover = float(np.clip(1.05 + 0.16 * state[5], 0.25, 3.0))
                sales = float(max(1.0, assets * asset_turnover * math.exp(rng.normal(0.0, 0.08))))
                gross_margin = float(np.clip(0.34 + 0.07 * state[0] + 0.03 * state[4], 0.05, 0.82))
                operating_margin = float(np.clip(0.10 + 0.045 * state[4] + 0.02 * state[3], -0.35, 0.45))
                ppent = float(assets * np.clip(0.30 - 0.06 * state[5], 0.03, 0.75))
                depreciation = float(ppent * np.clip(0.075 + rng.normal(0.0, 0.01), 0.03, 0.15))
                oiadp = float(sales * operating_margin)
                oibdp = float(oiadp + depreciation)
                xrd = float(max(0.0, sales * np.clip(0.035 + 0.022 * state[2], 0.0, 0.22)))
                xsga = float(max(0.5, sales * np.clip(0.20 - 0.025 * state[3], 0.04, 0.55)))
                xad = float(max(0.0, sales * np.clip(0.012 + 0.006 * state[3], 0.0, 0.08)))
                debt_ratio = float(np.clip(0.32 - 0.05 * true_viability + rng.normal(0.0, 0.04), 0.0, 0.85))
                dltt = float(assets * debt_ratio * 0.86)
                dlc = float(assets * debt_ratio * 0.14)
                cash = float(assets * np.clip(0.13 + 0.025 * state[2], 0.01, 0.55))
                current_assets = float(assets * np.clip(0.38 + rng.normal(0.0, 0.04), 0.12, 0.75))
                current_liabilities = float(assets * np.clip(0.22 + rng.normal(0.0, 0.04), 0.04, 0.60))
                equity = float(max(0.1, assets - dltt - dlc - current_liabilities * 0.35))
                pretax_like = oiadp - 0.045 * (dltt + dlc)
                net_income = float(pretax_like * (0.78 if pretax_like >= 0 else 1.0))
                shares = float(max(1.0, math.exp(log_assets - 2.8 + rng.normal(0.0, 0.12))))
                annual_price = float(max(0.5, equity / shares * math.exp(rng.normal(0.0, 0.18))))
                datadate = pd.Timestamp(fyear, 12, 31)
                rdq_delay = int(rng.integers(45, 121))
                rdq = datadate + pd.Timedelta(days=rdq_delay)

                row = {
                    "gvkey": gvkey,
                    "datadate": datadate,
                    "fyear": fyear,
                    "fyr": 12,
                    "rdq": rdq,
                    "sich": 2000 + industry * 50,
                    "naics": 310000 + industry * 100,
                    "revt": sales,
                    "sale": sales,
                    "cogs": sales * (1.0 - gross_margin),
                    "oibdp": oibdp,
                    "oiadp": oiadp,
                    "xrd": xrd,
                    "xsga": xsga,
                    "xad": xad,
                    "ib": net_income,
                    "ni": net_income,
                    "dp": depreciation,
                    "at": assets,
                    "ppent": ppent,
                    "capx": max(0.0, ppent * np.clip(0.10 + 0.025 * state[5], 0.01, 0.35)),
                    "dltt": dltt,
                    "dlc": dlc,
                    "ceq": equity,
                    "che": cash,
                    "act": current_assets,
                    "lct": current_liabilities,
                    "csho": shares,
                    "prcc_f": annual_price,
                }
                for column in funda_columns:
                    funda_data[column].append(row[column])

                truth_row = {
                    "gvkey": gvkey,
                    "datadate": datadate,
                    "fyear": fyear,
                    **{f"latent_axis_{i + 1}": float(state[i]) for i in range(6)},
                    "true_viability": true_viability,
                    "migration_surprise": migration_surprise,
                    "availability_date": pd.NaT,
                    "availability_source": "pending",
                    "injected_return_alpha": 0.0,
                    "regime_id": (
                        "post_shift"
                        if scenario == "regime_shift" and fyear >= settings.regime_shift_year
                        else "baseline"
                    ),
                    "exit_category": "none",
                }
                annual_rows.append(
                    {
                        "fyear": fyear,
                        "viability": true_viability,
                        "migration_surprise": migration_surprise,
                        "truth_index": len(truth_data["gvkey"]),
                    }
                )
                for column in truth_data:
                    truth_data[column].append(truth_row[column])

                forced: str | None = None
                forced_years = {
                    0: ("performance_failure", first_year + 2),
                    1: ("merger", first_year + 3),
                    2: ("voluntary_administrative", first_year + 4),
                    3: ("other_unknown", first_year + 5),
                }
                if firm_index in forced_years and fyear == forced_years[firm_index][1]:
                    forced = forced_years[firm_index][0]
                if fyear < last_year:
                    draw = rng.random()
                    performance_probability = _sigmoid(-5.0 - 0.95 * true_viability)
                    merger_probability = 0.006 + 0.002 * _sigmoid(log_assets - 5.0)
                    administrative_probability = 0.0015
                    other_probability = 0.0010
                    if forced is not None:
                        exit_category = forced
                    elif draw < performance_probability:
                        exit_category = "performance_failure"
                    elif draw < performance_probability + merger_probability:
                        exit_category = "merger"
                    elif draw < performance_probability + merger_probability + administrative_probability:
                        exit_category = "voluntary_administrative"
                    elif draw < performance_probability + merger_probability + administrative_probability + other_probability:
                        exit_category = "other_unknown"
                    if exit_category != "censored":
                        exit_month = int(rng.integers(6, 13))
                        exit_date = pd.Timestamp(fyear + 1, exit_month, 15)
                        truth_data["exit_category"][-1] = exit_category
                        break

                prior_viability = true_viability
                prior_sales = sales

            del prior_sales
            firm_state_rows[gvkey] = annual_rows
            exit_counts[exit_category] += 1
            link_end = (
                exit_date + pd.offsets.MonthEnd(0) if exit_date is not None else pd.NaT
            )
            entry_date = pd.Timestamp(entry_year, 1, 1)
            if switch_date is not None and (exit_date is None or switch_date < exit_date):
                first_link = {
                    "gvkey": gvkey,
                    "lpermno": base_permno,
                    "linkdt": entry_date,
                    "linkenddt": switch_date - pd.Timedelta(days=1),
                    "linktype": "LU",
                    "linkprim": "P",
                }
                second_link = {
                    "gvkey": gvkey,
                    "lpermno": replacement_permno,
                    "linkdt": switch_date,
                    "linkenddt": link_end,
                    "linktype": "LU",
                    "linkprim": "P",
                }
                for item in (first_link, second_link):
                    for column in link_data:
                        link_data[column].append(item[column])
            else:
                item = {
                    "gvkey": gvkey,
                    "lpermno": base_permno,
                    "linkdt": entry_date,
                    "linkenddt": link_end,
                    "linktype": "LU",
                    "linkprim": "P",
                }
                for column in link_data:
                    link_data[column].append(item[column])
            firm_month_rules[gvkey] = {
                "entry_date": entry_date,
                "exit_date": exit_date,
                "exit_category": exit_category,
                "base_permno": base_permno,
                "replacement_permno": replacement_permno,
                "switch_date": switch_date,
                "industry": industry,
                "loadings": rng.normal(
                    loc=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                    scale=np.array([0.18, 0.16, 0.16, 0.12, 0.12, 0.16]),
                ),
                "shares": float(max(500.0, rng.lognormal(9.3, 0.7))),
                "price": float(max(3.0, rng.lognormal(3.0, 0.45))),
            }

        funda = pd.DataFrame(funda_data)
        truth = pd.DataFrame(truth_data)
        missing_rdq = rng.random(len(funda)) < 0.12
        missing_xrd = rng.random(len(funda)) < (0.18 + 0.10 * (funda["sich"] % 3 == 0))
        missing_xad = rng.random(len(funda)) < 0.62
        if len(funda) >= 3:
            missing_rdq[0] = True
            missing_xrd[1] = True
            missing_xad[2] = True
        funda.loc[missing_rdq, "rdq"] = pd.NaT
        funda.loc[missing_xrd, "xrd"] = np.nan
        funda.loc[missing_xad, "xad"] = np.nan

        sec_mask = rng.random(len(funda)) < 0.84
        sec_delays = rng.integers(60, 151, size=len(funda))
        sec = funda.loc[sec_mask, ["gvkey", "datadate", "fyear"]].copy()
        sec["filing_timestamp"] = (
            sec["datadate"]
            + pd.to_timedelta(sec_delays[sec_mask], unit="D")
            + pd.Timedelta(hours=16)
        )
        sec_lookup = {
            (row.gvkey, row.fyear): row.filing_timestamp
            for row in sec.itertuples(index=False)
        }
        funda_lookup = {
            (row.gvkey, row.fyear): row.rdq for row in funda.itertuples(index=False)
        }
        alpha_by_firm_month: dict[tuple[str, pd.Period], float] = {}
        for truth_index, row in truth.iterrows():
            key = (str(row["gvkey"]), int(row["fyear"]))
            filing = sec_lookup.get(key)
            rdq = funda_lookup[key]
            if filing is not None:
                availability = pd.Timestamp(filing)
                source = "verified_sec_filing_timestamp"
            elif pd.notna(rdq):
                availability = pd.Timestamp(rdq)
                source = "verified_earnings_announcement_date"
            else:
                availability = pd.Timestamp(row["datadate"]) + pd.Timedelta(
                    days=config.availability.fallback_days
                )
                source = "datadate_plus_180_calendar_days"
            truth.at[truth_index, "availability_date"] = availability
            truth.at[truth_index, "availability_source"] = source
            if scenario == "migration_alpha":
                total_alpha = 0.0
                first_return_month = availability.to_period("M") + 2
                signal = float(row["migration_surprise"])
                for offset in range(settings.migration_decay_months):
                    contribution = (
                        settings.migration_alpha_monthly
                        * signal
                        * math.exp(-offset / 4.0)
                    )
                    period = first_return_month + offset
                    alpha_by_firm_month[(str(row["gvkey"]), period)] = (
                        alpha_by_firm_month.get((str(row["gvkey"]), period), 0.0)
                        + contribution
                    )
                    total_alpha += contribution
                truth.at[truth_index, "injected_return_alpha"] = total_alpha

        for gvkey, rule in firm_month_rules.items():
            start = pd.Timestamp(rule["entry_date"]) + pd.offsets.MonthEnd(0)
            end = (
                pd.Timestamp(rule["exit_date"]) + pd.offsets.MonthEnd(0)
                if rule["exit_date"] is not None
                else pd.Timestamp(last_year, 12, 31)
            )
            months = pd.date_range(start, end, freq="ME")
            state_rows = firm_state_rows[gvkey]
            state_by_year = {int(item["fyear"]): item for item in state_rows}
            available_years = sorted(state_by_year)
            price = float(rule["price"])
            shares = float(rule["shares"])
            prior_idiosyncratic = 0.0
            for month in months:
                period = month.to_period("M")
                state_year = min(max(month.year, available_years[0]), available_years[-1])
                annual_state = state_by_year[state_year]
                factor_row = factor_lookup.loc[period]
                loadings = np.asarray(rule["loadings"], dtype=float)
                systematic = float(
                    loadings
                    @ factor_row[["mkt_excess", "smb", "hml", "rmw", "cma", "mom"]].to_numpy(dtype=float)
                )
                idiosyncratic = float(
                    0.10 * prior_idiosyncratic + rng.normal(0.0, 0.075)
                )
                injected = alpha_by_firm_month.get((gvkey, period), 0.0)
                monthly_return = float(
                    np.clip(factor_row["rf"] + systematic + idiosyncratic + injected, -0.95, 1.50)
                )
                retx = float(np.clip(monthly_return - max(0.0, rng.normal(0.0008, 0.0005)), -0.95, 1.50))
                price = float(max(0.25, price * (1.0 + monthly_return)))
                shares = float(max(100.0, shares * math.exp(rng.normal(0.0005, 0.004))))
                size_scale = max(0.2, math.log1p(price * shares) / 14.0)
                spread_fraction = float(np.clip(0.008 / size_scale + abs(rng.normal(0.0, 0.0015)), 0.0005, 0.08))
                bid = price * (1.0 - spread_fraction / 2.0)
                ask = price * (1.0 + spread_fraction / 2.0)
                permno = (
                    int(rule["replacement_permno"])
                    if rule["switch_date"] is not None and month >= rule["switch_date"]
                    else int(rule["base_permno"])
                )
                values = {
                    "permno": permno,
                    "date": month,
                    "ret": monthly_return,
                    "retx": retx,
                    "prc": price,
                    "shrout": shares,
                    "vol": float(max(1.0, shares * rng.lognormal(-2.2, 0.55))),
                    "bid": bid,
                    "ask": ask,
                    "exchcd": 1 + (int(rule["industry"]) % 3),
                    "shrcd": 10 + (int(rule["industry"]) % 2),
                }
                for column in crsp_data:
                    crsp_data[column].append(values[column])
                prior_idiosyncratic = idiosyncratic

            if rule["exit_date"] is not None:
                category = str(rule["exit_category"])
                code_and_return = {
                    "performance_failure": (574, float(rng.uniform(-1.0, -0.35))),
                    "merger": (241, float(rng.uniform(-0.05, 0.35))),
                    "voluntary_administrative": (552, float(rng.uniform(-0.45, 0.05))),
                    "other_unknown": (100, float(rng.uniform(-0.30, 0.15))),
                }[category]
                delist_values = {
                    "permno": int(
                        rule["replacement_permno"]
                        if rule["switch_date"] is not None
                        and rule["exit_date"] >= rule["switch_date"]
                        else rule["base_permno"]
                    ),
                    "dlstdt": pd.Timestamp(rule["exit_date"]),
                    "dlstcd": code_and_return[0],
                    "dlret": code_and_return[1],
                    "exit_category": category,
                }
                for column in delist_data:
                    delist_data[column].append(delist_values[column])

        crsp = pd.DataFrame(crsp_data)
        quote_missing = rng.random(len(crsp)) < np.where(crsp["date"].dt.year < 1993, 0.15, 0.025)
        if len(crsp):
            quote_missing[0] = True
        crsp.loc[quote_missing, ["bid", "ask"]] = np.nan
        ccm = pd.DataFrame(link_data)
        delist = pd.DataFrame(delist_data)

        for frame, string_columns in (
            (funda, ("gvkey",)),
            (sec, ("gvkey",)),
            (ccm, ("gvkey", "linktype", "linkprim")),
            (delist, ("exit_category",)),
            (truth, ("gvkey", "availability_source", "regime_id", "exit_category")),
        ):
            for column in string_columns:
                frame[column] = frame[column].astype("string")

        frames = {
            "funda.parquet": funda,
            "sec_filing_dates.parquet": sec,
            "crsp_monthly.parquet": crsp,
            "crsp_delist.parquet": delist,
            "ccm_link.parquet": ccm,
            "factor_returns.parquet": factors,
            "synthetic_truth.parquet": truth,
        }
        for filename, frame in frames.items():
            _write_parquet(frame, staging / filename)

        validation = validate_raw_directory(staging, raise_on_error=True)
        injection = {
            "survival_signal": "lower true_viability increases performance-failure hazard",
            "migration_alpha_monthly": (
                settings.migration_alpha_monthly
                if scenario == "migration_alpha"
                else 0.0
            ),
            "migration_decay_months": (
                settings.migration_decay_months if scenario == "migration_alpha" else 0
            ),
            "regime_shift_year": (
                settings.regime_shift_year if scenario == "regime_shift" else None
            ),
            "viability_weights_pre": pre_weights.tolist(),
            "viability_weights_post": post_weights.tolist(),
        }
        metadata: dict[str, Any] = {
            "schema_version": 1,
            "generator_version": GENERATOR_VERSION,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "scenario": scenario,
            "seed": run_seed,
            "n_firms_requested": firm_count,
            "sample_years": {"start": first_year, "end": last_year},
            "n_industries": industry_count,
            "injection": injection,
            "expected_signs": {
                "true_viability_to_performance_failure": "negative",
                "migration_surprise_to_future_return": (
                    "positive" if scenario == "migration_alpha" else "zero"
                ),
                "dated_viability_surface_change": scenario == "regime_shift",
            },
            "exit_counts": exit_counts,
            "missingness": {
                "rdq": float(funda["rdq"].isna().mean()),
                "xrd": float(funda["xrd"].isna().mean()),
                "xad": float(funda["xad"].isna().mean()),
                "bid_or_ask": float(crsp[["bid", "ask"]].isna().any(axis=1).mean()),
            },
            "validation_status": validation["status"],
            "model_fitted": False,
        }
        atomic_write_json(staging / "synthetic_scenario_metadata.json", metadata)

        manifest_tables: dict[str, Any] = {}
        for filename, frame in frames.items():
            spec = TABLE_SPECS.get(filename) or OPTIONAL_TABLE_SPECS[filename]
            manifest_tables[filename] = _table_summary(
                staging / filename, frame, spec.date_columns
            )
        manifest = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "generator_version": GENERATOR_VERSION,
            "scenario": scenario,
            "seed": run_seed,
            "tables": manifest_tables,
            "metadata_file": "synthetic_scenario_metadata.json",
            "validation_status": validation["status"],
            "models_fitted": [],
        }
        atomic_write_json(staging / "data_manifest.json", manifest)
        os.replace(staging, output_dir)
        metadata["output_dir"] = str(output_dir)
        metadata["row_counts"] = {
            filename: int(len(frame)) for filename, frame in frames.items()
        }
        metadata["manifest_sha256"] = sha256_file(output_dir / "data_manifest.json")
        return metadata
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _read_tables(raw_dir: Path) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for filename in TABLE_SPECS:
        path = raw_dir / filename
        if not path.is_file():
            raise RawValidationError(f"Missing required raw file: {path}")
        frames[filename] = pd.read_parquet(path)
    for filename in OPTIONAL_TABLE_SPECS:
        path = raw_dir / filename
        if path.is_file():
            frames[filename] = pd.read_parquet(path)
    return frames


def validate_raw_directory(
    raw_dir: Path, *, raise_on_error: bool = False
) -> dict[str, Any]:
    """Validate schemas, keys, dates, and cross-file P1 relationships."""

    errors: list[str] = []
    warnings: list[str] = []
    try:
        frames = _read_tables(raw_dir)
    except RawValidationError as error:
        if raise_on_error:
            raise
        return {"status": "FAIL", "raw_dir": str(raw_dir), "errors": [str(error)]}

    table_reports: dict[str, Any] = {}
    all_specs = {**TABLE_SPECS, **OPTIONAL_TABLE_SPECS}
    for filename, frame in frames.items():
        spec = all_specs[filename]
        missing = [column for column in spec.columns if column not in frame.columns]
        if missing:
            errors.append(f"{filename}: missing columns {missing}")
            continue
        duplicate_count = int(frame.duplicated(list(spec.primary_key), keep=False).sum())
        if duplicate_count:
            errors.append(
                f"{filename}: {duplicate_count} rows violate primary key {spec.primary_key}"
            )
        for column in spec.primary_key:
            if frame[column].isna().any():
                errors.append(f"{filename}: primary-key column {column} contains nulls")
        for column in spec.date_columns:
            parsed = pd.to_datetime(frame[column], errors="coerce")
            invalid = int(parsed.isna().sum() - frame[column].isna().sum())
            if invalid:
                errors.append(f"{filename}: {invalid} invalid values in {column}")
        numeric = frame.select_dtypes(include=[np.number])
        infinite_count = int(np.isinf(numeric.to_numpy(dtype=float, na_value=np.nan)).sum())
        if infinite_count:
            errors.append(f"{filename}: {infinite_count} infinite numeric values")
        table_reports[filename] = {
            "rows": int(len(frame)),
            "columns": int(len(frame.columns)),
            "duplicate_key_rows": duplicate_count,
            "null_counts": {
                column: int(frame[column].isna().sum()) for column in spec.columns
            },
        }

    if errors:
        report = {
            "status": "FAIL",
            "raw_dir": str(raw_dir),
            "tables": table_reports,
            "errors": errors,
            "warnings": warnings,
            "models_fitted": [],
        }
        if raise_on_error:
            raise RawValidationError("; ".join(errors))
        return report

    funda = frames["funda.parquet"]
    crsp = frames["crsp_monthly.parquet"]
    delist = frames["crsp_delist.parquet"]
    ccm = frames["ccm_link.parquet"]
    factors = frames["factor_returns.parquet"]

    crsp_dates = pd.to_datetime(crsp["date"])
    if not (crsp_dates == crsp_dates + pd.offsets.MonthEnd(0)).all():
        errors.append("crsp_monthly.parquet: date must be calendar month-end")
    if (crsp["prc"] <= 0).any() or (crsp["shrout"] <= 0).any():
        errors.append("crsp_monthly.parquet: price and shares must be positive")
    if (crsp["ret"] < -1.0).any() or (delist["dlret"] < -1.0).any():
        errors.append("Returns below -100% are invalid")
    if not set(crsp["permno"]).issubset(set(ccm["lpermno"])):
        errors.append("CRSP contains PERMNOs absent from CCM")
    if not set(delist["permno"]).issubset(set(ccm["lpermno"])):
        errors.append("Delistings contain PERMNOs absent from CCM")
    if not set(crsp_dates).issubset(set(pd.to_datetime(factors["date"]))):
        errors.append("Factor returns do not cover all CRSP months")

    for gvkey, links in ccm.sort_values(["gvkey", "linkdt"]).groupby("gvkey"):
        previous_end: pd.Timestamp | None = None
        for link in links.itertuples(index=False):
            start = pd.Timestamp(link.linkdt)
            end = (
                pd.Timestamp(link.linkenddt)
                if pd.notna(link.linkenddt)
                else pd.Timestamp.max.normalize()
            )
            if end < start:
                errors.append(f"CCM link ends before it starts for gvkey={gvkey}")
            if previous_end is not None and start <= previous_end:
                errors.append(f"Overlapping CCM links for gvkey={gvkey}")
            previous_end = end

    if "sec_filing_dates.parquet" in frames:
        sec = frames["sec_filing_dates.parquet"]
        funda_keys = set(
            map(tuple, funda[["gvkey", "datadate", "fyear"]].itertuples(index=False, name=None))
        )
        sec_keys = set(
            map(tuple, sec[["gvkey", "datadate", "fyear"]].itertuples(index=False, name=None))
        )
        if not sec_keys.issubset(funda_keys):
            errors.append("SEC filing dates contain fiscal keys absent from funda")
        if (pd.to_datetime(sec["filing_timestamp"]) < pd.to_datetime(sec["datadate"])).any():
            errors.append("SEC filing timestamp precedes datadate")

    exit_categories = set(delist["exit_category"].astype(str))
    expected_categories = {
        "performance_failure",
        "merger",
        "voluntary_administrative",
        "other_unknown",
    }
    missing_categories = expected_categories - exit_categories
    if missing_categories:
        warnings.append(f"No observed delistings for categories: {sorted(missing_categories)}")

    report = {
        "status": "PASS" if not errors else "FAIL",
        "raw_dir": str(raw_dir),
        "tables": table_reports,
        "checks": {
            "required_schemas": not errors,
            "primary_keys_unique": all(
                table["duplicate_key_rows"] == 0 for table in table_reports.values()
            ),
            "crsp_month_end": bool(
                (crsp_dates == crsp_dates + pd.offsets.MonthEnd(0)).all()
            ),
            "ccm_permno_coverage": set(crsp["permno"]).issubset(
                set(ccm["lpermno"])
            ),
            "factor_month_coverage": set(crsp_dates).issubset(
                set(pd.to_datetime(factors["date"]))
            ),
            "delisting_categories_observed": sorted(exit_categories),
        },
        "errors": errors,
        "warnings": warnings,
        "models_fitted": [],
    }
    if errors and raise_on_error:
        raise RawValidationError("; ".join(errors))
    return report
