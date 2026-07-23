"""Point-in-time CRSP universe, delisting, and CCM-link construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from hypercube.config import HypercubeConfig
from hypercube.availability import (
    assign_accounting_availability,
    first_strict_month_end,
    staleness_in_months,
)
from hypercube.data import SCENARIOS, Scenario, atomic_write_json, sha256_file


class UniverseError(ValueError):
    """Raised when a security-month cannot be reconciled without ambiguity."""


@dataclass(frozen=True)
class UniverseResult:
    """Filtered security-months and their cumulative row-count waterfall."""

    frame: pd.DataFrame
    waterfall: pd.DataFrame


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise UniverseError(f"{label} is missing required columns: {missing}")


def _month_end(values: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(values, errors="coerce")
    if timestamps.isna().any():
        raise UniverseError("Monthly dates must be non-null and valid.")
    return timestamps.dt.normalize() + pd.offsets.MonthEnd(0)


def combine_delisting_returns(
    crsp_monthly: pd.DataFrame, crsp_delist: pd.DataFrame
) -> pd.DataFrame:
    """Attach exit metadata and compound ordinary and delisting returns."""

    _require_columns(
        crsp_monthly,
        ("permno", "date", "ret", "prc", "shrout", "vol", "exchcd", "shrcd"),
        "crsp_monthly",
    )
    _require_columns(
        crsp_delist, ("permno", "dlstdt", "dlstcd", "dlret", "exit_category"), "crsp_delist"
    )
    monthly = crsp_monthly.copy()
    monthly["date"] = _month_end(monthly["date"])
    if monthly.duplicated(["permno", "date"]).any():
        raise UniverseError("Duplicate CRSP security-month keys are rejected.")

    delist = crsp_delist.copy()
    delist["dlstdt"] = pd.to_datetime(delist["dlstdt"], errors="coerce")
    delist["date"] = _month_end(delist["dlstdt"])
    if delist.duplicated(["permno", "date"]).any():
        raise UniverseError("Multiple delisting events in one security-month are ambiguous.")
    delist["has_delist_event"] = True
    output = monthly.merge(
        delist[[
            "permno",
            "date",
            "dlstdt",
            "dlstcd",
            "dlret",
            "exit_category",
            "has_delist_event",
        ]],
        on=["permno", "date"],
        how="left",
        validate="one_to_one",
    )
    output["has_delist_event"] = output["has_delist_event"].eq(True)
    output["delist_return_missing"] = output["has_delist_event"] & output["dlret"].isna()
    ordinary = pd.to_numeric(output["ret"], errors="coerce")
    delisting = pd.to_numeric(output["dlret"], errors="coerce")
    output["ret_total"] = ordinary
    both = ordinary.notna() & delisting.notna()
    output.loc[both, "ret_total"] = (
        (1.0 + ordinary.loc[both]) * (1.0 + delisting.loc[both]) - 1.0
    )
    delist_only = ordinary.isna() & delisting.notna()
    output.loc[delist_only, "ret_total"] = delisting.loc[delist_only]
    return output.sort_values(["permno", "date"], kind="stable").reset_index(drop=True)


def apply_universe_filters(
    crsp_with_delistings: pd.DataFrame, config: HypercubeConfig
) -> UniverseResult:
    """Apply contemporaneous filters and preserve prior-eligible exit months."""

    frame = crsp_with_delistings.copy()
    frame["market_cap_millions"] = frame["prc"].abs() * frame["shrout"] / 1000.0
    stages: list[tuple[str, pd.Series]] = [
        ("raw_crsp", pd.Series(True, index=frame.index)),
        ("common_shares", frame["shrcd"].isin(config.universe.share_codes)),
        ("major_exchanges", frame["exchcd"].isin(config.universe.exchange_codes)),
        ("positive_price", frame["prc"].abs().gt(0.0)),
        ("positive_market_cap", frame["market_cap_millions"].gt(0.0)),
        ("minimum_price", frame["prc"].abs().ge(config.universe.minimum_price)),
        (
            "minimum_market_cap",
            frame["market_cap_millions"].ge(
                config.universe.minimum_market_cap_millions
            ),
        ),
        ("minimum_volume", frame["vol"].ge(config.universe.minimum_monthly_volume)),
    ]
    mask = pd.Series(True, index=frame.index)
    records: list[dict[str, int | str]] = []
    previous = len(frame)
    for name, condition in stages:
        mask &= condition.fillna(False)
        remaining = int(mask.sum())
        records.append(
            {
                "stage": name,
                "rows_remaining": remaining,
                "rows_removed_at_stage": previous - remaining,
            }
        )
        previous = remaining

    baseline = mask.copy()
    prior_key = frame.loc[baseline, ["permno", "date"]].copy()
    prior_key["date"] = prior_key["date"] + pd.offsets.MonthEnd(1)
    prior_eligible = pd.MultiIndex.from_frame(prior_key)
    current_key = pd.MultiIndex.from_frame(frame[["permno", "date"]])
    delist_override = frame["has_delist_event"] & current_key.isin(prior_eligible)
    final_mask = baseline | delist_override
    overrides = int((delist_override & ~baseline).sum())
    records.append(
        {
            "stage": "prior_eligible_delist_override",
            "rows_remaining": int(final_mask.sum()),
            "rows_removed_at_stage": -overrides,
        }
    )
    output = frame.loc[final_mask].copy()
    output["universe_eligible"] = baseline.loc[final_mask].to_numpy()
    output["delist_month_override"] = (
        delist_override.loc[final_mask] & ~baseline.loc[final_mask]
    ).to_numpy()
    return UniverseResult(
        frame=output.sort_values(["permno", "date"], kind="stable").reset_index(drop=True),
        waterfall=pd.DataFrame.from_records(records),
    )


def map_valid_ccm_links(
    security_months: pd.DataFrame,
    ccm_link: pd.DataFrame,
    config: HypercubeConfig,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Map valid CCM links with deterministic priority and reject equal ties."""

    _require_columns(
        ccm_link,
        ("gvkey", "lpermno", "linkdt", "linkenddt", "linktype", "linkprim"),
        "ccm_link",
    )
    links = ccm_link.copy()
    links["linkdt"] = pd.to_datetime(links["linkdt"], errors="coerce")
    links["linkenddt"] = pd.to_datetime(links["linkenddt"], errors="coerce")
    links = links.loc[
        links["linktype"].isin(config.panel.allowed_link_types)
        & links["linkprim"].isin(config.panel.allowed_link_primaries)
    ].copy()
    if links[["gvkey", "lpermno", "linkdt"]].isna().any(axis=None):
        raise UniverseError("CCM link keys must be non-null.")
    if (links["linkenddt"].notna() & (links["linkenddt"] < links["linkdt"])).any():
        raise UniverseError("A CCM link ends before it starts.")

    candidates = security_months.merge(
        links,
        left_on="permno",
        right_on="lpermno",
        how="left",
        validate="many_to_many",
    )
    valid = candidates["linkdt"].notna() & (candidates["date"] >= candidates["linkdt"])
    valid &= candidates["linkenddt"].isna() | (
        candidates["date"] <= candidates["linkenddt"]
    )
    candidates = candidates.loc[valid].copy()
    primary_rank = {"P": 0, "C": 1}
    type_rank = {"LC": 0, "LU": 1, "LS": 2}
    candidates["_primary_rank"] = candidates["linkprim"].map(primary_rank).fillna(99)
    candidates["_type_rank"] = candidates["linktype"].map(type_rank).fillna(99)
    candidates = candidates.sort_values(
        ["permno", "date", "_primary_rank", "_type_rank", "linkdt", "gvkey"],
        kind="stable",
    )
    candidates["_combined_rank"] = (
        candidates["_primary_rank"] * 10 + candidates["_type_rank"]
    )
    best_rank = candidates.groupby(["permno", "date"], observed=True)[
        "_combined_rank"
    ].transform("min")
    tied = candidates.loc[candidates["_combined_rank"].eq(best_rank)]
    ambiguous = tied.groupby(["permno", "date"], observed=True)["gvkey"].nunique()
    ambiguous = ambiguous[ambiguous > 1]
    if not ambiguous.empty:
        sample = [tuple(item) for item in ambiguous.index[:5]]
        raise UniverseError(f"Ambiguous equal-priority CCM links: {sample}")
    mapped = candidates.drop_duplicates(["permno", "date"], keep="first")
    mapped = mapped.drop(columns=["_primary_rank", "_type_rank", "_combined_rank"])
    diagnostics = {
        "input_security_months": int(len(security_months)),
        "mapped_security_months": int(len(mapped)),
        "unmapped_security_months": int(len(security_months) - len(mapped)),
        "ambiguous_equal_priority": 0,
        "eligible_link_rows": int(len(links)),
    }
    return mapped.sort_values(["permno", "date"], kind="stable").reset_index(drop=True), diagnostics


def verify_delisting_formula(frame: pd.DataFrame) -> int:
    """Return the number of rows violating the frozen compounding rule."""

    both = frame["ret"].notna() & frame["dlret"].notna()
    expected = (1.0 + frame.loc[both, "ret"]) * (1.0 + frame.loc[both, "dlret"]) - 1.0
    return int((~np.isclose(frame.loc[both, "ret_total"], expected, rtol=1e-12, atol=1e-12)).sum())


def scenario_processed_dir(config: HypercubeConfig, scenario: Scenario) -> Path:
    """Resolve sibling processed directories for the synthetic scenarios."""

    configured = Path(config.paths.processed_dir)
    if configured.name in SCENARIOS:
        return configured.parent / scenario
    return configured / scenario


def _read_raw_bundle(raw_dir: Path) -> dict[str, pd.DataFrame]:
    required = (
        "funda.parquet",
        "crsp_monthly.parquet",
        "crsp_delist.parquet",
        "ccm_link.parquet",
    )
    missing = [name for name in required if not (raw_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing P2 raw inputs in {raw_dir}: {missing}")
    frames = {name: pd.read_parquet(raw_dir / name) for name in required}
    sec_path = raw_dir / "sec_filing_dates.parquet"
    if sec_path.is_file():
        frames[sec_path.name] = pd.read_parquet(sec_path)
    return frames


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    frame.to_parquet(path, index=False, compression="zstd")


def _panel_from_frames(
    frames: Mapping[str, pd.DataFrame], config: HypercubeConfig
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    accounting = assign_accounting_availability(
        frames["funda.parquet"],
        frames.get("sec_filing_dates.parquet"),
        fallback_days=config.availability.fallback_days,
    )
    accounting = accounting.sort_values(
        ["gvkey", "formation_date", "datadate", "fyear"], kind="stable"
    ).reset_index(drop=True)
    accounting["reporting_history"] = (
        accounting.groupby("gvkey", observed=True).cumcount() + 1
    ).astype("int16")

    with_delistings = combine_delisting_returns(
        frames["crsp_monthly.parquet"], frames["crsp_delist.parquet"]
    )
    delist_after_crsp = int(with_delistings["has_delist_event"].sum())
    universe = apply_universe_filters(with_delistings, config)
    delist_after_universe = int(universe.frame["has_delist_event"].sum())
    mapped, link_diagnostics = map_valid_ccm_links(
        universe.frame, frames["ccm_link.parquet"], config
    )
    delist_after_ccm = int(mapped["has_delist_event"].sum())
    mapped["gvkey"] = mapped["gvkey"].astype("string")

    left = mapped.sort_values(["date", "gvkey", "permno"], kind="stable")
    right = accounting.sort_values(
        ["formation_date", "gvkey", "datadate", "fyear"], kind="stable"
    )
    panel = pd.merge_asof(
        left,
        right,
        by="gvkey",
        left_on="date",
        right_on="formation_date",
        direction="backward",
        allow_exact_matches=True,
    )
    mapped_rows = len(panel)
    panel = panel.loc[panel["formation_date"].notna()].copy()
    rows_after_available_accounting = len(panel)
    delist_after_accounting = int(panel["has_delist_event"].sum())
    panel = panel.loc[
        panel["reporting_history"].ge(config.panel.minimum_reporting_history)
    ].copy()
    rows_after_history = len(panel)
    delist_after_history = int(panel["has_delist_event"].sum())
    panel["staleness_months"] = staleness_in_months(
        panel["date"], panel["formation_date"]
    )
    panel = panel.loc[
        panel["staleness_months"].between(0, config.panel.max_staleness_months)
    ].copy()
    panel = panel.sort_values(["permno", "date"], kind="stable").reset_index(drop=True)

    waterfall = universe.waterfall.copy()
    additions = [
        ("valid_ccm_link", len(mapped)),
        ("available_accounting", rows_after_available_accounting),
        ("minimum_reporting_history", rows_after_history),
        ("maximum_staleness", len(panel)),
    ]
    previous = int(waterfall.iloc[-1]["rows_remaining"])
    for stage, remaining in additions:
        waterfall.loc[len(waterfall)] = {
            "stage": stage,
            "rows_remaining": int(remaining),
            "rows_removed_at_stage": max(0, previous - int(remaining)),
        }
        previous = int(remaining)

    availability_counts = {
        str(key): int(value)
        for key, value in accounting["availability_source"].value_counts().items()
    }
    exit_counts = {
        str(key): int(value)
        for key, value in panel.loc[panel["has_delist_event"], "exit_category"]
        .value_counts(dropna=False)
        .items()
    }
    diagnostics: dict[str, object] = {
        "status": "PASS",
        "phase": "P2",
        "models_fitted": [],
        "input_rows": {
            name: int(len(frame)) for name, frame in frames.items()
        },
        "output_rows": {
            "accounting_availability": int(len(accounting)),
            "universe_monthly": int(len(universe.frame)),
            "firm_month_panel": int(len(panel)),
        },
        "availability_source_counts": availability_counts,
        "availability_confidence_counts": {
            str(key): int(value)
            for key, value in accounting["availability_confidence"].value_counts().items()
        },
        "invalid_sec_timestamps": int(accounting["invalid_sec_timestamp"].sum()),
        "invalid_rdq_dates": int(accounting["invalid_rdq"].sum()),
        "ccm": link_diagnostics,
        "delisting_events_in_panel": int(panel["has_delist_event"].sum()),
        "delisting_reconciliation": {
            "raw_events": int(len(frames["crsp_delist.parquet"])),
            "matched_to_crsp_month": delist_after_crsp,
            "after_universe_filters": delist_after_universe,
            "after_ccm_mapping": delist_after_ccm,
            "after_available_accounting": delist_after_accounting,
            "after_minimum_history": delist_after_history,
            "final_panel": int(panel["has_delist_event"].sum()),
        },
        "missing_delisting_returns_in_panel": int(panel["delist_return_missing"].sum()),
        "exit_category_counts": exit_counts,
        "delisting_formula_violations": verify_delisting_formula(panel),
        "pre_availability_violations": int(
            (panel["formation_date"] <= panel["availability_date"]).sum()
            + (panel["date"] < panel["formation_date"]).sum()
        ),
        "duplicate_panel_keys": int(panel.duplicated(["permno", "date"]).sum()),
        "staleness_violations": int(
            (~panel["staleness_months"].between(0, config.panel.max_staleness_months)).sum()
        ),
        "universe_rows_before_accounting": int(mapped_rows),
    }
    return accounting, universe.frame, panel, waterfall, diagnostics


def _output_inventory(directory: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name != "p2_manifest.json":
            record: dict[str, object] = {
                "name": path.name,
                "bytes": int(path.stat().st_size),
                "sha256": sha256_file(path),
            }
            if path.suffix == ".parquet":
                record["rows"] = int(pq.ParquetFile(path).metadata.num_rows)
            records.append(record)
    return records


def build_point_in_time_panel(
    raw_dir: Path,
    output_dir: Path,
    config: HypercubeConfig,
    *,
    scenario: str,
) -> dict[str, object]:
    """Build and atomically publish one P2 point-in-time panel bundle."""

    if config.project.phase != "P2":
        raise UniverseError("Point-in-time panel construction requires a P2 config.")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite completed P2 output: {output_dir}")
    frames = _read_raw_bundle(raw_dir)
    accounting, universe, panel, waterfall, diagnostics = _panel_from_frames(
        frames, config
    )
    violations = sum(
        int(diagnostics[key])
        for key in (
            "delisting_formula_violations",
            "pre_availability_violations",
            "duplicate_panel_keys",
            "staleness_violations",
        )
    )
    if violations:
        raise UniverseError(f"P2 construction produced {violations} gate violations.")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.p2-", dir=output_dir.parent))
    try:
        _write_frame(accounting, staging / "accounting_availability.parquet")
        _write_frame(universe, staging / "universe_monthly.parquet")
        _write_frame(panel, staging / "firm_month_panel.parquet")
        waterfall.to_csv(staging / "row_count_waterfall.csv", index=False, lineterminator="\n")
        atomic_write_json(staging / "p2_diagnostics.json", diagnostics)
        atomic_write_json(
            staging / "resolved_config.json", config.model_dump(mode="json")
        )
        manifest: dict[str, object] = {
            "schema_version": 1,
            "phase": "P2",
            "scenario": scenario,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "raw_dir": str(raw_dir),
            "output_dir": str(output_dir),
            "seed": config.project.seed,
            "files": _output_inventory(staging),
            "models_fitted": [],
        }
        atomic_write_json(staging / "p2_manifest.json", manifest)
        if output_dir.exists():
            output_dir.rmdir()
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return diagnostics


def validate_p2_directory(
    output_dir: Path, config: HypercubeConfig
) -> dict[str, object]:
    """Independently reopen a saved P2 bundle and enforce its gate."""

    required = (
        "accounting_availability.parquet",
        "universe_monthly.parquet",
        "firm_month_panel.parquet",
        "row_count_waterfall.csv",
        "p2_diagnostics.json",
        "resolved_config.json",
        "p2_manifest.json",
    )
    errors: list[str] = []
    missing = [name for name in required if not (output_dir / name).is_file()]
    if missing:
        return {
            "status": "FAIL",
            "errors": [f"Missing P2 outputs: {missing}"],
            "warnings": [],
            "models_fitted": [],
        }
    accounting = pd.read_parquet(output_dir / "accounting_availability.parquet")
    universe = pd.read_parquet(output_dir / "universe_monthly.parquet")
    panel = pd.read_parquet(output_dir / "firm_month_panel.parquet")
    waterfall = pd.read_csv(output_dir / "row_count_waterfall.csv")
    manifest = json.loads((output_dir / "p2_manifest.json").read_text(encoding="utf-8"))

    if accounting.duplicated(["gvkey", "datadate", "fyear"]).any():
        errors.append("Duplicate fiscal keys in accounting availability output.")
    if accounting.duplicated(["gvkey", "fyear"]).any():
        errors.append("Multiple annual observations for one gvkey-fyear in output.")
    valid_sec = accounting["filing_timestamp"].notna() & (
        accounting["filing_timestamp"] >= accounting["datadate"]
    )
    valid_rdq = accounting["rdq"].notna() & (
        accounting["rdq"] >= accounting["datadate"]
    )
    expected_source = pd.Series(
        "datadate_plus_180_calendar_days", index=accounting.index
    )
    expected_source.loc[valid_rdq] = "verified_earnings_announcement_date"
    expected_source.loc[valid_sec] = "verified_sec_filing_timestamp"
    expected_date = accounting["datadate"] + pd.to_timedelta(
        config.availability.fallback_days, unit="D"
    )
    expected_date.loc[valid_rdq] = accounting.loc[valid_rdq, "rdq"]
    expected_date.loc[valid_sec] = accounting.loc[valid_sec, "filing_timestamp"]
    if not accounting["availability_source"].eq(expected_source).all():
        errors.append("Availability source does not follow the frozen hierarchy.")
    if not accounting["availability_date"].eq(expected_date).all():
        errors.append("Availability date does not match its selected public source.")
    expected_formation = first_strict_month_end(accounting["availability_date"])
    if not accounting["formation_date"].eq(expected_formation).all():
        errors.append("Saved formation date is not the first strict month-end.")
    if not (accounting["formation_date"] > accounting["availability_date"]).all():
        errors.append("Accounting formation is not strictly post-availability.")
    if universe.duplicated(["permno", "date"]).any():
        errors.append("Duplicate security-month keys in universe output.")
    if panel.duplicated(["permno", "date"]).any():
        errors.append("Duplicate security-month keys in panel output.")
    if not (panel["formation_date"] > panel["availability_date"]).all():
        errors.append("Panel contains non-strict formation mapping.")
    if (panel["date"] < panel["formation_date"]).any():
        errors.append("Panel uses accounting before its eligible formation month.")
    valid_links = (panel["date"] >= panel["linkdt"]) & (
        panel["linkenddt"].isna() | (panel["date"] <= panel["linkenddt"])
    )
    if not valid_links.all():
        errors.append("Panel contains rows outside CCM link validity dates.")
    if not panel["linktype"].isin(config.panel.allowed_link_types).all():
        errors.append("Panel contains a non-allowlisted CCM link type.")
    if not panel["linkprim"].isin(config.panel.allowed_link_primaries).all():
        errors.append("Panel contains a non-allowlisted CCM primary code.")
    if not panel["staleness_months"].between(0, config.panel.max_staleness_months).all():
        errors.append("Panel violates the maximum staleness rule.")
    formula_violations = verify_delisting_formula(panel)
    if formula_violations:
        errors.append(f"Delisting return formula violations: {formula_violations}.")
    events = panel["has_delist_event"]
    if panel.loc[events, ["dlstdt", "dlstcd", "exit_category"]].isna().any(axis=None):
        errors.append("A panel delisting event is missing exit metadata.")
    if int(waterfall.iloc[-1]["rows_remaining"]) != len(panel):
        errors.append("Waterfall final row count does not match panel.")
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
            "accounting_availability": int(len(accounting)),
            "universe_monthly": int(len(universe)),
            "firm_month_panel": int(len(panel)),
        },
        "availability_sources": {
            str(key): int(value)
            for key, value in accounting["availability_source"].value_counts().items()
        },
        "delisting_events": int(panel["has_delist_event"].sum()),
        "models_fitted": [],
    }
