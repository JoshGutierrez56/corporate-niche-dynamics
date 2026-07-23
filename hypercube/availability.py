"""Point-in-time accounting availability and monthly formation mapping."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


FISCAL_KEY: tuple[str, ...] = ("gvkey", "datadate", "fyear")


class AvailabilityError(ValueError):
    """Raised when public-information timing cannot be reconciled safely."""


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], label: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise AvailabilityError(f"{label} is missing required columns: {missing}")


def first_strict_month_end(values: pd.Series) -> pd.Series:
    """Map timestamps to the first calendar month-end strictly afterward."""

    timestamps = pd.to_datetime(values, errors="coerce")
    if timestamps.isna().any():
        raise AvailabilityError("Availability timestamps must be non-null and valid.")
    normalized = timestamps.dt.normalize()
    current_month_end = normalized + pd.offsets.MonthEnd(0)
    result = current_month_end.where(timestamps < current_month_end)
    result = result.fillna(normalized + pd.offsets.MonthEnd(1))
    if not (result > timestamps).all():
        raise AvailabilityError("Formation mapping is not strictly post-availability.")
    return pd.to_datetime(result)


def assign_accounting_availability(
    funda: pd.DataFrame,
    sec_filing_dates: pd.DataFrame | None,
    *,
    fallback_days: int = 180,
) -> pd.DataFrame:
    """Apply the frozen SEC, RDQ, then conservative-fallback hierarchy."""

    _require_columns(funda, (*FISCAL_KEY, "rdq"), "funda")
    output = funda.copy()
    output["gvkey"] = output["gvkey"].astype("string")
    output["datadate"] = pd.to_datetime(output["datadate"], errors="coerce")
    output["rdq"] = pd.to_datetime(output["rdq"], errors="coerce")
    if output[list(FISCAL_KEY)].isna().any(axis=None):
        raise AvailabilityError("funda fiscal keys must be non-null.")
    if output.duplicated(list(FISCAL_KEY)).any():
        raise AvailabilityError("Duplicate funda fiscal keys are rejected in P2.")
    if output.duplicated(["gvkey", "fyear"]).any():
        raise AvailabilityError(
            "Multiple annual funda observations for one gvkey-fyear are rejected in P2."
        )

    output["filing_timestamp"] = pd.NaT
    if sec_filing_dates is not None:
        _require_columns(
            sec_filing_dates, (*FISCAL_KEY, "filing_timestamp"), "sec_filing_dates"
        )
        sec = sec_filing_dates.loc[:, [*FISCAL_KEY, "filing_timestamp"]].copy()
        sec["gvkey"] = sec["gvkey"].astype("string")
        sec["datadate"] = pd.to_datetime(sec["datadate"], errors="coerce")
        sec["filing_timestamp"] = pd.to_datetime(
            sec["filing_timestamp"], errors="coerce"
        )
        if sec[list(FISCAL_KEY)].isna().any(axis=None):
            raise AvailabilityError("SEC filing fiscal keys must be non-null.")
        if sec.duplicated(list(FISCAL_KEY)).any():
            raise AvailabilityError("Duplicate SEC filing fiscal keys are rejected in P2.")
        output = output.drop(columns="filing_timestamp").merge(
            sec,
            on=list(FISCAL_KEY),
            how="left",
            validate="one_to_one",
        )

    valid_sec = output["filing_timestamp"].notna() & (
        output["filing_timestamp"] >= output["datadate"]
    )
    valid_rdq = output["rdq"].notna() & (output["rdq"] >= output["datadate"])
    fallback = output["datadate"] + pd.to_timedelta(fallback_days, unit="D")

    output["availability_date"] = fallback
    output["availability_source"] = "datadate_plus_180_calendar_days"
    output["availability_confidence"] = "low"
    output.loc[valid_rdq, "availability_date"] = output.loc[valid_rdq, "rdq"]
    output.loc[valid_rdq, "availability_source"] = (
        "verified_earnings_announcement_date"
    )
    output.loc[valid_rdq, "availability_confidence"] = "medium"
    output.loc[valid_sec, "availability_date"] = output.loc[
        valid_sec, "filing_timestamp"
    ]
    output.loc[valid_sec, "availability_source"] = "verified_sec_filing_timestamp"
    output.loc[valid_sec, "availability_confidence"] = "high"
    output["formation_date"] = first_strict_month_end(output["availability_date"])
    output["invalid_sec_timestamp"] = output["filing_timestamp"].notna() & ~valid_sec
    output["invalid_rdq"] = output["rdq"].notna() & ~valid_rdq

    if (output["availability_date"] < output["datadate"]).any():
        raise AvailabilityError("An accounting record became available before datadate.")
    if not (output["formation_date"] > output["availability_date"]).all():
        raise AvailabilityError("Formation must be strictly after public availability.")
    return output.sort_values(list(FISCAL_KEY), kind="stable").reset_index(drop=True)


def staleness_in_months(panel_date: pd.Series, formation_date: pd.Series) -> pd.Series:
    """Return completed calendar-month distance from formation to panel month."""

    panel = pd.to_datetime(panel_date, errors="coerce")
    formation = pd.to_datetime(formation_date, errors="coerce")
    if panel.isna().any() or formation.isna().any():
        raise AvailabilityError("Staleness dates must be non-null and valid.")
    return ((panel.dt.year - formation.dt.year) * 12 + panel.dt.month - formation.dt.month).astype(
        "int16"
    )
