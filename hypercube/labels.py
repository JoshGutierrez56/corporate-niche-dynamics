"""Point-in-time fixed-horizon failure labels for P4 viability models."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


class PerformanceLabelError(ValueError):
    """Raised when an observed exit or follow-up contract is inconsistent."""


LABEL_KEY = ("gvkey", "datadate", "fyear", "horizon_years")


def _dated_join(
    observations: pd.DataFrame,
    links: pd.DataFrame,
    *,
    date_column: str,
) -> pd.DataFrame:
    """Map PERMNO observations to GVKEY using only dated CCM validity."""

    mapped = observations.merge(
        links[["gvkey", "lpermno", "linkdt", "linkenddt"]],
        left_on="permno",
        right_on="lpermno",
        how="left",
        validate="many_to_many",
    )
    date = pd.to_datetime(mapped[date_column])
    valid = (date >= pd.to_datetime(mapped["linkdt"])) & (
        mapped["linkenddt"].isna()
        | (date <= pd.to_datetime(mapped["linkenddt"]))
    )
    mapped = mapped.loc[valid].copy()
    duplicated = mapped.duplicated(["permno", date_column], keep=False)
    if duplicated.any():
        ambiguous = mapped.loc[duplicated, ["permno", date_column, "gvkey"]]
        if ambiguous.groupby(["permno", date_column])["gvkey"].nunique().gt(1).any():
            raise PerformanceLabelError("A dated security observation maps to multiple firms.")
        mapped = mapped.drop_duplicates(["permno", date_column, "gvkey"])
    return mapped


def observed_exits(raw_dir: Path) -> pd.DataFrame:
    """Return one observed exit per issuer with dated CCM provenance."""

    delist_path = raw_dir / "crsp_delist.parquet"
    link_path = raw_dir / "ccm_link.parquet"
    if not delist_path.is_file() or not link_path.is_file():
        raise FileNotFoundError(f"Missing delisting or CCM input below {raw_dir}")
    delist = pd.read_parquet(delist_path)
    links = pd.read_parquet(link_path)
    mapped = _dated_join(delist, links, date_column="dlstdt")
    if mapped.empty:
        return pd.DataFrame(
            columns=["gvkey", "exit_date", "exit_category", "dlstcd", "dlret"]
        )
    mapped = mapped.rename(columns={"dlstdt": "exit_date"}).sort_values(
        ["gvkey", "exit_date"]
    )
    if mapped.duplicated("gvkey").any():
        raise PerformanceLabelError("More than one observed exit exists for an issuer.")
    return mapped[["gvkey", "exit_date", "exit_category", "dlstcd", "dlret"]]


def issuer_followup(raw_dir: Path) -> pd.DataFrame:
    """Measure issuer observation bounds from raw security months and dated links."""

    monthly_path = raw_dir / "crsp_monthly.parquet"
    link_path = raw_dir / "ccm_link.parquet"
    if not monthly_path.is_file() or not link_path.is_file():
        raise FileNotFoundError(f"Missing CRSP or CCM input below {raw_dir}")
    monthly = pd.read_parquet(monthly_path, columns=["permno", "date"])
    links = pd.read_parquet(link_path)
    mapped = _dated_join(monthly, links, date_column="date")
    followup = (
        mapped.groupby("gvkey", as_index=False)
        .agg(first_observed_date=("date", "min"), last_observed_date=("date", "max"))
        .sort_values("gvkey")
    )
    return followup


def construct_fixed_horizon_labels(
    events: pd.DataFrame,
    raw_dir: Path,
    horizons_years: Iterable[int],
) -> pd.DataFrame:
    """Construct three/five-year performance-failure labels without truth data."""

    required = {
        "permno",
        "gvkey",
        "datadate",
        "fyear",
        "availability_date",
        "formation_date",
        "feature_date",
    }
    missing = sorted(required - set(events.columns))
    if missing:
        raise PerformanceLabelError(f"P3 events lack label inputs: {missing}")
    if events.duplicated(["gvkey", "datadate", "fyear"]).any():
        raise PerformanceLabelError("P3 events are not unique by accounting update.")
    if (pd.to_datetime(events["feature_date"]) < pd.to_datetime(events["formation_date"])).any():
        raise PerformanceLabelError("A feature date precedes eligible formation.")

    exits = observed_exits(raw_dir).set_index("gvkey")
    followup = issuer_followup(raw_dir).set_index("gvkey")
    sample_end = pd.read_parquet(
        raw_dir / "crsp_monthly.parquet", columns=["date"]
    )["date"].max()
    pieces: list[pd.DataFrame] = []
    identity = [
        "permno",
        "gvkey",
        "datadate",
        "fyear",
        "availability_date",
        "formation_date",
        "feature_date",
    ]
    base = events[identity].copy()
    base = base.join(exits, on="gvkey").join(followup, on="gvkey")
    if base["last_observed_date"].isna().any():
        raise PerformanceLabelError("A P3 event lacks raw security follow-up.")

    for horizon in horizons_years:
        if int(horizon) not in {3, 5}:
            raise PerformanceLabelError(f"Unsupported P4 horizon: {horizon}")
        frame = base.copy()
        frame["horizon_years"] = int(horizon)
        frame["horizon_end"] = pd.to_datetime(frame["feature_date"]) + pd.DateOffset(
            years=int(horizon)
        )
        exit_in_horizon = (
            frame["exit_date"].notna()
            & (frame["exit_date"] > frame["feature_date"])
            & (frame["exit_date"] <= frame["horizon_end"])
        )
        failure = exit_in_horizon & frame["exit_category"].eq("performance_failure")
        competing = exit_in_horizon & ~frame["exit_category"].eq(
            "performance_failure"
        )
        observed_through_horizon = frame["last_observed_date"] >= frame["horizon_end"]

        frame["failure_within_horizon"] = pd.Series(
            pd.NA, index=frame.index, dtype="Int8"
        )
        frame.loc[failure, "failure_within_horizon"] = 1
        frame.loc[~failure & ~competing & observed_through_horizon, "failure_within_horizon"] = 0
        frame["label_status"] = "censored"
        frame.loc[frame["failure_within_horizon"].notna(), "label_status"] = "observed"
        frame["censor_reason"] = "none"
        frame.loc[competing, "censor_reason"] = "competing_exit"
        lost = ~failure & ~competing & ~observed_through_horizon
        frame.loc[lost & (frame["horizon_end"] > sample_end), "censor_reason"] = (
            "administrative_right_censor"
        )
        frame.loc[lost & (frame["horizon_end"] <= sample_end), "censor_reason"] = (
            "lost_followup"
        )
        frame["label_observed_date"] = pd.NaT
        frame.loc[failure | competing, "label_observed_date"] = frame.loc[
            failure | competing, "exit_date"
        ]
        negative = frame["failure_within_horizon"].eq(0).fillna(False)
        frame.loc[negative, "label_observed_date"] = frame.loc[
            negative, "horizon_end"
        ]
        frame["sample_end_date"] = pd.Timestamp(sample_end)
        pieces.append(frame)

    labels = pd.concat(pieces, ignore_index=True)
    labels["gvkey"] = labels["gvkey"].astype("string")
    labels["label_status"] = labels["label_status"].astype("string")
    labels["censor_reason"] = labels["censor_reason"].astype("string")
    if labels.duplicated(list(LABEL_KEY)).any():
        raise PerformanceLabelError("Duplicate fixed-horizon label keys.")
    if (
        labels["failure_within_horizon"].eq(1).fillna(False)
        & (labels["exit_category"] != "performance_failure")
    ).any():
        raise PerformanceLabelError("A positive label lacks a performance exit.")
    numeric = labels["failure_within_horizon"].dropna().astype(int)
    if not np.isin(numeric, [0, 1]).all():
        raise PerformanceLabelError("Fixed-horizon labels are not binary.")
    return labels.sort_values(list(LABEL_KEY)).reset_index(drop=True)
