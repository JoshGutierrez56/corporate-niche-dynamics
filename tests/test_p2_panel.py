"""P2 point-in-time availability, universe, link, and panel gates."""

from __future__ import annotations

from pathlib import Path
import json
import shutil

import pandas as pd
import pytest

from hypercube.availability import (
    AvailabilityError,
    assign_accounting_availability,
    first_strict_month_end,
    staleness_in_months,
)
from hypercube.config import load_config
from hypercube.data import generate_synthetic_bundle
from hypercube.universe import (
    UniverseError,
    build_point_in_time_panel,
    combine_delisting_returns,
    apply_universe_filters,
    map_valid_ccm_links,
    validate_p2_directory,
)


ROOT = Path(__file__).resolve().parents[1]


def _p2_config() -> object:
    config = load_config(ROOT / "configs" / "synthetic.yaml")
    return config.model_copy(
        update={"project": config.project.model_copy(update={"phase": "P2"})}
    )


def _minimal_funda() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gvkey": ["1", "2", "3", "4"],
            "datadate": pd.to_datetime(["2020-12-31"] * 4),
            "fyear": [2020] * 4,
            "rdq": pd.to_datetime(
                ["2021-02-15", "2021-02-20", None, "2020-01-01"]
            ),
        }
    )


def test_availability_hierarchy_and_invalid_source_fallback() -> None:
    """SEC wins, then valid RDQ, then the conservative 180-day fallback."""

    sec = pd.DataFrame(
        {
            "gvkey": ["1", "4"],
            "datadate": pd.to_datetime(["2020-12-31", "2020-12-31"]),
            "fyear": [2020, 2020],
            "filing_timestamp": pd.to_datetime(
                ["2021-03-01 16:00", "2020-01-02 16:00"]
            ),
        }
    )
    result = assign_accounting_availability(_minimal_funda(), sec)
    sources = result.set_index("gvkey")["availability_source"].to_dict()
    assert sources == {
        "1": "verified_sec_filing_timestamp",
        "2": "verified_earnings_announcement_date",
        "3": "datadate_plus_180_calendar_days",
        "4": "datadate_plus_180_calendar_days",
    }
    row = result.set_index("gvkey").loc["4"]
    assert row["invalid_sec_timestamp"]
    assert row["invalid_rdq"]
    assert row["availability_date"] == pd.Timestamp("2021-06-29")
    assert (result["formation_date"] > result["availability_date"]).all()


def test_strict_formation_mapping_handles_month_end_timestamp() -> None:
    """A month-end timestamp waits until the following formation month."""

    timestamps = pd.Series(
        pd.to_datetime(["2021-02-15 16:00", "2021-02-28 00:00", "2021-02-28 16:00"])
    )
    actual = first_strict_month_end(timestamps)
    expected = pd.Series(
        pd.to_datetime(["2021-02-28", "2021-03-31", "2021-03-31"])
    )
    pd.testing.assert_series_equal(actual.reset_index(drop=True), expected)


def test_duplicate_fiscal_keys_are_rejected() -> None:
    """P2 never resolves a fiscal duplicate using future knowledge."""

    duplicated = pd.concat([_minimal_funda(), _minimal_funda().iloc[[0]]])
    with pytest.raises(AvailabilityError, match="Duplicate funda"):
        assign_accounting_availability(duplicated, None)

    revised_year = _minimal_funda().copy()
    revision = revised_year.iloc[[0]].copy()
    revision["datadate"] = pd.Timestamp("2020-11-30")
    with pytest.raises(AvailabilityError, match="gvkey-fyear"):
        assign_accounting_availability(
            pd.concat([revised_year, revision], ignore_index=True), None
        )


def test_staleness_uses_completed_calendar_months() -> None:
    """Staleness is measured from formation month without day-count leakage."""

    actual = staleness_in_months(
        pd.Series(pd.to_datetime(["2022-06-30", "2022-07-31"])),
        pd.Series(pd.to_datetime(["2021-12-31", "2021-12-31"])),
    )
    assert actual.tolist() == [6, 7]


def test_delisting_returns_are_compounded_and_missingness_is_flagged() -> None:
    """Ordinary and delisting returns reconcile under the frozen rule."""

    crsp = pd.DataFrame(
        {
            "permno": [1, 2],
            "date": pd.to_datetime(["2020-01-31", "2020-01-31"]),
            "ret": [0.10, 0.05],
            "prc": [10.0, 20.0],
            "shrout": [1000.0, 1000.0],
            "vol": [1000.0, 1000.0],
            "exchcd": [1, 1],
            "shrcd": [10, 10],
        }
    )
    delist = pd.DataFrame(
        {
            "permno": [1, 2],
            "dlstdt": pd.to_datetime(["2020-01-15", "2020-01-15"]),
            "dlstcd": [500, 500],
            "dlret": [-0.50, float("nan")],
            "exit_category": ["performance_failure", "other_unknown"],
        }
    )
    result = combine_delisting_returns(crsp, delist).set_index("permno")
    assert result.loc[1, "ret_total"] == pytest.approx(-0.45)
    assert result.loc[2, "ret_total"] == pytest.approx(0.05)
    assert bool(result.loc[2, "delist_return_missing"])


def test_prior_eligible_delisting_month_is_preserved() -> None:
    """A distressed exit month survives if the prior month passed all filters."""

    config = _p2_config()
    crsp = pd.DataFrame(
        {
            "permno": [1, 1],
            "date": pd.to_datetime(["2020-01-31", "2020-02-29"]),
            "ret": [0.0, -0.5],
            "prc": [10.0, 0.5],
            "shrout": [2000.0, 2000.0],
            "vol": [1000.0, 10.0],
            "exchcd": [1, 1],
            "shrcd": [10, 10],
        }
    )
    delist = pd.DataFrame(
        {
            "permno": [1],
            "dlstdt": pd.to_datetime(["2020-02-15"]),
            "dlstcd": [500],
            "dlret": [-0.8],
            "exit_category": ["performance_failure"],
        }
    )
    combined = combine_delisting_returns(crsp, delist)
    result = apply_universe_filters(combined, config).frame
    exit_row = result.loc[result["date"].eq(pd.Timestamp("2020-02-29"))].iloc[0]
    assert bool(exit_row["delist_month_override"])
    assert not bool(exit_row["universe_eligible"])


def test_ambiguous_equal_priority_ccm_links_are_rejected() -> None:
    """Equal-priority links to different firms are not guessed away."""

    config = _p2_config()
    security = pd.DataFrame(
        {"permno": [1], "date": pd.to_datetime(["2020-01-31"])}
    )
    links = pd.DataFrame(
        {
            "gvkey": ["1", "2"],
            "lpermno": [1, 1],
            "linkdt": pd.to_datetime(["2010-01-01", "2010-01-01"]),
            "linkenddt": pd.to_datetime([None, None]),
            "linktype": ["LU", "LU"],
            "linkprim": ["P", "P"],
        }
    )
    with pytest.raises(UniverseError, match="Ambiguous equal-priority"):
        map_valid_ccm_links(security, links, config)


@pytest.fixture(scope="module")
def compact_p2_bundle(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, object]:
    """Construct one compact P2 bundle through the public build path."""

    root = tmp_path_factory.mktemp("p2_bundle")
    raw = root / "raw"
    output = root / "processed"
    config = _p2_config()
    generate_synthetic_bundle(
        config,
        "null_alpha",
        raw,
        n_firms=30,
        start_year=1995,
        end_year=2005,
        seed=7761,
    )
    build_point_in_time_panel(raw, output, config, scenario="null_alpha")
    return output, config


def test_compact_p2_bundle_passes_independent_gate(
    compact_p2_bundle: tuple[Path, object],
) -> None:
    """A saved bundle reopens with valid timing, links, returns, and counts."""

    output, config = compact_p2_bundle
    report = validate_p2_directory(output, config)  # type: ignore[arg-type]
    assert report["status"] == "PASS", report["errors"]
    assert report["rows"]["firm_month_panel"] > 0
    assert report["delisting_events"] > 0
    assert report["models_fitted"] == []
    panel = pd.read_parquet(output / "firm_month_panel.parquet")
    assert (panel["formation_date"] > panel["availability_date"]).all()
    assert (panel["date"] >= panel["formation_date"]).all()
    assert not panel.duplicated(["permno", "date"]).any()
    manifest = json.loads((output / "p2_manifest.json").read_text(encoding="utf-8"))
    rows = {item["name"]: item.get("rows") for item in manifest["files"]}
    assert rows["firm_month_panel.parquet"] == len(panel)


def test_validator_detects_saved_panel_tampering(
    compact_p2_bundle: tuple[Path, object], tmp_path: Path
) -> None:
    """Independent validation fails if a completed panel is changed."""

    source, config = compact_p2_bundle
    corrupted = tmp_path / "corrupted"
    shutil.copytree(source, corrupted)
    panel_path = corrupted / "firm_month_panel.parquet"
    panel = pd.read_parquet(panel_path)
    panel.loc[panel.index[0], "availability_date"] = panel.loc[
        panel.index[0], "formation_date"
    ]
    panel.to_parquet(panel_path, index=False)
    report = validate_p2_directory(corrupted, config)  # type: ignore[arg-type]
    assert report["status"] == "FAIL"
    assert any("formation" in error.lower() or "hash" in error.lower() for error in report["errors"])
