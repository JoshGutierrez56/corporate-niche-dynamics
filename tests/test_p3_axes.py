"""P3 component, standardization, diagnostics, and axis gates."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import pytest

from hypercube.axes import (
    COMPONENT_SPECS,
    build_axis_bundle,
    construct_accounting_components,
    validate_p3_directory,
)
from hypercube.config import HypercubeConfig, load_config
from hypercube.data import generate_synthetic_bundle
from hypercube.standardize import (
    anchored_standardize_events,
    relative_standardize_events,
)
from hypercube.universe import build_point_in_time_panel


ROOT = Path(__file__).resolve().parents[1]


def _config_with_small_references() -> HypercubeConfig:
    config = load_config(ROOT / "configs" / "synthetic.yaml")
    return config.model_copy(
        update={
            "project": config.project.model_copy(update={"phase": "P3"}),
            "standardization": config.standardization.model_copy(
                update={
                    "minimum_peer_group_size": 2,
                    "minimum_anchor_observations": 2,
                }
            )
        }
    )


def _p2_copy(config: HypercubeConfig) -> HypercubeConfig:
    return config.model_copy(
        update={"project": config.project.model_copy(update={"phase": "P2"})}
    )


def _accounting_example() -> pd.DataFrame:
    years = list(range(2015, 2020))
    sale = [100.0, 110.0, 121.0, 133.1, 146.41]
    return pd.DataFrame(
        {
            "gvkey": ["1"] * 5,
            "datadate": pd.to_datetime([f"{year}-12-31" for year in years]),
            "fyear": years,
            "availability_date": pd.to_datetime(
                [f"{year + 1}-02-15" for year in years]
            ),
            "formation_date": pd.to_datetime(
                [f"{year + 1}-02-29" if (year + 1) % 4 == 0 else f"{year + 1}-02-28" for year in years]
            ),
            "sich": [3571] * 5,
            "sale": sale,
            "cogs": [value * 0.6 for value in sale],
            "xrd": [None, 5.0, 6.0, 7.0, 8.0],
            "xsga": [20.0, 21.0, 22.0, 23.0, 24.0],
            "xad": [1.0, None, 1.2, None, 1.4],
            "oiadp": [12.0, 13.2, 14.52, 15.972, 17.5692],
            "ib": [8.0, 8.8, 9.68, 10.648, 11.7128],
            "dp": [2.0] * 5,
            "at": [80.0, 85.0, 90.0, 95.0, 100.0],
            "ppent": [20.0] * 5,
            "capx": [3.0] * 5,
            "dltt": [10.0] * 5,
            "dlc": [2.0] * 5,
            "ceq": [40.0] * 5,
            "che": [5.0] * 5,
        }
    )


def test_component_formulas_use_only_current_and_prior_rows() -> None:
    """Hand-check transparent formulas and consecutive-history gates."""

    config = load_config(ROOT / "configs" / "synthetic.yaml")
    result = construct_accounting_components(_accounting_example(), config)
    assert result.loc[0, "gross_margin"] == pytest.approx(0.4)
    assert pd.isna(result.loc[2, "revenue_cagr_3y"])
    assert result.loc[3, "revenue_cagr_3y"] == pytest.approx(0.10)
    assert result.loc[2, "gross_margin_volatility_3y"] == pytest.approx(0.0)
    assert result.loc[0, "rd_intensity_zero"] == pytest.approx(0.0)
    assert result.loc[0, "rd_missing_indicator"] == 1
    assert result.loc[1, "asset_turnover"] == pytest.approx(110.0 / 80.0)
    assert result.loc[4, "asset_lightness"] == pytest.approx(0.8)
    values = result[
        [item.slug for item in COMPONENT_SPECS if item.slug in result]
    ].to_numpy(dtype=float)
    assert not np.isinf(values).any()


def test_relative_scores_ignore_future_peer_rows() -> None:
    """Adding a later cross-section cannot change an earlier relative score."""

    config = _config_with_small_references()
    peer = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31"] * 3),
            "sic2": [35, 35, 35],
            "sic1": [3, 3, 3],
            "value": [1.0, 2.0, 3.0],
        }
    )
    events = pd.DataFrame(
        {
            "feature_date": pd.to_datetime(["2020-01-31"]),
            "sic2": [35],
            "sic1": [3],
            "value": [2.0],
        }
    )
    before = relative_standardize_events(
        peer, events, ["value"], {"value": "industry"}, config
    )
    future = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-31"] * 3),
            "sic2": [35, 35, 35],
            "sic1": [3, 3, 3],
            "value": [100.0, 200.0, 300.0],
        }
    )
    after = relative_standardize_events(
        pd.concat([peer, future], ignore_index=True),
        events,
        ["value"],
        {"value": "industry"},
        config,
    )
    assert before.loc[0, "rel_value"] == after.loc[0, "rel_value"]
    assert before.loc[0, "rel_level_value"] == "sic2"


def test_anchored_scores_ignore_current_year_and_future_rows() -> None:
    """Historical anchors end in the prior calendar year."""

    config = _config_with_small_references()
    events = pd.DataFrame(
        {
            "feature_date": pd.to_datetime(
                ["2018-02-28", "2018-03-31", "2019-02-28", "2020-02-29"]
            ),
            "value": [1.0, 3.0, 2.0, 4.0],
        }
    )
    before = anchored_standardize_events(events, ["value"], config)
    extended = pd.concat(
        [
            events,
            pd.DataFrame(
                {"feature_date": pd.to_datetime(["2025-02-28"]), "value": [999.0]}
            ),
        ],
        ignore_index=True,
    )
    after = anchored_standardize_events(extended, ["value"], config)
    pd.testing.assert_series_equal(
        before["anch_value"], after.loc[: len(before) - 1, "anch_value"]
    )
    assert before.loc[2, "anch_n_value"] == 2
    assert before.loc[2, "anchor_reference_end_year"] == 2018


@pytest.fixture(scope="module")
def compact_p3_bundle(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, HypercubeConfig]:
    """Run compact P1-P3 data paths without fitting any model."""

    root = tmp_path_factory.mktemp("p3_bundle")
    raw = root / "raw"
    p2 = root / "p2"
    p3 = root / "p3"
    config = _config_with_small_references()
    generate_synthetic_bundle(
        config,
        "null_alpha",
        raw,
        n_firms=30,
        start_year=1995,
        end_year=2005,
        seed=9182,
    )
    build_point_in_time_panel(raw, p2, _p2_copy(config), scenario="null_alpha")
    build_axis_bundle(p2, p3, config, scenario="null_alpha")
    return p3, config


def test_compact_p3_bundle_passes_independent_gate(
    compact_p3_bundle: tuple[Path, HypercubeConfig],
) -> None:
    """Saved component and axis artifacts reopen and recompute exactly."""

    output, config = compact_p3_bundle
    report = validate_p3_directory(output, config)
    assert report["status"] == "PASS", report["errors"]
    assert report["rows"]["component_features"] > 0
    assert report["rows"]["component_features"] == report["rows"]["axis_scores"]
    assert report["models_fitted"] == []
    metadata = json.loads(
        (output / "transformation_metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["synthetic_truth_read"] is False
    assert len(metadata["component_catalog"]) == 19


def test_p3_validator_detects_component_tampering(
    compact_p3_bundle: tuple[Path, HypercubeConfig], tmp_path: Path
) -> None:
    """A changed saved component table fails recomputation or hash validation."""

    source, config = compact_p3_bundle
    corrupted = tmp_path / "corrupted"
    shutil.copytree(source, corrupted)
    path = corrupted / "component_features.parquet"
    frame = pd.read_parquet(path)
    frame.loc[frame.index[0], "rel_gross_margin"] = 99.0
    frame.to_parquet(path, index=False)
    report = validate_p3_directory(corrupted, config)
    assert report["status"] == "FAIL"
    assert any("clip" in item.lower() or "hash" in item.lower() for item in report["errors"])
