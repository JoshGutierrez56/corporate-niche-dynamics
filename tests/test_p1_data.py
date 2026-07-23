"""P1 synthetic generation and raw-contract tests."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

import pandas as pd
import pytest

from hypercube.config import load_config
from hypercube.data import (
    SCENARIOS,
    TABLE_SPECS,
    generate_synthetic_bundle,
    validate_raw_directory,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def generated_scenarios(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Generate compact deterministic versions of all three P1 scenarios."""

    root = tmp_path_factory.mktemp("p1_scenarios")
    config = load_config(ROOT / "configs" / "synthetic.yaml")
    outputs: dict[str, Path] = {}
    for offset, scenario in enumerate(SCENARIOS):
        output = root / scenario
        generate_synthetic_bundle(
            config,
            scenario,
            output,
            n_firms=30,
            start_year=1995,
            end_year=2005,
            seed=9010 + offset,
        )
        outputs[scenario] = output
    return outputs


def test_all_scenarios_pass_schema_and_key_gate(
    generated_scenarios: dict[str, Path],
) -> None:
    """Every scenario emits identical required schemas with unique keys."""

    for scenario, output in generated_scenarios.items():
        report = validate_raw_directory(output)
        assert report["status"] == "PASS", (scenario, report["errors"])
        assert report["models_fitted"] == []
        for filename, spec in TABLE_SPECS.items():
            frame = pd.read_parquet(output / filename)
            assert set(spec.columns).issubset(frame.columns)
            assert not frame.duplicated(list(spec.primary_key)).any()


def test_metadata_discloses_only_predeclared_injections(
    generated_scenarios: dict[str, Path],
) -> None:
    """Scenario metadata makes the known effects and no-model boundary explicit."""

    metadata = {
        name: json.loads((path / "synthetic_scenario_metadata.json").read_text())
        for name, path in generated_scenarios.items()
    }
    assert metadata["null_alpha"]["expected_signs"]["migration_surprise_to_future_return"] == "zero"
    assert metadata["null_alpha"]["injection"]["migration_alpha_monthly"] == 0.0
    assert metadata["migration_alpha"]["expected_signs"]["migration_surprise_to_future_return"] == "positive"
    assert metadata["migration_alpha"]["injection"]["migration_alpha_monthly"] > 0.0
    assert metadata["regime_shift"]["expected_signs"]["dated_viability_surface_change"] is True
    assert metadata["regime_shift"]["injection"]["regime_shift_year"] == 2000
    assert all(item["model_fitted"] is False for item in metadata.values())


def test_truth_table_preserves_null_alpha_and_known_scenario_markers(
    generated_scenarios: dict[str, Path],
) -> None:
    """Truth sidecars expose injections without requiring a fitted model."""

    null_truth = pd.read_parquet(
        generated_scenarios["null_alpha"] / "synthetic_truth.parquet"
    )
    migration_truth = pd.read_parquet(
        generated_scenarios["migration_alpha"] / "synthetic_truth.parquet"
    )
    regime_truth = pd.read_parquet(
        generated_scenarios["regime_shift"] / "synthetic_truth.parquet"
    )
    assert (null_truth["injected_return_alpha"] == 0.0).all()
    assert migration_truth["injected_return_alpha"].abs().sum() > 0.0
    assert {"baseline", "post_shift"}.issubset(set(regime_truth["regime_id"]))


def test_missingness_links_and_delistings_are_present(
    generated_scenarios: dict[str, Path],
) -> None:
    """Synthetic raw files include the difficult P2 cases by construction."""

    output = generated_scenarios["null_alpha"]
    funda = pd.read_parquet(output / "funda.parquet")
    crsp = pd.read_parquet(output / "crsp_monthly.parquet")
    ccm = pd.read_parquet(output / "ccm_link.parquet")
    delist = pd.read_parquet(output / "crsp_delist.parquet")
    assert funda["rdq"].isna().any()
    assert funda["xrd"].isna().any()
    assert funda["xad"].isna().any()
    assert crsp[["bid", "ask"]].isna().any(axis=None)
    assert len(ccm) >= funda["gvkey"].nunique()
    assert {
        "performance_failure",
        "merger",
        "voluntary_administrative",
        "other_unknown",
    }.issubset(set(delist["exit_category"]))


def test_generator_is_deterministic(tmp_path: Path) -> None:
    """The same seed and settings reproduce observation values exactly."""

    config = load_config(ROOT / "configs" / "synthetic.yaml")
    first = tmp_path / "first"
    second = tmp_path / "second"
    for output in (first, second):
        generate_synthetic_bundle(
            config,
            "null_alpha",
            output,
            n_firms=20,
            start_year=1995,
            end_year=2002,
            seed=777,
        )
    for filename in (*TABLE_SPECS, "synthetic_truth.parquet"):
        pd.testing.assert_frame_equal(
            pd.read_parquet(first / filename),
            pd.read_parquet(second / filename),
        )


def test_validator_rejects_duplicate_primary_key(
    generated_scenarios: dict[str, Path], tmp_path: Path
) -> None:
    """A duplicated firm-year fails rather than being silently resolved."""

    source = generated_scenarios["null_alpha"]
    corrupted = tmp_path / "corrupted"
    shutil.copytree(source, corrupted)
    funda_path = corrupted / "funda.parquet"
    funda = pd.read_parquet(funda_path)
    pd.concat([funda, funda.iloc[[0]]], ignore_index=True).to_parquet(
        funda_path, index=False
    )
    report = validate_raw_directory(corrupted)
    assert report["status"] == "FAIL"
    assert any("primary key" in error for error in report["errors"])
