"""P0-P10 gates: imports, configuration, tree, and phase boundaries."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from hypercube.config import load_config


ROOT = Path(__file__).resolve().parents[1]
MODULES = (
    "hypercube",
    "hypercube.config",
    "hypercube.data",
    "hypercube.availability",
    "hypercube.universe",
    "hypercube.axes",
    "hypercube.standardize",
    "hypercube.labels",
    "hypercube.viability",
    "hypercube.dynamics",
    "hypercube.survival",
    "hypercube.returns",
    "hypercube.costs",
    "hypercube.portfolio",
    "hypercube.clustering",
    "hypercube.exploratory",
    "hypercube.power_calibration",
    "hypercube.proxy_redesign",
    "hypercube.canary",
    "hypercube.diagnostics",
    "hypercube.visualization",
    "hypercube.run",
)
REQUIRED_DIRECTORIES = (
    "configs",
    "docs",
    "hypercube",
    "scripts",
    "data/raw",
    "data/interim",
    "data/processed",
    "artifacts/manifests",
    "artifacts/models",
    "artifacts/tables",
    "artifacts/logs",
    "figures",
    "tests",
)


@pytest.mark.parametrize("module_name", MODULES)
def test_imports_every_package_module(module_name: str) -> None:
    """Every declared package module imports without empirical side effects."""

    importlib.import_module(module_name)


@pytest.mark.parametrize(
    ("config_name", "phase", "mode", "run_name", "generate"),
    (
        ("base.yaml", "P0", "synthetic", "base_p0", False),
        ("synthetic.yaml", "P10", "synthetic", "synthetic_p10", False),
        ("real.yaml", "P10", "real", "real_p10", False),
    ),
)
def test_config_loads_and_enforces_phase_boundary(
    config_name: str, phase: str, mode: str, run_name: str, generate: bool
) -> None:
    """Base and P2 overlays validate with raw generation disabled."""

    config = load_config(ROOT / "configs" / config_name)
    assert config.project.phase == phase
    assert config.project.run_name == run_name
    assert config.data.mode == mode
    assert config.data.generate_synthetic is generate
    assert config.data.allow_network is False
    assert config.data.allow_wrds is False
    assert config.availability.fallback_days == 180
    assert config.panel.max_staleness_months == 18
    assert config.standardization.minimum_peer_group_size == 20
    assert config.innovation.stock_depreciation_rate == 0.20
    assert config.viability.horizons_years == (3, 5)
    assert config.dynamics.primary_model == "combined_axes_logit"
    assert config.dynamics.survival_probability_threshold == 0.95
    assert config.survival.causes == ("performance_failure", "merger")
    assert config.returns.horizons_months == (1, 3, 6, 12)
    assert config.returns.primary_signal == "migration_surprise"
    assert config.costs.primary_cost_scenario == "conservative"
    assert tuple(item.name for item in config.costs.scenarios) == (
        "low",
        "conservative",
        "severe",
    )
    assert config.clustering.training_end_year == 2004
    assert config.clustering.representation == "anchored"
    assert len(config.axes.components) == 6


def test_repository_tree_has_p0_contract() -> None:
    """All requested P0 directories and root files exist."""

    for relative in REQUIRED_DIRECTORIES:
        assert (ROOT / relative).is_dir(), relative
    for relative in (
        "Makefile",
        "pyproject.toml",
        "requirements.txt",
        "README.md",
        "report.md",
        "docs/research_spec.md",
        "docs/point_in_time_policy.md",
    ):
        assert (ROOT / relative).is_file(), relative


def test_p10_figure_directory_contains_no_partial_files() -> None:
    """P10 figures must be atomically published without temporary names."""

    assert not any(
        path.name.startswith(".") and path.name != ".gitkeep"
        for path in (ROOT / "figures").glob("*")
    )
