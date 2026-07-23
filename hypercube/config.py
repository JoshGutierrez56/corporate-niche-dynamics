"""Strict configuration loading for phased Hypercube runs."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Base model that rejects undeclared configuration fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ProjectConfig(StrictModel):
    """Project identity and reproducibility settings."""

    name: Literal["business-niche-hypercube"]
    subtitle: str
    phase: Literal["P0", "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"]
    run_name: str
    python_minimum: str
    seed: int = Field(ge=0)


class PathsConfig(StrictModel):
    """Project-relative artifact locations."""

    raw_dir: str
    interim_dir: str
    processed_dir: str
    artifacts_dir: str
    figures_dir: str


class DataConfig(StrictModel):
    """Data-mode controls for offline synthetic or local real inputs."""

    mode: Literal["synthetic", "real"]
    scenario: Literal[
        "null_alpha", "migration_alpha", "regime_shift", "not_applicable"
    ]
    require_existing_raw: bool
    generate_synthetic: bool
    allow_network: bool
    allow_wrds: bool

    @model_validator(mode="after")
    def validate_mode(self) -> "DataConfig":
        """Keep scenario and real-data settings internally consistent."""

        if self.mode == "real" and self.scenario != "not_applicable":
            raise ValueError("Real mode requires scenario=not_applicable.")
        if self.mode == "synthetic" and self.scenario == "not_applicable":
            raise ValueError("Synthetic mode requires an injected scenario.")
        if self.mode == "real" and self.generate_synthetic:
            raise ValueError("Real mode cannot generate synthetic inputs.")
        if self.allow_network or self.allow_wrds:
            raise ValueError("Network and WRDS access are disabled through P1.")
        return self


class SyntheticConfig(StrictModel):
    """Deterministic P1 data-generating-process settings."""

    n_firms: int = Field(ge=20)
    start_year: int = Field(ge=1900)
    end_year: int = Field(le=2100)
    n_industries: int = Field(ge=3)
    minimum_history_years: int = Field(ge=2)
    regime_shift_year: int
    migration_alpha_monthly: float = Field(ge=0.0)
    migration_decay_months: int = Field(ge=1, le=24)
    annual_state_persistence: float = Field(gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def validate_window(self) -> "SyntheticConfig":
        """Require a usable history and an interior regime date."""

        if self.end_year - self.start_year + 1 < self.minimum_history_years:
            raise ValueError("Synthetic window is shorter than minimum history.")
        if not self.start_year < self.regime_shift_year <= self.end_year:
            raise ValueError("Regime shift year must be inside the sample window.")
        return self


class AvailabilityConfig(StrictModel):
    """Conservative public-information timing policy."""

    hierarchy: tuple[
        Literal[
            "verified_sec_filing_timestamp",
            "verified_earnings_announcement_date",
            "datadate_plus_180_calendar_days",
        ],
        ...,
    ]
    fallback_days: int = Field(ge=180)
    prohibit_pre_availability_use: bool

    @model_validator(mode="after")
    def validate_hierarchy(self) -> "AvailabilityConfig":
        """Require the exact predeclared availability hierarchy."""

        expected = (
            "verified_sec_filing_timestamp",
            "verified_earnings_announcement_date",
            "datadate_plus_180_calendar_days",
        )
        if self.hierarchy != expected:
            raise ValueError(f"Availability hierarchy must be {expected!r}.")
        if not self.prohibit_pre_availability_use:
            raise ValueError("Pre-availability use must remain prohibited.")
        return self


class UniverseConfig(StrictModel):
    """Point-in-time investable-universe boundary."""

    share_codes: tuple[int, ...]
    exchange_codes: tuple[int, ...]
    require_positive_price: bool
    require_positive_market_cap: bool
    include_delisted_firms: bool
    minimum_price: float = Field(ge=0.0)
    minimum_market_cap_millions: float = Field(ge=0.0)
    minimum_monthly_volume: float = Field(ge=0.0)
    preserve_prior_eligible_delist_month: bool

    @model_validator(mode="after")
    def validate_codes(self) -> "UniverseConfig":
        """Require the P0 candidate security and exchange codes."""

        if set(self.share_codes) != {10, 11}:
            raise ValueError("P0 share codes must be {10, 11}.")
        if set(self.exchange_codes) != {1, 2, 3}:
            raise ValueError("P0 exchange codes must be {1, 2, 3}.")
        if not self.include_delisted_firms:
            raise ValueError("Delisted firms must remain included.")
        if not self.preserve_prior_eligible_delist_month:
            raise ValueError("Prior-eligible delisting months must be preserved.")
        return self


class PanelConfig(StrictModel):
    """Frozen P2 panel-construction and reconciliation policies."""

    allowed_link_types: tuple[Literal["LC", "LU", "LS"], ...]
    allowed_link_primaries: tuple[Literal["P", "C"], ...]
    max_staleness_months: int = Field(ge=1)
    minimum_reporting_history: int = Field(ge=1)
    fiscal_duplicate_policy: Literal["reject"]
    delisting_return_rule: Literal["compound_with_ordinary_return"]
    strict_post_availability_formation: bool

    @model_validator(mode="after")
    def validate_panel_policy(self) -> "PanelConfig":
        """Prevent silent relaxation of the predeclared P2 policies."""

        if not self.allowed_link_types:
            raise ValueError("At least one CCM link type must be allowed.")
        if not self.allowed_link_primaries:
            raise ValueError("At least one CCM primary code must be allowed.")
        if not self.strict_post_availability_formation:
            raise ValueError("Formation must remain strictly post-availability.")
        return self


class AxisComponentConfig(StrictModel):
    """Declaration and minimum-coverage rule for one P3 axis."""

    name: str
    slug: str
    enabled: bool
    implementation_phase: Literal["P3"]
    minimum_components: int = Field(ge=1)
    ablation_enabled: bool


class AxesConfig(StrictModel):
    """Six-axis naming and orientation contract."""

    orientation: Literal["higher_is_more_viable"]
    components: tuple[AxisComponentConfig, ...]

    @model_validator(mode="after")
    def validate_axes(self) -> "AxesConfig":
        """Require six enabled, uniquely named P3 axis declarations."""

        if len(self.components) != 6:
            raise ValueError("Exactly six axes are required.")
        slugs = [component.slug for component in self.components]
        if len(set(slugs)) != len(slugs):
            raise ValueError("Axis slugs must be unique.")
        if not all(component.enabled for component in self.components):
            raise ValueError("All baseline axes must be enabled in P0.")
        return self


class StandardizationConfig(StrictModel):
    """Frozen P3 robust relative and historical-reference rules."""

    winsor_lower: float = Field(ge=0.0, lt=0.5)
    winsor_upper: float = Field(gt=0.5, le=1.0)
    minimum_peer_group_size: int = Field(ge=5)
    relative_method: Literal["median_mad"]
    peer_fallback_hierarchy: tuple[Literal["sic2", "sic1", "market"], ...]
    anchored_method: Literal["expanding_prior_calendar_years"]
    minimum_anchor_observations: int = Field(ge=20)
    score_clip: float = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_standardization(self) -> "StandardizationConfig":
        """Require ordered tails and the declared peer fallback hierarchy."""

        if self.winsor_lower >= self.winsor_upper:
            raise ValueError("Winsor lower tail must be below upper tail.")
        if self.peer_fallback_hierarchy != ("sic2", "sic1", "market"):
            raise ValueError("Peer fallback must be sic2, sic1, then market.")
        return self


class InnovationConfig(StrictModel):
    """Frozen P3 R&D treatment and capitalized-stock assumptions."""

    primary_missing_policy: Literal["zero_with_indicator"]
    sensitivity_missing_policy: Literal["contemporaneous_peer_median"]
    stock_depreciation_rate: float = Field(gt=0.0, lt=1.0)
    persistence_years: Literal[3]
    fixed_nopat_tax_rate: float = Field(ge=0.0, lt=1.0)


class ViabilityFoldConfig(StrictModel):
    """One untouched outer test window for fixed-horizon viability."""

    start_year: int = Field(ge=1900)
    end_year: int = Field(le=2100)

    @model_validator(mode="after")
    def validate_years(self) -> "ViabilityFoldConfig":
        """Require a nonempty calendar interval."""

        if self.end_year < self.start_year:
            raise ValueError("Viability fold end year precedes start year.")
        return self


class ViabilityConfig(StrictModel):
    """Frozen P4 label, validation, calibration, and model ladder."""

    horizons_years: tuple[Literal[3, 5], ...]
    validation_years: int = Field(ge=2)
    outer_folds: tuple[ViabilityFoldConfig, ...]
    ridge_c_grid: tuple[float, ...]
    calibration: Literal["platt"]
    cell_bins: Literal[3]
    minimum_train_observations: int = Field(ge=100)
    minimum_train_failures: int = Field(ge=5)
    minimum_validation_failures: int = Field(ge=2)
    primary_model: Literal["combined_axes_logit"]
    nonlinear_model: Literal["hist_gradient_boosting"]
    tree_max_iter: int = Field(ge=20, le=500)
    tree_max_leaf_nodes: int = Field(ge=3, le=63)
    tree_learning_rate: float = Field(gt=0.0, le=0.5)
    tree_min_samples_leaf: int = Field(ge=10)
    tree_l2_regularization: float = Field(ge=0.0)

    @model_validator(mode="after")
    def validate_viability_protocol(self) -> "ViabilityConfig":
        """Prevent silent expansion of the predeclared P4 search."""

        if self.horizons_years != (3, 5):
            raise ValueError("P4 horizons must remain exactly three and five years.")
        if not self.outer_folds:
            raise ValueError("At least one outer viability fold is required.")
        prior_end = -1
        for fold in self.outer_folds:
            if fold.start_year <= prior_end:
                raise ValueError("Outer viability folds must be ordered and disjoint.")
            prior_end = fold.end_year
        if not self.ridge_c_grid or any(value <= 0.0 for value in self.ridge_c_grid):
            raise ValueError("All ridge C candidates must be positive.")
        if tuple(sorted(set(self.ridge_c_grid))) != self.ridge_c_grid:
            raise ValueError("Ridge C grid must be unique and increasing.")
        return self


class DynamicsConfig(StrictModel):
    """Frozen P5 frontier, density, and migration-surprise protocol."""

    primary_model: Literal["combined_axes_logit"]
    survival_probability_threshold: float = Field(gt=0.0, lt=1.0)
    probability_clip: float = Field(gt=0.0, lt=0.01)
    cross_section_minimum: int = Field(ge=5)
    crowding_neighbors: int = Field(ge=2)
    successful_density_neighbors: int = Field(ge=2)
    successful_density_minimum: int = Field(ge=20)
    migration_ridge_alpha: float = Field(gt=0.0)
    migration_minimum_train_observations: int = Field(ge=100)
    recovery_primary_horizon_years: Literal[5]
    recovery_minimum_level_spearman: float = Field(ge=0.0, le=1.0)
    recovery_minimum_velocity_spearman: float = Field(ge=0.0, le=1.0)
    recovery_minimum_surprise_spearman: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_dynamics_protocol(self) -> "DynamicsConfig":
        """Keep the P5 primary surface aligned with the frozen P4 model."""

        if self.probability_clip >= 1.0 - self.survival_probability_threshold:
            raise ValueError("Probability clipping may not reach the frontier threshold.")
        if self.successful_density_neighbors >= self.successful_density_minimum:
            raise ValueError("Successful-density minimum must exceed its neighbor count.")
        return self


class SurvivalConfig(StrictModel):
    """Frozen P6 cause-specific survival and time-split protocol."""

    primary_horizon_years: Literal[5]
    time_origin_date: Literal["1900-01-01"]
    minimum_reporting_history: int = Field(ge=2)
    causes: tuple[Literal["performance_failure", "merger"], ...]
    features: tuple[
        Literal[
            "viability_log_odds",
            "velocity_log_odds",
            "acceleration_log_odds",
            "crowding_density",
            "historical_success_density",
            "log_market_cap",
            "book_leverage",
            "operating_margin",
        ],
        ...,
    ]
    ties: Literal["breslow"]
    outer_folds: tuple[ViabilityFoldConfig, ...]
    minimum_train_intervals: int = Field(ge=100)
    minimum_train_events: int = Field(ge=5)
    minimum_test_events: int = Field(ge=2)
    calibration: Literal["logistic_risk_score"]
    ph_diagnostic_alpha: float = Field(gt=0.0, lt=0.10)

    @model_validator(mode="after")
    def validate_survival_protocol(self) -> "SurvivalConfig":
        """Prevent cause, feature, or time-window specification search."""

        if self.causes != ("performance_failure", "merger"):
            raise ValueError("P6 must model performance failure then merger.")
        expected = (
            "viability_log_odds",
            "velocity_log_odds",
            "acceleration_log_odds",
            "crowding_density",
            "historical_success_density",
            "log_market_cap",
            "book_leverage",
            "operating_margin",
        )
        if self.features != expected:
            raise ValueError(f"P6 feature order must remain {expected!r}.")
        if not self.outer_folds:
            raise ValueError("P6 requires at least one time-split test window.")
        previous_end = -1
        for fold in self.outer_folds:
            if fold.start_year <= previous_end:
                raise ValueError("P6 outer folds must be ordered and disjoint.")
            previous_end = fold.end_year
        return self


class ReturnsConfig(StrictModel):
    """Frozen P7 target, inference, and gross-portfolio protocol."""

    primary_surface_horizon_years: Literal[5]
    horizons_months: tuple[Literal[1, 3, 6, 12], ...]
    primary_horizon_months: Literal[6]
    signals: tuple[
        Literal["viability_log_odds", "velocity_log_odds", "migration_surprise"],
        ...,
    ]
    primary_signal: Literal["migration_surprise"]
    controls: tuple[
        Literal[
            "log_market_cap",
            "book_to_market",
            "momentum_12_2",
            "operating_margin",
            "asset_growth",
            "book_leverage",
            "working_capital_assets",
            "operating_income_assets",
            "equity_to_liabilities",
            "sales_to_assets",
            "log_dollar_volume_12m",
            "realized_volatility_12m",
            "equity_beta_24m",
        ],
        ...,
    ]
    factor_columns: tuple[
        Literal["mkt_excess", "smb", "hml", "rmw", "cma", "mom"], ...
    ]
    entry_lag_months: Literal[1]
    require_complete_path_or_delist: bool
    missing_delisting_return_policy: Literal["missing_target"]
    winsor_lower: float = Field(ge=0.0, lt=0.5)
    winsor_upper: float = Field(gt=0.5, le=1.0)
    minimum_cross_section: int = Field(ge=20)
    minimum_fmb_months: int = Field(ge=12)
    quantiles: Literal[5]
    weighting_schemes: tuple[Literal["equal", "value"], ...]
    industry_neutral_modes: tuple[bool, ...]
    minimum_industry_group: int = Field(ge=5)
    overlapping_hac_rule: Literal["horizon_minus_one"]
    multiple_testing_method: Literal["holm"]
    synthetic_null_max_abs_primary_ic: float = Field(gt=0.0, le=0.10)
    synthetic_migration_min_primary_ic: float = Field(ge=0.0, le=0.10)
    synthetic_oracle_slope_lower: float
    synthetic_oracle_slope_upper: float

    @model_validator(mode="after")
    def validate_returns_protocol(self) -> "ReturnsConfig":
        """Prevent horizon, signal, control, or portfolio specification search."""

        if self.horizons_months != (1, 3, 6, 12):
            raise ValueError("P7 horizons must remain one, three, six, and twelve months.")
        if self.primary_horizon_months not in self.horizons_months:
            raise ValueError("Primary return horizon must be in the declared horizon set.")
        expected_signals = (
            "viability_log_odds",
            "velocity_log_odds",
            "migration_surprise",
        )
        if self.signals != expected_signals:
            raise ValueError(f"P7 signal order must remain {expected_signals!r}.")
        if self.primary_signal not in self.signals:
            raise ValueError("Primary P7 signal must be declared.")
        if self.weighting_schemes != ("equal", "value"):
            raise ValueError("P7 must report equal- and value-weighted portfolios.")
        if self.industry_neutral_modes != (False, True):
            raise ValueError("P7 must report raw and industry-neutral portfolios.")
        if not self.require_complete_path_or_delist:
            raise ValueError("Forward returns require a complete path or a valid delisting.")
        if self.synthetic_oracle_slope_lower >= self.synthetic_oracle_slope_upper:
            raise ValueError("Synthetic oracle slope bounds are reversed.")
        return self


class CostScenarioConfig(StrictModel):
    """One frozen one-way execution-cost sensitivity."""

    name: Literal["low", "conservative", "severe"]
    quoted_half_spread_multiplier: float = Field(ge=0.0, le=3.0)
    fixed_slippage_bps: float = Field(ge=0.0, le=100.0)


class CostsConfig(StrictModel):
    """Frozen P8 cost, borrow, capacity, and delay assumptions."""

    primary_horizon_months: Literal[6]
    primary_weighting: Literal["value"]
    primary_industry_neutral: Literal[True]
    primary_cost_scenario: Literal["conservative"]
    scenarios: tuple[CostScenarioConfig, ...]
    volume_unit_multiplier: float = Field(gt=0.0)
    trading_days_per_month: int = Field(ge=15, le=31)
    portfolio_notional_millions_per_leg: float = Field(gt=0.0)
    max_adv_participation: float = Field(gt=0.0, le=0.25)
    max_market_cap_fraction: float = Field(gt=0.0, le=0.01)
    max_position_weight: float = Field(gt=0.0, le=0.25)
    max_half_spread: float = Field(gt=0.0, le=0.10)
    fallback_half_spread_bps: float = Field(gt=0.0, le=500.0)
    minimum_short_market_cap_millions: float = Field(gt=0.0)
    minimum_short_adv_millions: float = Field(gt=0.0)
    hard_borrow_market_cap_millions: float = Field(gt=0.0)
    hard_borrow_adv_millions: float = Field(gt=0.0)
    borrow_annual_bps: float = Field(ge=0.0, le=5000.0)
    hard_borrow_annual_bps: float = Field(ge=0.0, le=10000.0)
    delayed_execution_months: Literal[1]

    @model_validator(mode="after")
    def validate_cost_protocol(self) -> "CostsConfig":
        """Prevent post-P7 parameter search in the implementability gate."""

        expected = ("low", "conservative", "severe")
        if tuple(item.name for item in self.scenarios) != expected:
            raise ValueError(f"P8 cost scenarios must remain {expected!r}.")
        if self.primary_horizon_months != 6:
            raise ValueError("P8 primary horizon must remain six months.")
        if self.hard_borrow_annual_bps < self.borrow_annual_bps:
            raise ValueError("Hard-borrow cost must not be below ordinary borrow cost.")
        if self.hard_borrow_market_cap_millions < self.minimum_short_market_cap_millions:
            raise ValueError("Hard-borrow size threshold must exceed the short exclusion.")
        if self.hard_borrow_adv_millions < self.minimum_short_adv_millions:
            raise ValueError("Hard-borrow liquidity threshold must exceed the exclusion.")
        return self


class ClusteringConfig(StrictModel):
    """Frozen descriptive P9 archetype protocol."""

    representation: Literal["anchored"]
    training_end_year: int = Field(ge=1980, le=2015)
    maximum_training_rows: int = Field(ge=1000)
    minimum_cluster_size: int = Field(ge=20)
    minimum_samples: int = Field(ge=5)
    assignment_radius_quantile: float = Field(gt=0.50, lt=1.0)
    stability_repetitions: int = Field(ge=2, le=20)
    stability_sample_fraction: float = Field(gt=0.50, lt=1.0)

    @model_validator(mode="after")
    def validate_clustering_protocol(self) -> "ClusteringConfig":
        """Keep P9 descriptive and computationally bounded."""

        if self.representation != "anchored":
            raise ValueError("P9 uses the historically anchored representation.")
        if self.maximum_training_rows < self.minimum_cluster_size * 5:
            raise ValueError("P9 training cap is too small for the cluster-size rule.")
        return self


class AuditConfig(StrictModel):
    """Reproducibility and write-safety controls."""

    hash_algorithm: Literal["sha256"]
    atomic_writes: bool
    archive_completed_real_runs: bool
    record_package_versions: bool
    record_input_hashes: bool


class HypercubeConfig(StrictModel):
    """Resolved and validated project configuration."""

    project: ProjectConfig
    paths: PathsConfig
    data: DataConfig
    synthetic: SyntheticConfig
    availability: AvailabilityConfig
    universe: UniverseConfig
    panel: PanelConfig
    axes: AxesConfig
    standardization: StandardizationConfig
    innovation: InnovationConfig
    viability: ViabilityConfig
    dynamics: DynamicsConfig
    survival: SurvivalConfig
    returns: ReturnsConfig
    costs: CostsConfig
    clustering: ClusteringConfig
    audit: AuditConfig

    @model_validator(mode="after")
    def validate_phase_boundary(self) -> "HypercubeConfig":
        """Keep generation and panel construction inside their phases."""

        if self.project.phase == "P0" and self.data.generate_synthetic:
            raise ValueError("Synthetic generation is not authorized in P0.")
        if (
            self.project.phase == "P1"
            and self.data.mode == "synthetic"
            and not self.data.generate_synthetic
        ):
            raise ValueError("P1 synthetic mode must enable generation.")
        if self.project.phase in {
            "P2",
            "P3",
            "P4",
            "P5",
            "P6",
            "P7",
            "P8",
            "P9",
            "P10",
        } and self.data.generate_synthetic:
            raise ValueError(
                f"{self.project.phase} consumes frozen inputs; it may not regenerate them."
            )
        return self


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge a configuration overlay without mutating inputs."""

    merged = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_mapping(path: Path, seen: frozenset[Path]) -> dict[str, Any]:
    """Load YAML and resolve a local single-parent configuration overlay."""

    resolved = path.resolve()
    if resolved in seen:
        raise ValueError(f"Configuration inheritance cycle at {resolved}.")
    if not resolved.is_file():
        raise FileNotFoundError(f"Configuration file not found: {resolved}")
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping: {resolved}")
    parent = raw.pop("extends", None)
    if parent is None:
        return raw
    if not isinstance(parent, str):
        raise ValueError("extends must be a relative YAML path.")
    parent_path = (resolved.parent / parent).resolve()
    base = _load_mapping(parent_path, seen | {resolved})
    return _deep_merge(base, raw)


def load_config(path: str | Path) -> HypercubeConfig:
    """Load, merge, and strictly validate a YAML configuration."""

    return HypercubeConfig.model_validate(_load_mapping(Path(path), frozenset()))


def resolved_config(path: str | Path) -> dict[str, Any]:
    """Return a JSON-compatible resolved configuration mapping."""

    return load_config(path).model_dump(mode="json")
