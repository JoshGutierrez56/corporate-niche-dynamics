"""P5 frontier, model-vintage, density, and residualization gates."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from hypercube.config import HypercubeConfig, load_config
from hypercube.dynamics import (
    NUMERIC_MIGRATION_CONTROLS,
    _neighbor_density,
    add_time_dynamics,
    residualize_migration,
    select_oos_surface,
)
from hypercube.viability import ANCHORED_AXES, DISTRESS_FEATURES, PROFITABILITY_FEATURES, RELATIVE_AXES


ROOT = Path(__file__).resolve().parents[1]


def _compact_config() -> HypercubeConfig:
    config = load_config(ROOT / "configs" / "synthetic.yaml")
    dynamics = config.dynamics.model_copy(
        update={"migration_minimum_train_observations": 100}
    )
    return config.model_copy(update={"dynamics": dynamics})


def _dynamic_fixture() -> pd.DataFrame:
    rows = []
    levels = (0.90, 0.92, 0.94, 0.93, 0.96)
    folds = (1, 1, 1, 2, 2)
    for year, level, fold in zip(range(2000, 2005), levels, folds, strict=True):
        rows.append(
            {
                "gvkey": "A",
                "datadate": pd.Timestamp(year - 1, 12, 31),
                "fyear": year - 1,
                "horizon_years": 5,
                "feature_date": pd.Timestamp(year, 6, 30),
                "fold": fold,
                "viability_level": level,
                "viability_log_odds": np.log(level / (1.0 - level)),
                "above_frontier": level >= 0.95,
                "crowding_density": 0.4 + 0.01 * (year - 2000),
            }
        )
    return pd.DataFrame(rows)


def test_dynamics_do_not_cross_outer_model_vintages() -> None:
    """Velocity, acceleration, and crossings are absent at a P4 fold boundary."""

    result = add_time_dynamics(_dynamic_fixture()).sort_values("feature_date")
    assert pd.isna(result.iloc[0]["velocity_log_odds"])
    assert result.iloc[1]["velocity_log_odds"] > 0.0
    assert np.isfinite(result.iloc[2]["acceleration_log_odds"])
    assert bool(result.iloc[3]["model_vintage_changed"])
    assert pd.isna(result.iloc[3]["velocity_log_odds"])
    assert pd.isna(result.iloc[3]["frontier_crossing"])
    assert result.iloc[4]["frontier_crossing"] == 1


def test_neighbor_density_excludes_the_query_itself() -> None:
    """Contemporaneous crowding uses other firms rather than zero self-distance."""

    columns = [
        "relative_demand_strength_pricing_power",
        "relative_competitive_defensibility",
        "relative_innovation_intensity",
        "relative_go_to_market_efficiency",
        "relative_unit_economics_profit_quality",
        "relative_scalability_capital_efficiency",
    ]
    reference = pd.DataFrame(
        {
            "gvkey": ["A", "B", "C"],
            "datadate": pd.to_datetime(["2000-12-31"] * 3),
            **{column: [0.0, 1.0, 2.0] for column in columns},
        }
    )
    density = _neighbor_density(
        reference,
        reference.iloc[[0]],
        neighbors=2,
        exclude_self=True,
    )
    assert density.shape == (1,)
    assert 0.0 < density[0] < 1.0
    missing_query = reference.iloc[[0]].copy()
    missing_query[columns] = np.nan
    assert np.isnan(
        _neighbor_density(
            reference, missing_query, neighbors=2, exclude_self=True
        )[0]
    )


def _migration_fixture() -> pd.DataFrame:
    rng = np.random.default_rng(415)
    rows = []
    for year in range(2000, 2005):
        for firm in range(60):
            level = rng.normal()
            row = {
                "gvkey": f"{firm:03d}",
                "datadate": pd.Timestamp(year - 1, 12, 31),
                "fyear": year - 1,
                "horizon_years": 5,
                "feature_date": pd.Timestamp(year, 6, 30),
                "feature_year": year,
                "velocity_log_odds": 0.2 * level + rng.normal(0.0, 0.1),
                "sic2": 20 + firm % 3,
            }
            for index, column in enumerate(NUMERIC_MIGRATION_CONTROLS):
                row[column] = level + index * 0.01 + rng.normal(0.0, 0.05)
            rows.append(row)
    return pd.DataFrame(rows)


def test_migration_expectations_train_on_prior_calendar_years_only(
    tmp_path: Path,
) -> None:
    """An expanding migration model cannot see its prediction year."""

    frame, diagnostics = residualize_migration(
        _migration_fixture(), _compact_config(), tmp_path / "models"
    )
    fitted = diagnostics.loc[diagnostics["status"].eq("fitted")]
    assert not fitted.empty
    assert (
        pd.to_datetime(fitted["train_max_feature_date"])
        < pd.to_datetime(fitted["prediction_year"].astype(str) + "-01-01")
    ).all()
    available = frame.loc[frame["migration_surprise"].notna()]
    assert not available.empty
    assert np.allclose(
        available["migration_surprise"],
        available["velocity_log_odds"] - available["expected_velocity_log_odds"],
    )
    assert list((tmp_path / "models").glob("*.joblib"))


def test_surface_scores_censored_outer_test_events(tmp_path: Path) -> None:
    """P5 population membership cannot depend on eventually observed labels."""

    config = _compact_config()
    features = [*RELATIVE_AXES, *ANCHORED_AXES]
    estimator = LogisticRegression().fit(
        pd.DataFrame(
            [
                {feature: -1.0 for feature in features},
                {feature: 1.0 for feature in features},
            ]
        ),
        [0, 1],
    )
    rows = []
    models = tmp_path / "models"
    models.mkdir()
    for horizon in config.viability.horizons_years:
        for fold_index, fold in enumerate(config.viability.outer_folds, start=1):
            for firm, outcome in ((0, 0), (1, pd.NA)):
                year = fold.start_year
                row = {
                    "permno": horizon * 10000 + fold_index * 10 + firm,
                    "gvkey": f"h{horizon}f{fold_index}i{firm}",
                    "datadate": pd.Timestamp(year - 1, 12, 31),
                    "fyear": year - 1,
                    "feature_date": pd.Timestamp(year, 6, 30),
                    "horizon_end": pd.Timestamp(year + horizon, 6, 30),
                    "horizon_years": horizon,
                    "failure_within_horizon": outcome,
                    "availability_date": pd.Timestamp(year, 4, 1),
                    "formation_date": pd.Timestamp(year, 4, 30),
                    "label_observed_date": (
                        pd.Timestamp(year + horizon, 6, 30)
                        if firm == 0
                        else pd.NaT
                    ),
                    "sic2": 20,
                    "market_cap_millions": 100.0,
                    "event_year": year,
                }
                row.update({feature: float(firm) for feature in features})
                row.update({feature: 0.1 for feature in PROFITABILITY_FEATURES})
                row.update({feature: 0.1 for feature in DISTRESS_FEATURES})
                rows.append(row)
            artifact = {
                "horizon_years": horizon,
                "model": config.dynamics.primary_model,
                "fold": {"fold": fold_index},
                "fitted": {
                    "features": features,
                    "estimator": estimator,
                    "calibrator": {"kind": "identity"},
                },
            }
            joblib.dump(
                artifact,
                models
                / f"h{horizon}_fold{fold_index}_{config.dynamics.primary_model}.joblib",
            )
    pd.DataFrame(rows).to_parquet(tmp_path / "model_matrix.parquet", index=False)
    surface = select_oos_surface(tmp_path, config)
    assert len(surface) == 16
    assert surface["failure_within_horizon"].isna().sum() == 8
    assert surface["viability_level"].between(0.0, 1.0).all()
