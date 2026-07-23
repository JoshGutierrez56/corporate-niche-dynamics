"""P4 failure-label, nested-time, calibration, and model-ladder gates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hypercube.config import HypercubeConfig, load_config
from hypercube.labels import construct_fixed_horizon_labels
from hypercube.viability import (
    MODEL_ORDER,
    _fold_frames,
    fold_definitions,
    prediction_metrics,
    run_nested_viability,
)


ROOT = Path(__file__).resolve().parents[1]


def _write_label_fixture(root: Path) -> pd.DataFrame:
    gvkeys = ["A", "B", "C", "D"]
    permnos = [1, 2, 3, 4]
    links = pd.DataFrame(
        {
            "gvkey": gvkeys,
            "lpermno": permnos,
            "linkdt": pd.to_datetime(["2000-01-01"] * 4),
            "linkenddt": pd.to_datetime([None] * 4),
            "linktype": ["LU"] * 4,
            "linkprim": ["P"] * 4,
        }
    )
    months = []
    for permno in (1, 2, 3):
        for date in pd.date_range("2000-01-31", "2010-12-31", freq="ME"):
            months.append({"permno": permno, "date": date})
    for date in pd.date_range("2000-01-31", "2002-12-31", freq="ME"):
        months.append({"permno": 4, "date": date})
    delist = pd.DataFrame(
        {
            "permno": [1, 2],
            "dlstdt": pd.to_datetime(["2002-06-15", "2003-06-15"]),
            "dlstcd": [574, 241],
            "dlret": [-0.8, 0.1],
            "exit_category": ["performance_failure", "merger"],
        }
    )
    links.to_parquet(root / "ccm_link.parquet", index=False)
    pd.DataFrame(months).to_parquet(root / "crsp_monthly.parquet", index=False)
    delist.to_parquet(root / "crsp_delist.parquet", index=False)
    return pd.DataFrame(
        {
            "permno": permnos,
            "gvkey": gvkeys,
            "datadate": pd.to_datetime(["2000-12-31"] * 4),
            "fyear": [2000] * 4,
            "availability_date": pd.to_datetime(["2001-01-15"] * 4),
            "formation_date": pd.to_datetime(["2001-01-31"] * 4),
            "feature_date": pd.to_datetime(["2001-01-31"] * 4),
        }
    )


def test_fixed_horizon_labels_separate_failure_competing_and_censoring(
    tmp_path: Path,
) -> None:
    """Observed failures, competing exits, survivors, and lost follow-up differ."""

    events = _write_label_fixture(tmp_path)
    labels = construct_fixed_horizon_labels(events, tmp_path, (3,)).set_index("gvkey")
    assert labels.loc["A", "failure_within_horizon"] == 1
    assert labels.loc["A", "label_observed_date"] == pd.Timestamp("2002-06-15")
    assert pd.isna(labels.loc["B", "failure_within_horizon"])
    assert labels.loc["B", "censor_reason"] == "competing_exit"
    assert labels.loc["C", "failure_within_horizon"] == 0
    assert pd.isna(labels.loc["D", "failure_within_horizon"])
    assert labels.loc["D", "censor_reason"] == "lost_followup"


def _compact_config() -> HypercubeConfig:
    config = load_config(ROOT / "configs" / "synthetic.yaml")
    viability = config.viability.model_copy(
        update={
            "outer_folds": (config.viability.outer_folds[0],),
            "minimum_train_observations": 100,
            "minimum_train_failures": 5,
            "minimum_validation_failures": 2,
            "ridge_c_grid": (0.1, 1.0),
            "tree_max_iter": 30,
            "tree_min_samples_leaf": 20,
        }
    )
    return config.model_copy(update={"viability": viability})


def _model_matrix() -> pd.DataFrame:
    rng = np.random.default_rng(4107)
    rows = []
    relative = (
        "relative_demand_strength_pricing_power",
        "relative_competitive_defensibility",
        "relative_innovation_intensity",
        "relative_go_to_market_efficiency",
        "relative_unit_economics_profit_quality",
        "relative_scalability_capital_efficiency",
    )
    anchored = tuple(item.replace("relative_", "anchored_") for item in relative)
    for year in range(1975, 2020):
        for firm in range(40):
            latent = rng.normal()
            failure = int(firm % 10 == 0 or (latent < -1.7 and firm % 3 == 0))
            for horizon in (3, 5):
                feature_date = pd.Timestamp(year, 6, 30)
                row = {
                    "permno": year * 100 + firm,
                    "gvkey": f"{year}-{firm}",
                    "datadate": pd.Timestamp(year - 1, 12, 31),
                    "fyear": year - 1,
                    "feature_date": feature_date,
                    "horizon_end": feature_date + pd.DateOffset(years=horizon),
                    "horizon_years": horizon,
                    "failure_within_horizon": failure,
                    "label_observed_date": feature_date + pd.DateOffset(years=horizon),
                    "event_year": year,
                    "sic2": 20 + firm % 8,
                    "operating_margin": latent + rng.normal(0, 0.2),
                    "gross_profitability_lag_assets": latent + rng.normal(0, 0.2),
                    "log_market_cap": 5.0 + latent,
                    "book_leverage": 0.4 - 0.1 * latent,
                    "working_capital_assets": 0.1 + 0.05 * latent,
                    "operating_income_assets": 0.08 + 0.03 * latent,
                    "equity_to_liabilities": 0.8 + 0.2 * latent,
                    "sales_to_assets": 1.0 + 0.1 * latent,
                }
                for index, feature in enumerate(relative):
                    row[feature] = latent + rng.normal(0, 0.3) + index * 0.01
                for index, feature in enumerate(anchored):
                    row[feature] = latent + rng.normal(0, 0.4) + index * 0.01
                rows.append(row)
    return pd.DataFrame(rows)


def test_fold_contract_purges_unobservable_training_labels() -> None:
    """Every training/validation label exists before the next split begins."""

    config = _compact_config()
    matrix = _model_matrix()
    for horizon in (3, 5):
        definition = fold_definitions(config, horizon)[0]
        train, validation, test = _fold_frames(
            matrix.loc[matrix["horizon_years"] == horizon], definition
        )
        assert train["label_observed_date"].max() < pd.Timestamp(
            definition["validation_start_year"], 1, 1
        )
        assert validation["label_observed_date"].max() < pd.Timestamp(
            definition["test_start_year"], 1, 1
        )
        assert test["event_year"].between(2000, 2004).all()


def test_nested_ladder_writes_all_models_and_bounded_probabilities(
    tmp_path: Path,
) -> None:
    """The compact P4 ladder fits without random splitting or return outputs."""

    predictions, metrics, trials, definitions = run_nested_viability(
        _model_matrix(), _compact_config(), tmp_path / "models"
    )
    assert set(predictions["model"]) == set(MODEL_ORDER)
    assert set(metrics["model"]) == set(MODEL_ORDER)
    assert len(definitions) == 2
    assert predictions["predicted_failure_probability"].between(0, 1).all()
    assert np.allclose(
        predictions["calibrated_survival_probability"],
        1 - predictions["predicted_failure_probability"],
    )
    assert set(trials["candidate"]).issuperset({"C=0.1", "C=1"})
    assert len(list((tmp_path / "models").glob("*.joblib"))) == 14


def test_prediction_metrics_are_deterministic() -> None:
    """Saved-probability validation can recompute every P4 metric."""

    y = np.array([0, 0, 1, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.3, 0.7, 0.9])
    first = prediction_metrics(y, p)
    second = prediction_metrics(y, p)
    assert first == second
    assert first["roc_auc"] == 1.0
    assert first["brier_score"] < 0.1
