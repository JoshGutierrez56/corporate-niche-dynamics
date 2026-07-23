"""Deterministic, traceable P10 figures and closeout statistics."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA

from hypercube.clustering import AXIS_COLUMNS
from hypercube.config import HypercubeConfig
from hypercube.data import SCENARIOS, atomic_write_json


class VisualizationError(ValueError):
    """Raised when a P10 figure cannot be traced or validated."""


P10_VERSION = "hypercube-final-figures-v1"
PRIMARY_SCENARIO = "migration_alpha"
RELATIVE_AXES = tuple(column.replace("anchored_", "relative_") for column in AXIS_COLUMNS)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.to_csv(name, index=False, lineterminator="\n")
        os.replace(name, path)
    except Exception:
        Path(name).unlink(missing_ok=True)
        raise


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=path.suffix, dir=path.parent
    )
    os.close(descriptor)
    try:
        fig.savefig(name, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        with Image.open(name) as image:
            image.verify()
        os.replace(name, path)
    except Exception:
        plt.close(fig)
        Path(name).unlink(missing_ok=True)
        raise


def _primary_mask(frame: pd.DataFrame) -> pd.Series:
    neutral = frame["industry_neutral"].astype(str).str.lower().eq("true")
    return (
        frame["horizon_months"].eq(6)
        & frame["signal"].eq("migration_surprise")
        & frame["weighting"].eq("value")
        & neutral
    )


def _scenario_root(project_root: Path, scenario: str) -> Path:
    return project_root / "data" / "processed" / "synthetic" / scenario


def _load_primary(project_root: Path) -> dict[str, pd.DataFrame]:
    root = _scenario_root(project_root, PRIMARY_SCENARIO)
    return {
        "axes": pd.read_parquet(root / "p3" / "axis_scores.parquet"),
        "predictions": pd.read_parquet(root / "p4" / "oos_predictions.parquet"),
        "dynamics": pd.read_parquet(root / "p5" / "frontier_dynamics.parquet"),
        "p7_monthly": pd.read_csv(root / "p7" / "portfolio_monthly_returns.csv"),
        "p7_factors": pd.read_csv(root / "p7" / "portfolio_factor_results.csv"),
        "p8_monthly": pd.read_csv(root / "p8" / "cost_aware_monthly_returns.csv"),
        "p8_waterfall": pd.read_csv(root / "p8" / "cost_waterfall.csv"),
        "assignments": pd.read_parquet(root / "p9" / "archetype_assignments.parquet"),
        "transitions": pd.read_csv(root / "p9" / "transition_matrix.csv"),
    }


def _axis_distribution_figure(
    axes: pd.DataFrame,
    tables_dir: Path,
    figures_dir: Path,
) -> tuple[Path, list[str]]:
    sample = axes[list(AXIS_COLUMNS)].copy()
    summary = sample.describe(percentiles=[0.01, 0.25, 0.5, 0.75, 0.99]).T
    summary.insert(0, "axis", summary.index)
    summary = summary.reset_index(drop=True)
    correlation = sample.corr()
    correlation.insert(0, "axis", correlation.index)
    summary_path = tables_dir / "p10_axis_distribution_summary.csv"
    correlation_path = tables_dir / "p10_axis_correlation_matrix.csv"
    _atomic_csv(summary, summary_path)
    _atomic_csv(correlation.reset_index(drop=True), correlation_path)
    fig, axes_plot = plt.subplots(2, 4, figsize=(15, 7))
    for column, axis in zip(AXIS_COLUMNS, axes_plot.flat[:6], strict=True):
        axis.hist(sample[column].dropna(), bins=40, color="#3a6ea5", alpha=0.85)
        axis.set_title(column.replace("anchored_", "").replace("_", " "), fontsize=8)
        axis.set_ylabel("Events")
    axes_plot[1, 2].axis("off")
    image_axis = axes_plot[1, 3]
    matrix = sample.corr().to_numpy()
    image = image_axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
    image_axis.set_title("Axis correlation")
    image_axis.set_xticks(range(6), range(1, 7))
    image_axis.set_yticks(range(6), range(1, 7))
    fig.colorbar(image, ax=image_axis, fraction=0.046)
    fig.suptitle("Anchored six-axis distributions and correlation — synthetic")
    path = figures_dir / "01_axis_distributions_and_correlation.png"
    _save_figure(fig, path)
    return path, [str(summary_path), str(correlation_path)]


def _pca_figure(
    axes: pd.DataFrame,
    assignments: pd.DataFrame,
    dynamics: pd.DataFrame,
    config: HypercubeConfig,
    tables_dir: Path,
    figures_dir: Path,
) -> tuple[Path, list[str]]:
    frame = assignments.merge(
        dynamics.loc[
            dynamics["horizon_years"].eq(5),
            ["gvkey", "datadate", "feature_date", "viability_level"],
        ],
        on=["gvkey", "datadate", "feature_date"],
        how="left",
        validate="one_to_one",
    )
    train = frame.loc[frame["fyear"].le(config.clustering.training_end_year)]
    median = train[list(AXIS_COLUMNS)].median()
    scale = (1.4826 * (train[list(AXIS_COLUMNS)] - median).abs().median()).replace(
        0.0, 1.0
    )
    transformed = (
        (frame[list(AXIS_COLUMNS)].fillna(median) - median) / scale
    ).clip(-8, 8)
    pca = PCA(n_components=2, random_state=config.project.seed)
    pca.fit(transformed.loc[train.index])
    coordinates = pca.transform(transformed)
    rng = np.random.default_rng(config.project.seed)
    selected = np.sort(
        rng.choice(
            len(frame),
            size=min(6000, len(frame)),
            replace=False,
        )
    )
    plot = frame.iloc[selected][
        ["gvkey", "feature_date", "archetype", "viability_level"]
    ].copy()
    plot["pc1"] = coordinates[selected, 0]
    plot["pc2"] = coordinates[selected, 1]
    table_path = tables_dir / "p10_pca_projection_sample.csv"
    _atomic_csv(plot, table_path)
    fig, figure_axes = plt.subplots(1, 2, figsize=(13, 5))
    scatter = figure_axes[0].scatter(
        plot["pc1"],
        plot["pc2"],
        c=plot["viability_level"],
        s=5,
        alpha=0.5,
        cmap="viridis",
    )
    figure_axes[0].set_title("PCA colored by OOS viability")
    fig.colorbar(scatter, ax=figure_axes[0], label="Survival probability")
    labels = sorted(plot["archetype"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(labels))))
    for label, color in zip(labels, colors, strict=True):
        group = plot.loc[plot["archetype"].eq(label)]
        figure_axes[1].scatter(
            group["pc1"], group["pc2"], s=5, alpha=0.5, label=label, color=color
        )
    figure_axes[1].set_title("Same PCA colored by P9 archetype")
    figure_axes[1].legend(fontsize=7, markerscale=2)
    for axis in figure_axes:
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
    fig.suptitle("Two-dimensional projection of the six-dimensional state")
    path = figures_dir / "02_pca_viability_and_archetypes.png"
    _save_figure(fig, path)
    return path, [str(table_path)]


def _calibration_figure(
    predictions: pd.DataFrame,
    tables_dir: Path,
    figures_dir: Path,
) -> tuple[Path, list[str]]:
    frame = predictions.loc[
        predictions["horizon_years"].eq(5)
        & predictions["model"].eq("combined_axes_logit")
        & predictions["failure_within_horizon"].notna()
    ].copy()
    frame["predicted_survival"] = frame["calibrated_survival_probability"]
    frame["observed_survival"] = 1.0 - frame["failure_within_horizon"].astype(float)
    frame["bin"] = pd.qcut(
        frame["predicted_survival"], q=10, duplicates="drop"
    )
    curve = (
        frame.groupby("bin", observed=True)
        .agg(
            observations=("observed_survival", "size"),
            mean_predicted_survival=("predicted_survival", "mean"),
            observed_survival_rate=("observed_survival", "mean"),
        )
        .reset_index()
    )
    curve["bin"] = curve["bin"].astype(str)
    table_path = tables_dir / "p10_calibration_curve.csv"
    _atomic_csv(curve, table_path)
    fig, axis = plt.subplots(figsize=(6, 5))
    axis.plot([0, 1], [0, 1], linestyle="--", color="black", label="Perfect")
    axis.plot(
        curve["mean_predicted_survival"],
        curve["observed_survival_rate"],
        marker="o",
        color="#3a6ea5",
        label="Combined axes",
    )
    axis.set_xlabel("Mean predicted five-year survival")
    axis.set_ylabel("Observed five-year survival")
    axis.set_title("Out-of-sample calibration — synthetic")
    axis.legend()
    path = figures_dir / "03_calibrated_viability_curve.png"
    _save_figure(fig, path)
    return path, [str(table_path)]


def _margin_figure(
    dynamics: pd.DataFrame,
    tables_dir: Path,
    figures_dir: Path,
) -> tuple[Path, list[str]]:
    frame = dynamics.loc[
        dynamics["horizon_years"].eq(5)
        & dynamics["failure_within_horizon"].notna()
    ].copy()
    table = (
        frame.groupby("failure_within_horizon", observed=True)[
            "viability_margin_log_odds"
        ]
        .describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9])
        .reset_index()
    )
    table_path = tables_dir / "p10_viability_margin_by_outcome.csv"
    _atomic_csv(table, table_path)
    fig, axis = plt.subplots(figsize=(7, 5))
    data = [
        frame.loc[
            frame["failure_within_horizon"].eq(value),
            "viability_margin_log_odds",
        ].dropna()
        for value in (0.0, 1.0)
    ]
    axis.hist(
        data,
        bins=50,
        density=True,
        alpha=0.6,
        label=["No performance failure", "Performance failure"],
        color=["#3a6ea5", "#b53a3a"],
    )
    axis.axvline(0.0, color="black", linestyle="--")
    axis.set_xlabel("Viability frontier margin (log odds)")
    axis.set_ylabel("Density")
    axis.set_title("Viability margin by five-year outcome")
    axis.legend()
    path = figures_dir / "04_viability_margin_by_outcome.png"
    _save_figure(fig, path)
    return path, [str(table_path)]


def _selected_firm_figure(
    dynamics: pd.DataFrame,
    tables_dir: Path,
    figures_dir: Path,
) -> tuple[Path, list[str]]:
    frame = dynamics.loc[dynamics["horizon_years"].eq(5)].copy()
    counts = frame.groupby("gvkey", observed=True)["feature_date"].transform("size")
    eligible = frame.loc[
        counts.ge(3)
        & frame["migration_surprise"].notna()
        & frame["market_cap_millions"].ge(frame["market_cap_millions"].median())
    ]
    positive = eligible.nlargest(1, "migration_surprise").iloc[0]
    negative = eligible.nsmallest(1, "migration_surprise").iloc[0]
    selection = pd.DataFrame(
        [
            {
                "selection_rule": "largest_positive_migration_surprise_among_liquid_firms",
                "gvkey": positive["gvkey"],
                "feature_date": positive["feature_date"],
                "migration_surprise": positive["migration_surprise"],
                "market_cap_millions": positive["market_cap_millions"],
            },
            {
                "selection_rule": "largest_negative_migration_surprise_among_liquid_firms",
                "gvkey": negative["gvkey"],
                "feature_date": negative["feature_date"],
                "migration_surprise": negative["migration_surprise"],
                "market_cap_millions": negative["market_cap_millions"],
            },
        ]
    )
    table_path = tables_dir / "p10_selected_firm_examples.csv"
    _atomic_csv(selection, table_path)
    fig, axes_plot = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    metrics = (
        ("viability_level", "Viability level"),
        ("velocity_log_odds", "Velocity (log odds)"),
        ("acceleration_log_odds", "Acceleration (log odds)"),
    )
    for label, row, color in (
        ("Largest positive surprise", positive, "#2b8c4b"),
        ("Largest negative surprise", negative, "#b53a3a"),
    ):
        history = frame.loc[frame["gvkey"].eq(row["gvkey"])].sort_values(
            "feature_date"
        )
        for axis, (metric, title) in zip(axes_plot, metrics, strict=True):
            axis.plot(history["feature_date"], history[metric], marker="o", label=label, color=color)
            axis.set_ylabel(title)
    axes_plot[0].legend()
    axes_plot[-1].set_xlabel("Feature availability date")
    fig.suptitle("Predeclared synthetic firm examples; not cherry-picked")
    path = figures_dir / "05_niche_dynamics_examples.png"
    _save_figure(fig, path)
    return path, [str(table_path)]


def _transition_figure(
    transitions: pd.DataFrame,
    tables_dir: Path,
    figures_dir: Path,
) -> tuple[Path, list[str]]:
    frame = transitions.loc[transitions["sample_role"].eq("out_of_sample")]
    matrix = frame.pivot(
        index="prior_archetype",
        columns="archetype",
        values="transition_probability",
    ).fillna(0.0)
    table = matrix.copy()
    table.insert(0, "prior_archetype", table.index)
    table_path = tables_dir / "p10_oos_transition_matrix.csv"
    _atomic_csv(table.reset_index(drop=True), table_path)
    fig, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix.to_numpy(), vmin=0, vmax=1, cmap="Blues")
    axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    axis.set_yticks(range(len(matrix.index)), matrix.index)
    axis.set_title("Out-of-sample archetype transition probabilities")
    fig.colorbar(image, ax=axis, label="Probability")
    path = figures_dir / "06_archetype_transition_matrix.png"
    _save_figure(fig, path)
    return path, [str(table_path)]


def _state_evolution_figure(
    axes: pd.DataFrame,
    tables_dir: Path,
    figures_dir: Path,
) -> tuple[Path, list[str]]:
    frame = axes.copy()
    frame["decade"] = (frame["fyear"] // 10 * 10).astype(int)
    summary = (
        frame.groupby("decade", observed=True)[list(AXIS_COLUMNS)]
        .mean()
        .reset_index()
    )
    table_path = tables_dir / "p10_anchored_state_by_decade.csv"
    _atomic_csv(summary, table_path)
    fig, axis = plt.subplots(figsize=(10, 5))
    for column in AXIS_COLUMNS:
        axis.plot(
            summary["decade"],
            summary[column],
            marker="o",
            label=column.replace("anchored_", "").replace("_", " "),
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xlabel("Decade")
    axis.set_ylabel("Mean anchored score")
    axis.set_title("Historically anchored state-space evolution")
    axis.legend(fontsize=7, ncol=2)
    path = figures_dir / "07_anchored_state_space_evolution.png"
    _save_figure(fig, path)
    return path, [str(table_path)]


def _cell_occupancy_figure(
    axes: pd.DataFrame,
    tables_dir: Path,
    figures_dir: Path,
) -> tuple[Path, list[str]]:
    frame = axes.copy()
    codes = []
    for column in RELATIVE_AXES:
        values = pd.to_numeric(frame[column], errors="coerce")
        codes.append(np.select([values.lt(-0.5), values.gt(0.5)], [0, 2], default=1))
    code_matrix = np.stack(codes, axis=1)
    frame["cell_id"] = ["".join(str(int(value)) for value in row) for row in code_matrix]
    occupancy = (
        frame.groupby("cell_id", observed=True)
        .agg(observations=("gvkey", "size"), firms=("gvkey", "nunique"))
        .reset_index()
        .sort_values("observations", ascending=False)
    )
    occupancy["share"] = occupancy["observations"] / occupancy["observations"].sum()
    table_path = tables_dir / "p10_729_cell_occupancy.csv"
    _atomic_csv(occupancy, table_path)
    top = occupancy.head(40).sort_values("observations")
    fig, axis = plt.subplots(figsize=(10, 8))
    axis.barh(top["cell_id"], top["observations"], color="#6c7a89")
    axis.set_xlabel("Synthetic firm-year events")
    axis.set_ylabel("Six ternary digits (low/middle/high)")
    axis.set_title("Top occupied cells — secondary 729-cell description")
    path = figures_dir / "08_729_cell_occupancy_secondary.png"
    _save_figure(fig, path)
    return path, [str(table_path)]


def _cost_figure(
    p7_monthly: pd.DataFrame,
    p8_waterfall: pd.DataFrame,
    tables_dir: Path,
    figures_dir: Path,
) -> tuple[Path, list[str]]:
    p7 = p7_monthly.loc[_primary_mask(p7_monthly)].copy()
    p7["holding_date"] = pd.to_datetime(p7["holding_date"])
    p7["cumulative_gross"] = (1.0 + p7["long_short_return"]).cumprod() - 1.0
    waterfall = p8_waterfall.loc[
        _primary_mask(p8_waterfall)
        & p8_waterfall["delay_months"].eq(0)
    ].copy()
    table_path = tables_dir / "p10_primary_cost_waterfall.csv"
    _atomic_csv(waterfall, table_path)
    fig, axes_plot = plt.subplots(1, 3, figsize=(15, 4.5))
    axes_plot[0].plot(p7["holding_date"], p7["cumulative_gross"], color="#3a6ea5")
    axes_plot[0].set_title("P7 cumulative gross spread")
    axes_plot[0].set_ylabel("Cumulative return")
    axes_plot[1].plot(p7["holding_date"], p7["turnover"], color="#8c5a2b")
    axes_plot[1].set_title("Monthly turnover")
    axes_plot[1].set_ylabel("Turnover")
    cases = waterfall.sort_values("cost_scenario")
    axes_plot[2].bar(
        cases["cost_scenario"],
        cases["annualized_gross_capacity_return"],
        label="Gross after capacity",
        color="#6aaed6",
    )
    axes_plot[2].bar(
        cases["cost_scenario"],
        cases["annualized_net_return"],
        label="Net",
        color="#b53a3a",
        alpha=0.75,
    )
    axes_plot[2].axhline(0.0, color="black", linewidth=0.8)
    axes_plot[2].set_title("Cost waterfall sensitivity")
    axes_plot[2].legend(fontsize=8)
    fig.suptitle("Return spread, turnover, and implementation costs")
    path = figures_dir / "09_return_turnover_cost_waterfall.png"
    _save_figure(fig, path)
    return path, [str(table_path)]


def _drawdown_factor_figure(
    p8_monthly: pd.DataFrame,
    p7_factors: pd.DataFrame,
    tables_dir: Path,
    figures_dir: Path,
) -> tuple[Path, list[str]]:
    monthly = p8_monthly.loc[
        _primary_mask(p8_monthly)
        & p8_monthly["delay_months"].eq(0)
        & p8_monthly["cost_scenario"].eq("conservative")
    ].copy()
    monthly["holding_date"] = pd.to_datetime(monthly["holding_date"])
    wealth = (1.0 + monthly["net_return"]).cumprod()
    monthly["drawdown"] = wealth / wealth.cummax() - 1.0
    factors = p7_factors.loc[_primary_mask(p7_factors)].copy()
    beta_columns = ["beta_mkt_excess", "beta_smb", "beta_hml", "beta_rmw", "beta_cma", "beta_mom"]
    beta_table = factors[["signal", *beta_columns]].melt(
        id_vars="signal", var_name="factor", value_name="beta"
    )
    drawdown_path = tables_dir / "p10_primary_net_drawdown.csv"
    beta_path = tables_dir / "p10_primary_factor_exposures.csv"
    _atomic_csv(monthly[["holding_date", "net_return", "drawdown"]], drawdown_path)
    _atomic_csv(beta_table, beta_path)
    fig, axes_plot = plt.subplots(1, 2, figsize=(12, 4.5))
    axes_plot[0].fill_between(
        monthly["holding_date"],
        monthly["drawdown"],
        0.0,
        color="#b53a3a",
        alpha=0.7,
    )
    axes_plot[0].set_title("Conservative net drawdown")
    axes_plot[0].set_ylabel("Drawdown")
    axes_plot[1].bar(
        beta_table["factor"].str.replace("beta_", ""),
        beta_table["beta"],
        color="#3a6ea5",
    )
    axes_plot[1].axhline(0.0, color="black", linewidth=0.8)
    axes_plot[1].set_title("P7 factor exposures")
    axes_plot[1].tick_params(axis="x", rotation=45)
    path = figures_dir / "10_drawdown_and_factor_exposures.png"
    _save_figure(fig, path)
    return path, [str(drawdown_path), str(beta_path)]


def _projection_3d(
    axes: pd.DataFrame,
    tables_dir: Path,
    figures_dir: Path,
    config: HypercubeConfig,
) -> tuple[list[Path], list[str]]:
    columns = (
        "anchored_demand_strength_pricing_power",
        "anchored_innovation_intensity",
        "anchored_unit_economics_profit_quality",
    )
    rng = np.random.default_rng(config.project.seed)
    selected = np.sort(
        rng.choice(len(axes), size=min(3000, len(axes)), replace=False)
    )
    sample = axes.iloc[selected][["gvkey", "fyear", *columns]].copy()
    table_path = tables_dir / "p10_three_axis_projection_sample.csv"
    _atomic_csv(sample, table_path)
    fig = plt.figure(figsize=(8, 6))
    axis = fig.add_subplot(111, projection="3d")
    points = axis.scatter(
        sample[columns[0]],
        sample[columns[1]],
        sample[columns[2]],
        c=sample["fyear"],
        cmap="viridis",
        s=6,
        alpha=0.55,
    )
    axis.set_xlabel("Demand/pricing")
    axis.set_ylabel("Innovation")
    axis.set_zlabel("Unit economics")
    axis.set_title("Three-axis projection of six-dimensional state")
    fig.colorbar(points, ax=axis, label="Fiscal year", shrink=0.7)
    static_path = figures_dir / "11_three_axis_projection_static.png"
    _save_figure(fig, static_path)

    fig = plt.figure(figsize=(7, 6))
    axis = fig.add_subplot(111, projection="3d")
    axis.scatter(
        sample[columns[0]],
        sample[columns[1]],
        sample[columns[2]],
        c=sample["fyear"],
        cmap="viridis",
        s=5,
        alpha=0.5,
    )
    axis.set_xlabel("Demand/pricing")
    axis.set_ylabel("Innovation")
    axis.set_zlabel("Unit economics")
    axis.set_title("Projection only — not the full hypercube")

    def rotate(angle: float) -> None:
        axis.view_init(elev=25, azim=angle)

    movie = animation.FuncAnimation(
        fig,
        rotate,
        frames=np.linspace(0, 360, 24, endpoint=False),
        interval=150,
    )
    gif_path = figures_dir / "11_three_axis_projection_rotating.gif"
    descriptor, name = tempfile.mkstemp(
        prefix=".three-axis.", suffix=".gif", dir=figures_dir
    )
    os.close(descriptor)
    try:
        movie.save(name, writer=animation.PillowWriter(fps=6), dpi=90)
        plt.close(fig)
        with Image.open(name) as image:
            if getattr(image, "n_frames", 1) < 2:
                raise VisualizationError("Rotating projection has fewer than two frames.")
        os.replace(name, gif_path)
    except Exception:
        plt.close(fig)
        Path(name).unlink(missing_ok=True)
        raise
    return [static_path, gif_path], [str(table_path)]


def _report_statistics(project_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    recovery_receipt = json.loads(
        (
            project_root / "artifacts" / "manifests" / "p7_validation.json"
        ).read_text(encoding="utf-8")
    )
    recovery_by_scenario = {
        item["scenario"]: item["recovery"]
        for item in recovery_receipt["reports"]
    }
    for scenario in SCENARIOS:
        root = _scenario_root(project_root, scenario)
        fmb = pd.read_csv(root / "p7" / "fmb_summary.csv")
        fmb_primary = fmb.loc[
            fmb["horizon_months"].eq(6)
            & fmb["signal"].eq("migration_surprise")
        ].iloc[0]
        recovery = recovery_by_scenario[scenario]
        costs = pd.read_csv(root / "p8" / "cost_aware_summary.csv")
        primary = costs.loc[
            _primary_mask(costs)
            & costs["delay_months"].eq(0)
            & costs["cost_scenario"].eq("conservative")
        ].iloc[0]
        stability = pd.read_csv(root / "p9" / "cluster_stability.csv")
        assignments = pd.read_parquet(
            root / "p9" / "archetype_assignments.parquet",
            columns=["is_noise_or_unassigned"],
        )
        rows.append(
            {
                "scenario": scenario,
                "p7_primary_ic": recovery["primary_spearman_ic"],
                "p7_primary_coefficient": fmb_primary["mean_coefficient"],
                "p7_primary_p_value": fmb_primary["p_value"],
                "p7_primary_holm_p_value": fmb_primary["holm_p_value"],
                "p8_annualized_gross_capacity_return": primary[
                    "annualized_gross_capacity_return"
                ],
                "p8_annualized_net_return": primary["annualized_net_return"],
                "p8_annualized_net_sharpe": primary["annualized_net_sharpe"],
                "p8_net_maximum_drawdown": primary["net_maximum_drawdown"],
                "p8_average_turnover": primary["average_turnover"],
                "p8_average_capacity_fill_ratio": primary[
                    "average_capacity_fill_ratio"
                ],
                "p9_noise_or_unassigned_rate": assignments[
                    "is_noise_or_unassigned"
                ].mean(),
                "p9_mean_stability_ari": stability[
                    "adjusted_rand_index"
                ].mean(),
            }
        )
    return pd.DataFrame(rows)


def build_visualization_bundle(
    project_root: Path,
    config: HypercubeConfig,
) -> dict[str, Any]:
    """Generate all required P10 figures and traceability tables."""

    if config.project.phase != "P10":
        raise VisualizationError("Final visualization requires a P10 config.")
    figures_dir = project_root / "figures"
    tables_dir = project_root / "artifacts" / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    data = _load_primary(project_root)
    registry: list[dict[str, Any]] = []

    def record(
        title: str,
        paths: Path | list[Path],
        sources: list[str],
    ) -> None:
        values = paths if isinstance(paths, list) else [paths]
        for path in values:
            registry.append(
                {
                    "figure": path.name,
                    "title": title,
                    "scenario": PRIMARY_SCENARIO,
                    "source_tables": " | ".join(sources),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256(path),
                }
            )

    path, sources = _axis_distribution_figure(
        data["axes"], tables_dir, figures_dir
    )
    record("Axis distributions and correlation", path, sources)
    path, sources = _pca_figure(
        data["axes"],
        data["assignments"],
        data["dynamics"],
        config,
        tables_dir,
        figures_dir,
    )
    record("PCA viability and archetypes", path, sources)
    path, sources = _calibration_figure(
        data["predictions"], tables_dir, figures_dir
    )
    record("Calibrated viability curve", path, sources)
    path, sources = _margin_figure(data["dynamics"], tables_dir, figures_dir)
    record("Viability margin by outcome", path, sources)
    path, sources = _selected_firm_figure(
        data["dynamics"], tables_dir, figures_dir
    )
    record("Niche dynamics examples", path, sources)
    path, sources = _transition_figure(
        data["transitions"], tables_dir, figures_dir
    )
    record("Archetype transition matrix", path, sources)
    path, sources = _state_evolution_figure(
        data["axes"], tables_dir, figures_dir
    )
    record("Anchored state-space evolution", path, sources)
    path, sources = _cell_occupancy_figure(
        data["axes"], tables_dir, figures_dir
    )
    record("729-cell occupancy", path, sources)
    path, sources = _cost_figure(
        data["p7_monthly"], data["p8_waterfall"], tables_dir, figures_dir
    )
    record("Return, turnover, and cost waterfall", path, sources)
    path, sources = _drawdown_factor_figure(
        data["p8_monthly"], data["p7_factors"], tables_dir, figures_dir
    )
    record("Drawdown and factor exposures", path, sources)
    paths, sources = _projection_3d(
        data["axes"], tables_dir, figures_dir, config
    )
    record("Three-axis projection", paths, sources)

    statistics = _report_statistics(project_root)
    statistics_path = tables_dir / "p10_report_statistics.csv"
    _atomic_csv(statistics, statistics_path)
    registry_frame = pd.DataFrame(registry)
    registry_path = tables_dir / "p10_figure_registry.csv"
    _atomic_csv(registry_frame, registry_path)
    metadata = {
        "schema_version": 1,
        "phase": "P10",
        "version": P10_VERSION,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_visual_scenario": PRIMARY_SCENARIO,
        "figure_files": int(len(registry_frame)),
        "figure_topics": 11,
        "report_statistics": str(statistics_path),
        "figure_registry": str(registry_path),
        "synthetic_only": True,
        "real_data_run": False,
        "primary_statistical_conclusion": "MIGRATION_SIGNAL_NOT_RECOVERED",
        "primary_implementation_conclusion": "NEGATIVE_AFTER_COSTS",
        "archetype_conclusion": "UNSTABLE_MOSTLY_NOISE",
    }
    atomic_write_json(
        project_root / "artifacts" / "manifests" / "p10_visualization.json",
        metadata,
    )
    return metadata


def validate_visualization_bundle(project_root: Path) -> dict[str, Any]:
    """Validate required figure topics, hashes, images, GIF, and source tables."""

    registry_path = project_root / "artifacts" / "tables" / "p10_figure_registry.csv"
    statistics_path = project_root / "artifacts" / "tables" / "p10_report_statistics.csv"
    errors: list[str] = []
    if not registry_path.is_file() or not statistics_path.is_file():
        return {
            "status": "FAIL",
            "errors": ["P10 registry or report-statistics table is missing."],
        }
    registry = pd.read_csv(registry_path)
    if len(registry) != 12:
        errors.append("P10 must contain 11 topics and 12 files including the GIF.")
    for _, row in registry.iterrows():
        path = project_root / "figures" / row["figure"]
        if not path.is_file():
            errors.append(f"Missing P10 figure: {path.name}")
            continue
        if path.stat().st_size != row["bytes"] or _sha256(path) != row["sha256"]:
            errors.append(f"P10 figure receipt mismatch: {path.name}")
            continue
        try:
            with Image.open(path) as image:
                image.verify()
            if path.suffix == ".gif":
                with Image.open(path) as image:
                    if getattr(image, "n_frames", 1) < 2:
                        errors.append("P10 rotating figure is not animated.")
        except Exception as exc:
            errors.append(f"P10 image validation failed for {path.name}: {exc}")
        for source in str(row["source_tables"]).split(" | "):
            if source and not Path(source).is_file():
                errors.append(f"P10 source table is missing: {source}")
    statistics = pd.read_csv(statistics_path)
    if set(statistics["scenario"]) != set(SCENARIOS):
        errors.append("P10 report statistics omit a synthetic scenario.")
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "figure_files": int(len(registry)),
        "figure_topics": 11,
        "statistics_rows": int(len(statistics)),
        "synthetic_only": True,
        "real_data_run": False,
    }
