"""Outcome-free measurement-utility audit for phase P21E."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

from hypercube.proxy_redesign import (
    ANCHORED_AXES,
    BENCHMARK,
    EVENT_KEY,
    RELATIVE_AXES,
)


P21E_VERSION = "hypercube-measurement-utility-v1"
LATENT_AXES = tuple(f"latent_axis_{index}" for index in range(1, 7))
PRIMARY_DRIFT_SCORE = "anchored_axis_innovation"
OBSERVED_SPACES: Mapping[str, tuple[str, ...]] = {
    "anchored": ANCHORED_AXES,
    "relative": RELATIVE_AXES,
}
EVALUATION_YEARS = (2013, 2018)
STABILITY_BLOCKS = {
    "2013_2015": (2013, 2015),
    "2016_2018": (2016, 2018),
}


class MeasurementUtilityError(ValueError):
    """Raised when P21E inputs or deterministic contracts are violated."""


def _standardize(values: np.ndarray) -> np.ndarray:
    """Cross-sectionally standardize columns with a safe zero-variance rule."""

    means = values.mean(axis=0)
    scales = values.std(axis=0, ddof=0)
    scales = np.where(scales > 0.0, scales, 1.0)
    return (values - means) / scales


def _nearest_other(
    values: np.ndarray,
    neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact nearest-other indices and distances for every row."""

    if len(values) <= neighbors:
        raise MeasurementUtilityError(
            f"Need more than {neighbors} rows for nearest-neighbor evaluation."
        )
    model = NearestNeighbors(
        n_neighbors=neighbors + 1,
        metric="euclidean",
        algorithm="auto",
    )
    model.fit(values)
    raw_distances, raw_indices = model.kneighbors(values, return_distance=True)
    selected_indices = np.empty((len(values), neighbors), dtype=np.int64)
    selected_distances = np.empty((len(values), neighbors), dtype=float)
    for row_index in range(len(values)):
        keep = raw_indices[row_index] != row_index
        indices = raw_indices[row_index][keep][:neighbors]
        distances = raw_distances[row_index][keep][:neighbors]
        if len(indices) != neighbors:
            raise MeasurementUtilityError(
                "Nearest-neighbor search did not return enough non-self rows."
            )
        selected_indices[row_index] = indices
        selected_distances[row_index] = distances
    return selected_indices, selected_distances


def _deterministic_top_fraction(
    frame: pd.DataFrame,
    value_column: str,
    fraction: float,
) -> pd.Series:
    """Flag an exact top fraction by absolute value within each year."""

    output = pd.Series(False, index=frame.index, dtype=bool)
    for _, group in frame.groupby("fyear", sort=True):
        available = group.loc[group[value_column].notna()].copy()
        if available.empty:
            continue
        count = max(1, int(math.ceil(fraction * len(available))))
        available["_absolute_value"] = available[value_column].abs()
        chosen = available.sort_values(
            ["_absolute_value", "gvkey"],
            ascending=[False, True],
            kind="mergesort",
        ).head(count)
        output.loc[chosen.index] = True
    return output


def build_drift_events(
    candidates: pd.DataFrame,
    truth: pd.DataFrame,
    scenario: str,
    *,
    alert_fraction: float = 0.10,
    evaluation_years: tuple[int, int] = EVALUATION_YEARS,
) -> pd.DataFrame:
    """Build deterministic drift-alert and truth-extreme indicators."""

    candidate_columns = [
        *EVENT_KEY,
        PRIMARY_DRIFT_SCORE,
        BENCHMARK,
    ]
    truth_columns = [*EVENT_KEY, "migration_surprise"]
    missing_candidates = sorted(set(candidate_columns).difference(candidates.columns))
    missing_truth = sorted(set(truth_columns).difference(truth.columns))
    if missing_candidates or missing_truth:
        raise MeasurementUtilityError(
            f"Drift inputs missing candidate={missing_candidates}, truth={missing_truth}."
        )
    if candidates.duplicated(list(EVENT_KEY)).any():
        raise MeasurementUtilityError("Candidate drift events are duplicated.")
    if truth.duplicated(list(EVENT_KEY)).any():
        raise MeasurementUtilityError("Truth drift events are duplicated.")

    joined = candidates[candidate_columns].merge(
        truth[truth_columns].rename(
            columns={"migration_surprise": "truth_migration_surprise"}
        ),
        on=list(EVENT_KEY),
        how="left",
        validate="one_to_one",
    )
    joined = joined.loc[
        joined["fyear"].between(*evaluation_years)
        & joined[BENCHMARK].notna()
        & joined["truth_migration_surprise"].notna()
    ].copy()
    if joined.empty:
        raise MeasurementUtilityError("No comparable P21E drift events remain.")

    truth_flag = _deterministic_top_fraction(
        joined,
        "truth_migration_surprise",
        alert_fraction,
    )
    rows: list[pd.DataFrame] = []
    for score in (PRIMARY_DRIFT_SCORE, BENCHMARK):
        saved = joined[
            [*EVENT_KEY, score, "truth_migration_surprise"]
        ].rename(columns={score: "score_value"})
        saved.insert(0, "scenario", scenario)
        saved.insert(1, "score", score)
        saved["score_available"] = saved["score_value"].notna()
        saved["is_alert"] = _deterministic_top_fraction(
            joined,
            score,
            alert_fraction,
        )
        saved["is_truth_extreme"] = truth_flag
        saved["correct_extreme"] = saved["is_alert"] & saved["is_truth_extreme"]
        saved["sign_correct"] = (
            np.sign(saved["score_value"])
            == np.sign(saved["truth_migration_surprise"])
        )
        rows.append(saved)
    return pd.concat(rows, ignore_index=True)


def summarize_drift_events(events: pd.DataFrame) -> pd.DataFrame:
    """Summarize drift alerts overall, by scenario, and by stability block."""

    rows: list[dict[str, Any]] = []

    def add_scope(scope: str, label: str, frame: pd.DataFrame) -> None:
        for score, group in frame.groupby("score", sort=True):
            alerts = int(group["is_alert"].sum())
            truth_events = int(group["is_truth_extreme"].sum())
            correct = int(group["correct_extreme"].sum())
            alert_rows = group.loc[group["is_alert"]]
            rows.append(
                {
                    "scope": scope,
                    "scope_value": label,
                    "score": score,
                    "rows": int(len(group)),
                    "available_rows": int(group["score_available"].sum()),
                    "coverage": float(group["score_available"].mean()),
                    "alerts": alerts,
                    "truth_extremes": truth_events,
                    "correct_extremes": correct,
                    "precision": float(correct / alerts) if alerts else float("nan"),
                    "recall": (
                        float(correct / truth_events)
                        if truth_events
                        else float("nan")
                    ),
                    "sign_accuracy": (
                        float(alert_rows["sign_correct"].mean())
                        if alerts
                        else float("nan")
                    ),
                }
            )

    add_scope("overall", "all", events)
    for scenario, group in events.groupby("scenario", sort=True):
        add_scope("scenario", str(scenario), group)
    for block, (start, end) in STABILITY_BLOCKS.items():
        add_scope(
            "block",
            block,
            events.loc[events["fyear"].between(start, end)],
        )
    return pd.DataFrame(rows).sort_values(
        ["scope", "scope_value", "score"]
    ).reset_index(drop=True)


def drift_alert_gate(metrics: pd.DataFrame) -> dict[str, Any]:
    """Apply the frozen strategic-drift decision rule."""

    def row(scope: str, value: str, score: str) -> pd.Series:
        selected = metrics.loc[
            metrics["scope"].eq(scope)
            & metrics["scope_value"].eq(value)
            & metrics["score"].eq(score)
        ]
        if len(selected) != 1:
            raise MeasurementUtilityError(
                f"Expected one drift metric row for {scope}/{value}/{score}."
            )
        return selected.iloc[0]

    primary = row("overall", "all", PRIMARY_DRIFT_SCORE)
    benchmark = row("overall", "all", BENCHMARK)
    blocks = [
        row("block", block, PRIMARY_DRIFT_SCORE)
        for block in STABILITY_BLOCKS
    ]
    improvement = float(primary["precision"] - benchmark["precision"])
    gates: Mapping[str, bool] = {
        "overall_precision_at_least_0_25": bool(primary["precision"] >= 0.25),
        "both_blocks_precision_at_least_0_20": bool(
            all(block["precision"] >= 0.20 for block in blocks)
        ),
        "sign_accuracy_at_least_0_75": bool(primary["sign_accuracy"] >= 0.75),
        "coverage_at_least_0_95": bool(primary["coverage"] >= 0.95),
        "precision_improvement_at_least_0_05": bool(improvement >= 0.05),
    }
    return {
        "verdict": "SUPPORTED" if all(gates.values()) else "NOT_SUPPORTED",
        "gates": dict(gates),
        "overall_precision": float(primary["precision"]),
        "overall_recall": float(primary["recall"]),
        "overall_sign_accuracy": float(primary["sign_accuracy"]),
        "coverage": float(primary["coverage"]),
        "benchmark_precision": float(benchmark["precision"]),
        "precision_improvement": improvement,
        "block_precision": {
            block: float(row("block", block, PRIMARY_DRIFT_SCORE)["precision"])
            for block in STABILITY_BLOCKS
        },
    }


def build_peer_geometry(
    axes: pd.DataFrame,
    truth: pd.DataFrame,
    scenario: str,
    *,
    neighbors: int = 20,
    evaluation_years: tuple[int, int] = EVALUATION_YEARS,
    example_neighbors: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Evaluate structural peers and crowding against latent geometry."""

    axis_columns = [*EVENT_KEY, *ANCHORED_AXES, *RELATIVE_AXES]
    truth_columns = [*EVENT_KEY, *LATENT_AXES]
    missing_axes = sorted(set(axis_columns).difference(axes.columns))
    missing_truth = sorted(set(truth_columns).difference(truth.columns))
    if missing_axes or missing_truth:
        raise MeasurementUtilityError(
            f"Peer inputs missing axes={missing_axes}, truth={missing_truth}."
        )
    if axes.duplicated(list(EVENT_KEY)).any():
        raise MeasurementUtilityError("P3 peer events are duplicated.")
    if truth.duplicated(list(EVENT_KEY)).any():
        raise MeasurementUtilityError("Truth peer events are duplicated.")

    evaluation = axes.loc[
        axes["fyear"].between(*evaluation_years),
        axis_columns,
    ].copy()
    total_rows = int(len(evaluation))
    joined = evaluation.merge(
        truth[truth_columns],
        on=list(EVENT_KEY),
        how="left",
        validate="one_to_one",
    )
    complete_columns = [*ANCHORED_AXES, *RELATIVE_AXES, *LATENT_AXES]
    joined = joined.dropna(subset=complete_columns).copy()
    if joined.empty:
        raise MeasurementUtilityError("No complete P21E peer events remain.")

    row_metrics: list[pd.DataFrame] = []
    example_edges: list[dict[str, Any]] = []
    for fyear, group in joined.groupby("fyear", sort=True):
        ordered = group.sort_values("gvkey", kind="mergesort").reset_index(drop=True)
        if len(ordered) <= neighbors:
            raise MeasurementUtilityError(
                f"Peer group {scenario}/{fyear} is too small."
            )
        latent_values = _standardize(
            ordered[list(LATENT_AXES)].to_numpy(float)
        )
        latent_neighbors, latent_distances = _nearest_other(
            latent_values,
            neighbors,
        )
        latent_sets = [set(indices.tolist()) for indices in latent_neighbors]
        latent_crowding = -latent_distances.mean(axis=1)
        for space, columns in OBSERVED_SPACES.items():
            observed_values = _standardize(
                ordered[list(columns)].to_numpy(float)
            )
            observed_neighbors, observed_distances = _nearest_other(
                observed_values,
                neighbors,
            )
            recall = np.asarray(
                [
                    len(latent_sets[index].intersection(peers.tolist()))
                    / neighbors
                    for index, peers in enumerate(observed_neighbors)
                ],
                dtype=float,
            )
            saved = ordered[list(EVENT_KEY)].copy()
            saved.insert(0, "scenario", scenario)
            saved.insert(1, "space", space)
            saved["group_rows"] = int(len(ordered))
            saved["neighbors"] = int(neighbors)
            saved["recall_at_k"] = recall
            saved["random_recall"] = float(neighbors / (len(ordered) - 1))
            saved["observed_crowding"] = -observed_distances.mean(axis=1)
            saved["latent_crowding"] = latent_crowding
            row_metrics.append(saved)

            if space == "anchored":
                for row_index, peers in enumerate(observed_neighbors):
                    for rank, peer_index in enumerate(
                        peers[:example_neighbors],
                        start=1,
                    ):
                        example_edges.append(
                            {
                                "scenario": scenario,
                                "fyear": int(fyear),
                                "gvkey": str(ordered.loc[row_index, "gvkey"]),
                                "peer_gvkey": str(
                                    ordered.loc[peer_index, "gvkey"]
                                ),
                                "peer_rank": int(rank),
                                "observed_distance": float(
                                    observed_distances[row_index, rank - 1]
                                ),
                                "is_latent_top20_peer": bool(
                                    peer_index in latent_sets[row_index]
                                ),
                            }
                        )
    return (
        pd.concat(row_metrics, ignore_index=True),
        pd.DataFrame(example_edges),
        {"p3_evaluation_rows": total_rows, "complete_rows": int(len(joined))},
    )


def summarize_peer_geometry(
    rows: pd.DataFrame,
    coverage: Mapping[str, Mapping[str, int]],
) -> pd.DataFrame:
    """Summarize peer recall and crowding fidelity at frozen scopes."""

    output: list[dict[str, Any]] = []

    def add_scope(
        scope: str,
        value: str,
        frame: pd.DataFrame,
        complete_rows: int,
        total_rows: int,
    ) -> None:
        for space, group in frame.groupby("space", sort=True):
            correlation = float(
                spearmanr(
                    group["observed_crowding"],
                    group["latent_crowding"],
                ).statistic
            )
            mean_recall = float(group["recall_at_k"].mean())
            mean_random = float(group["random_recall"].mean())
            output.append(
                {
                    "scope": scope,
                    "scope_value": value,
                    "space": space,
                    "rows": int(len(group)),
                    "complete_rows": int(complete_rows),
                    "p3_evaluation_rows": int(total_rows),
                    "coverage": float(complete_rows / total_rows),
                    "mean_recall_at_20": mean_recall,
                    "mean_random_recall": mean_random,
                    "recall_lift_vs_random": float(mean_recall / mean_random),
                    "crowding_spearman": correlation,
                }
            )

    total_complete = sum(item["complete_rows"] for item in coverage.values())
    total_rows = sum(item["p3_evaluation_rows"] for item in coverage.values())
    add_scope("overall", "all", rows, total_complete, total_rows)
    for scenario, group in rows.groupby("scenario", sort=True):
        counts = coverage[str(scenario)]
        add_scope(
            "scenario",
            str(scenario),
            group,
            counts["complete_rows"],
            counts["p3_evaluation_rows"],
        )
    for block, (start, end) in STABILITY_BLOCKS.items():
        block_rows = rows.loc[rows["fyear"].between(start, end)]
        block_complete = int(
            block_rows.loc[block_rows["space"].eq("anchored")].shape[0]
        )
        block_total = 0
        for scenario in coverage:
            scenario_rows = rows.loc[
                rows["scenario"].eq(scenario)
                & rows["space"].eq("anchored")
                & rows["fyear"].between(start, end)
            ]
            scenario_coverage = coverage[scenario]
            if scenario_coverage["complete_rows"]:
                ratio = (
                    scenario_coverage["p3_evaluation_rows"]
                    / scenario_coverage["complete_rows"]
                )
                block_total += int(round(len(scenario_rows) * ratio))
        add_scope(
            "block",
            block,
            block_rows,
            block_complete,
            block_total,
        )
    return pd.DataFrame(output).sort_values(
        ["scope", "scope_value", "space"]
    ).reset_index(drop=True)


def peer_map_gate(metrics: pd.DataFrame) -> dict[str, Any]:
    """Apply the frozen anchored peer-map decision rule."""

    primary = metrics.loc[
        metrics["scope"].eq("overall")
        & metrics["scope_value"].eq("all")
        & metrics["space"].eq("anchored")
    ].iloc[0]
    scenarios = metrics.loc[
        metrics["scope"].eq("scenario")
        & metrics["space"].eq("anchored")
    ]
    gates: Mapping[str, bool] = {
        "overall_recall_at_least_0_05": bool(
            primary["mean_recall_at_20"] >= 0.05
        ),
        "every_scenario_recall_at_least_0_03": bool(
            len(scenarios) == 3
            and scenarios["mean_recall_at_20"].ge(0.03).all()
        ),
        "recall_lift_at_least_5x_random": bool(
            primary["recall_lift_vs_random"] >= 5.0
        ),
        "coverage_at_least_0_85": bool(primary["coverage"] >= 0.85),
    }
    return {
        "verdict": "SUPPORTED" if all(gates.values()) else "NOT_SUPPORTED",
        "gates": dict(gates),
        "mean_recall_at_20": float(primary["mean_recall_at_20"]),
        "random_recall": float(primary["mean_random_recall"]),
        "recall_lift_vs_random": float(primary["recall_lift_vs_random"]),
        "coverage": float(primary["coverage"]),
        "scenario_recall": {
            str(row["scope_value"]): float(row["mean_recall_at_20"])
            for _, row in scenarios.iterrows()
        },
    }


def crowding_gate(metrics: pd.DataFrame) -> dict[str, Any]:
    """Apply the frozen anchored crowding-state decision rule."""

    primary = metrics.loc[
        metrics["scope"].eq("overall")
        & metrics["scope_value"].eq("all")
        & metrics["space"].eq("anchored")
    ].iloc[0]
    scenarios = metrics.loc[
        metrics["scope"].eq("scenario")
        & metrics["space"].eq("anchored")
    ]
    gates: Mapping[str, bool] = {
        "overall_spearman_at_least_0_20": bool(
            primary["crowding_spearman"] >= 0.20
        ),
        "every_scenario_spearman_at_least_0_10": bool(
            len(scenarios) == 3
            and scenarios["crowding_spearman"].ge(0.10).all()
        ),
        "coverage_at_least_0_85": bool(primary["coverage"] >= 0.85),
    }
    return {
        "verdict": "SUPPORTED" if all(gates.values()) else "NOT_SUPPORTED",
        "gates": dict(gates),
        "crowding_spearman": float(primary["crowding_spearman"]),
        "coverage": float(primary["coverage"]),
        "scenario_spearman": {
            str(row["scope_value"]): float(row["crowding_spearman"])
            for _, row in scenarios.iterrows()
        },
    }


def product_gate(
    drift: Mapping[str, Any],
    peer_map: Mapping[str, Any],
    crowding: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the frozen overall measurement-monitor product gate."""

    drift_passed = drift["verdict"] == "SUPPORTED"
    structural_passed = (
        peer_map["verdict"] == "SUPPORTED"
        or crowding["verdict"] == "SUPPORTED"
    )
    gates = {
        "strategic_drift_supported": drift_passed,
        "peer_or_crowding_supported": structural_passed,
    }
    return {
        "verdict": (
            "MEASUREMENT_MONITOR_SUPPORTED"
            if all(gates.values())
            else "RESEARCH_BENCHMARK_ONLY"
        ),
        "gates": gates,
        "supported_use_cases": [
            name
            for name, result in (
                ("strategic_drift_alerts", drift),
                ("structural_peer_discovery", peer_map),
                ("competitive_crowding_state", crowding),
            )
            if result["verdict"] == "SUPPORTED"
        ],
    }
