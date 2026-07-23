"""P13F tests for outcome-blind construction and locked truth gates."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from hypercube.data import atomic_write_json
from hypercube.exploratory import atomic_write_csv
from hypercube.proxy_redesign import (
    ANCHORED_AXES,
    RELATIVE_AXES,
    atomic_write_parquet,
    audit_p7_eligibility,
    build_proxy_candidates,
    evaluate_proxy_candidates,
    validate_proxy_redesign_outputs,
)


def _fixtures(tmp_path: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(1313)
    rows = []
    truth_rows = []
    for issuer in range(120):
        level = rng.normal()
        for year in range(1999, 2019):
            innovation = rng.normal(scale=0.5)
            level = 0.72 * level + innovation
            row = {
                "gvkey": f"{issuer:06d}",
                "datadate": pd.Timestamp(year, 12, 31),
                "fyear": year,
                "feature_date": pd.Timestamp(year + 1, 6, 30),
                "horizon_years": 5,
                "migration_surprise": innovation + rng.normal(scale=0.9),
            }
            for column in RELATIVE_AXES:
                row[column] = level + rng.normal(scale=0.08)
            for column in ANCHORED_AXES:
                row[column] = level + rng.normal(scale=0.18)
            rows.append(row)
            truth_rows.append(
                {
                    "gvkey": row["gvkey"],
                    "datadate": row["datadate"],
                    "fyear": year,
                    "migration_surprise": innovation,
                }
            )
    p5_path = tmp_path / "p5.parquet"
    truth_path = tmp_path / "truth.parquet"
    pd.DataFrame(rows).to_parquet(p5_path, index=False)
    pd.DataFrame(truth_rows).to_parquet(truth_path, index=False)
    return p5_path, truth_path


def _build_kwargs() -> dict[str, int]:
    return {"minimum_train_transitions": 100, "minimum_prior_years": 2}


def test_proxy_builder_excludes_truth_and_returns(tmp_path: Path) -> None:
    p5_path, _ = _fixtures(tmp_path)
    candidates, coefficients, receipt = build_proxy_candidates(
        p5_path, **_build_kwargs()
    )

    assert receipt["synthetic_truth_read"] is False
    assert receipt["return_outcomes_read"] is False
    assert receipt["injected_alpha_read"] is False
    assert set(candidates).isdisjoint(
        {"truth_migration_surprise", "injected_return_alpha", "forward_excess_return"}
    )
    fitted = coefficients.dropna(subset=["train_max_year"])
    assert fitted["train_max_year"].lt(fitted["prediction_year"]).all()


def test_proxy_evaluation_selects_a_held_out_candidate(tmp_path: Path) -> None:
    p5_path, truth_path = _fixtures(tmp_path)
    candidates, _, _ = build_proxy_candidates(p5_path, **_build_kwargs())
    candidates_path = tmp_path / "candidates.parquet"
    atomic_write_parquet(candidates, candidates_path)

    summary, metrics = evaluate_proxy_candidates(candidates_path, truth_path)

    assert summary["status"] == "GO"
    assert summary["selected_candidate"].endswith("_axis_innovation")
    assert all(summary["gates"].values())
    assert len(metrics) == 4


def test_proxy_validator_reconstructs_both_stages(tmp_path: Path) -> None:
    p5_path, truth_path = _fixtures(tmp_path)
    candidates, coefficients, receipt = build_proxy_candidates(
        p5_path, **_build_kwargs()
    )
    candidates_path = tmp_path / "candidates.parquet"
    coefficients_path = tmp_path / "coefficients.csv"
    receipt_path = tmp_path / "receipt.json"
    evaluation_path = tmp_path / "evaluation.json"
    metrics_path = tmp_path / "metrics.csv"
    atomic_write_parquet(candidates, candidates_path)
    atomic_write_csv(coefficients, coefficients_path)
    atomic_write_json(receipt_path, receipt)
    summary, metrics = evaluate_proxy_candidates(candidates_path, truth_path)
    atomic_write_json(evaluation_path, summary)
    atomic_write_csv(metrics, metrics_path)

    report = validate_proxy_redesign_outputs(
        p5_path,
        truth_path,
        candidates_path,
        coefficients_path,
        receipt_path,
        evaluation_path,
        metrics_path,
        build_kwargs=_build_kwargs(),
    )

    assert report["status"] == "PASS"
    assert report["errors"] == []


def test_p7_eligibility_audit_reads_keys_not_return_values(tmp_path: Path) -> None:
    p5_path, truth_path = _fixtures(tmp_path)
    candidates, _, _ = build_proxy_candidates(p5_path, **_build_kwargs())
    candidates_path = tmp_path / "candidates.parquet"
    atomic_write_parquet(candidates, candidates_path)
    truth = pd.read_parquet(truth_path)
    targets = truth.loc[
        truth["fyear"].ge(2002), list(("gvkey", "datadate", "fyear"))
    ].copy()
    targets["horizon_months"] = 6
    targets["target_valid"] = True
    targets["forward_excess_return"] = np.nan
    targets_path = tmp_path / "targets.parquet"
    targets.to_parquet(targets_path, index=False)

    summary, table = audit_p7_eligibility(
        candidates_path,
        targets_path,
        truth_path,
        minimum_improvement=0.05,
    )

    assert summary["status"] == "GO"
    assert summary["return_values_read"] is False
    assert "forward_excess_return" not in summary["target_columns_read"]
    assert len(table) == 2
