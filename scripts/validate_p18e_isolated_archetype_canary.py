"""Independently validate the saved P18E isolated P9 canary."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from hypercube.clustering import validate_p9_directory
from hypercube.config import load_config
from hypercube.data import SCENARIOS, atomic_write_json, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANARY_ROOT = Path(
    os.environ.get(
        "HYPERCUBE_CANARY_ROOT",
        PROJECT_ROOT.parent / ".p16-hypercube-canary",
    )
)


def _main_manifest_hashes() -> dict[str, str]:
    root = PROJECT_ROOT / "artifacts" / "manifests"
    return {
        path.name: sha256_file(path)
        for path in sorted(root.glob("*.json"))
        if not path.name.startswith("p18e_")
    }


def _check_record(
    record: dict[str, Any], root: Path, errors: list[str], label: str
) -> None:
    path = root / record["path"]
    if not path.is_file():
        errors.append(f"{label} is missing: {record['path']}")
        return
    if path.stat().st_size != record["bytes"]:
        errors.append(f"{label} byte count changed: {record['path']}")
    if sha256_file(path) != record["sha256"]:
        errors.append(f"{label} hash changed: {record['path']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canary-root",
        type=Path,
        default=DEFAULT_CANARY_ROOT,
    )
    parser.add_argument(
        "--input-report",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "manifests"
        / "p18e_archetype_canary.json",
    )
    parser.add_argument(
        "--summary-table",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "tables"
        / "p18e_archetype_summary.csv",
    )
    parser.add_argument(
        "--profiles-table",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "tables"
        / "p18e_archetype_profiles.csv",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "manifests"
        / "p18e_archetype_canary_validation.json",
    )
    args = parser.parse_args()

    canary_root = args.canary_root.resolve()
    result = json.loads(args.input_report.read_text(encoding="utf-8"))
    config = load_config(
        canary_root
        / "artifacts"
        / "checkpoints"
        / "full_pipeline_configs"
        / "p9.yaml"
    )
    errors: list[str] = []
    warnings: list[str] = []
    scenario_reports = []

    for record in result["input_manifests"]:
        _check_record(record, canary_root, errors, "P18E input")
    for record in result["isolated_reports"]:
        _check_record(record, canary_root, errors, "P18E isolated report")
    for record in result["isolated_outputs"]:
        _check_record(record, canary_root, errors, "P18E isolated output")

    if result["main_manifest_hashes_before"] != result[
        "main_manifest_hashes_after"
    ]:
        errors.append("Main manifest hashes changed during P18E.")
    if _main_manifest_hashes() != result["main_manifest_hashes_after"]:
        errors.append("Main manifest hashes changed after the P18E run.")
    if result.get("synthetic_truth_read") is not False:
        errors.append("P18E reports synthetic truth access.")
    if result.get("return_models_refit") is not False:
        errors.append("P18E reports a return-model refit.")
    if result.get("p10_run") is not False:
        errors.append("P18E reports crossing the P10 boundary.")

    prior_cwd = Path.cwd()
    try:
        os.chdir(canary_root)
        for scenario in SCENARIOS:
            p2_dir = (
                canary_root
                / "data"
                / "processed"
                / "synthetic"
                / scenario
            )
            p9_dir = p2_dir / "p9"
            checked = validate_p9_directory(
                p9_dir, p2_dir / "p3", config, scenario=scenario
            )
            scenario_reports.append({"scenario": scenario, **checked})
            errors.extend(
                f"{scenario}: {error}" for error in checked.get("errors", [])
            )
            warnings.extend(
                f"{scenario}: {warning}"
                for warning in checked.get("warnings", [])
            )
            p10_dir = p2_dir / "p10"
            if p10_dir.exists() and any(p10_dir.iterdir()):
                errors.append(f"{scenario}: isolated P10 output exists.")
    finally:
        os.chdir(prior_cwd)

    summary = pd.read_csv(args.summary_table)
    if set(summary["scenario"]) != set(SCENARIOS):
        errors.append("P18E summary scenario coverage is incomplete.")
    if not summary["validation_status"].eq("PASS").all():
        errors.append("P18E summary contains a failed scenario.")
    for saved in result["scenario_summaries"]:
        row = summary.loc[summary["scenario"].eq(saved["scenario"])]
        if len(row) != 1:
            errors.append(f"Duplicate/missing summary: {saved['scenario']}")
            continue
        observed = row.iloc[0]
        for column in (
            "assignment_rows",
            "fit_rows",
            "cluster_count",
            "successful_stability_refits",
            "stability_repetitions",
        ):
            if int(observed[column]) != int(saved[column]):
                errors.append(
                    f"{saved['scenario']}: summary mismatch for {column}."
                )
        if abs(
            float(observed["noise_or_unassigned_rate"])
            - float(saved["noise_or_unassigned_rate"])
        ) > 1e-12:
            errors.append(
                f"{saved['scenario']}: summary mismatch for noise rate."
            )

    profiles = pd.read_csv(args.profiles_table)
    if set(profiles["scenario"]) != set(SCENARIOS):
        errors.append("P18E profile scenario coverage is incomplete.")
    allowed = {"Archetype A", "Archetype B", "Noise / Unassigned"}
    unexpected = set(profiles["archetype"]) - allowed
    if unexpected:
        errors.append(f"P18E contains unexpected archetype labels: {unexpected}")
    if set(profiles["sample_role"]) != {"training_period", "out_of_sample"}:
        errors.append("P18E profile sample-role coverage is incomplete.")

    payload = {
        "schema_version": 1,
        "phase": "P18E",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "warnings": warnings,
        "scenario_validations": scenario_reports,
        "recorded_hashes_recomputed": True,
        "frozen_main_manifests_unchanged": not any(
            "Main manifest" in error for error in errors
        ),
        "synthetic_truth_read": False,
        "return_models_refit": False,
        "p10_run": False,
    }
    atomic_write_json(args.report, payload)
    if errors:
        raise SystemExit("\n".join(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
