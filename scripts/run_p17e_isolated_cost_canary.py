"""Run the preregistered P17E locked-proxy P8 canary."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path

import pandas as pd

from hypercube.canary_costs import (
    SCENARIOS,
    build_locked_proxy_cost_scenario,
    validate_locked_proxy_cost_scenario,
)
from hypercube.config import load_config
from hypercube.data import atomic_write_json, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANARY_ROOT = Path(
    os.environ.get(
        "HYPERCUBE_CANARY_ROOT",
        PROJECT_ROOT.parent / ".p16-hypercube-canary",
    )
)


def _main_phase_hashes() -> dict[str, str]:
    root = PROJECT_ROOT / "artifacts" / "manifests"
    selected = [
        path
        for path in root.glob("*.json")
        if path.name.startswith(tuple(f"p{i}" for i in range(11)))
        or path.name == "full_pipeline_receipt.json"
    ]
    return {path.name: sha256_file(path) for path in sorted(selected)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canary-root",
        type=Path,
        default=DEFAULT_CANARY_ROOT,
    )
    parser.add_argument(
        "--preregistration",
        type=Path,
        default=PROJECT_ROOT
        / "docs"
        / "p17e_isolated_cost_canary_preregistration.md",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "manifests"
        / "p17e_cost_canary.json",
    )
    parser.add_argument(
        "--validation-report",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "manifests"
        / "p17e_cost_canary_validation.json",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "tables"
        / "p17e_primary_cost_results.csv",
    )
    args = parser.parse_args()
    config = load_config(args.canary_root / "configs" / "synthetic.yaml")
    before = _main_phase_hashes()
    results = []
    validations = []
    for scenario in SCENARIOS:
        candidate = (
            PROJECT_ROOT
            / "artifacts"
            / "tables"
            / f"p16e_{'migration' if scenario == 'migration_alpha' else 'null'}"
            "_proxy_candidates.parquet"
        )
        p7_dir = (
            args.canary_root
            / "data"
            / "processed"
            / "synthetic"
            / scenario
            / "p7"
        )
        raw_dir = (
            args.canary_root / "data" / "raw" / "synthetic" / scenario
        )
        output_dir = (
            args.canary_root
            / "data"
            / "processed"
            / "synthetic"
            / scenario
            / "p17e_p8"
        )
        result = build_locked_proxy_cost_scenario(
            scenario=scenario,
            p7_dir=p7_dir,
            raw_dir=raw_dir,
            candidate_path=candidate,
            output_dir=output_dir,
            config=config,
            preregistration_path=args.preregistration,
        )
        validation = validate_locked_proxy_cost_scenario(
            output_dir, config
        )
        if validation["status"] != "PASS":
            raise RuntimeError(
                f"P17E validation failed for {scenario}: "
                f"{validation['errors']}"
            )
        results.append(result)
        validations.append(validation)
    after = _main_phase_hashes()
    frozen_unchanged = before == after
    if not frozen_unchanged:
        raise RuntimeError("Main P0-P10 manifest hashes changed during P17E.")
    rows = []
    for item in validations:
        rows.append({"scenario": item["scenario"], **item["primary_result"]})
    args.table.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.table, index=False, lineterminator="\n")
    report = {
        "schema_version": 1,
        "phase": "P17E",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "selected_candidate": "anchored_axis_innovation",
        "isolated_canary": True,
        "scenarios": list(SCENARIOS),
        "results": results,
        "main_p0_p10_manifest_hashes_before": before,
        "main_p0_p10_manifest_hashes_after": after,
        "frozen_main_p0_p10_unchanged": frozen_unchanged,
        "synthetic_truth_read": False,
        "performance_gate_applied": False,
        "p9_p10_run": False,
        "real_data_run": False,
    }
    validation_report = {
        "schema_version": 1,
        "phase": "P17E",
        "status": "PASS",
        "errors": [],
        "scenario_validations": validations,
        "frozen_main_p0_p10_unchanged": frozen_unchanged,
        "synthetic_truth_read": False,
        "performance_gate_applied": False,
    }
    atomic_write_json(args.report, report)
    atomic_write_json(args.validation_report, validation_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
