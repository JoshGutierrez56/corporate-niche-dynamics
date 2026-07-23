"""Run the preregistered P18E isolated P9 descriptive-archetype canary."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from hypercube.config import load_config
from hypercube.data import SCENARIOS, atomic_write_json, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANARY_ROOT = Path(
    os.environ.get(
        "HYPERCUBE_CANARY_ROOT",
        PROJECT_ROOT.parent / ".p16-hypercube-canary",
    )
)
CANONICAL_FILES = (
    "hypercube/clustering.py",
    "scripts/build_archetypes.py",
    "scripts/validate_archetypes.py",
)
INPUT_MANIFESTS = (
    "p3_build.json",
    "p4_build.json",
    "p5_build.json",
    "p7_build.json",
    "p7_manifest.json",
)


def _file_record(path: Path, root: Path | None = None) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root) if root else path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _main_manifest_hashes() -> dict[str, str]:
    root = PROJECT_ROOT / "artifacts" / "manifests"
    return {
        path.name: sha256_file(path)
        for path in sorted(root.glob("*.json"))
        if not path.name.startswith("p18e_")
    }


def _assert_clean_boundary(canary_root: Path) -> None:
    for scenario in SCENARIOS:
        processed = (
            canary_root
            / "data"
            / "processed"
            / "synthetic"
            / scenario
        )
        for phase in ("p9", "p10"):
            output = processed / phase
            if output.exists() and any(output.iterdir()):
                raise FileExistsError(
                    f"Isolated {phase.upper()} output already exists: {output}"
                )


def _assert_frozen_contract(canary_root: Path, config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    frozen = {
        "phase": config.project.phase,
        "representation": config.clustering.representation,
        "training_end_year": config.clustering.training_end_year,
        "maximum_training_rows": config.clustering.maximum_training_rows,
        "minimum_cluster_size": config.clustering.minimum_cluster_size,
        "minimum_samples": config.clustering.minimum_samples,
        "assignment_radius_quantile": (
            config.clustering.assignment_radius_quantile
        ),
        "stability_repetitions": config.clustering.stability_repetitions,
        "stability_sample_fraction": (
            config.clustering.stability_sample_fraction
        ),
    }
    expected = {
        "phase": "P9",
        "representation": "anchored",
        "training_end_year": 2004,
        "maximum_training_rows": 25_000,
        "minimum_cluster_size": 200,
        "minimum_samples": 10,
        "assignment_radius_quantile": 0.95,
        "stability_repetitions": 5,
        "stability_sample_fraction": 0.8,
    }
    if frozen != expected:
        raise RuntimeError(f"P18E frozen P9 contract mismatch: {frozen}")
    code_records = []
    for relative in CANONICAL_FILES:
        main_path = PROJECT_ROOT / relative
        canary_path = canary_root / relative
        main_hash = sha256_file(main_path)
        canary_hash = sha256_file(canary_path)
        if main_hash != canary_hash:
            raise RuntimeError(f"Isolated canonical-code mismatch: {relative}")
        code_records.append(
            {
                "path": relative,
                "main_sha256": main_hash,
                "isolated_sha256": canary_hash,
                "match": True,
            }
        )
    return {
        "frozen_contract": frozen,
        "canonical_code": code_records,
        "config": _file_record(config_path, canary_root),
    }


def _run(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


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
        / "p18e_isolated_archetype_canary_preregistration.md",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "manifests"
        / "p18e_archetype_canary.json",
    )
    parser.add_argument(
        "--validation-report",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "manifests"
        / "p18e_archetype_canary_validation.json",
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
    args = parser.parse_args()

    if not args.preregistration.is_file():
        raise FileNotFoundError("P18E preregistration must exist before the run.")
    for output in (
        args.report,
        args.validation_report,
        args.summary_table,
        args.profiles_table,
    ):
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite P18E output: {output}")

    canary_root = args.canary_root.resolve()
    config_path = (
        canary_root
        / "artifacts"
        / "checkpoints"
        / "full_pipeline_configs"
        / "p9.yaml"
    )
    _assert_clean_boundary(canary_root)
    contract = _assert_frozen_contract(canary_root, config_path)
    input_records = [
        _file_record(canary_root / "artifacts" / "manifests" / name, canary_root)
        for name in INPUT_MANIFESTS
    ]
    before = _main_manifest_hashes()

    isolated_build = (
        canary_root / "artifacts" / "manifests" / "p18e_p9_build.json"
    )
    isolated_validation = (
        canary_root / "artifacts" / "manifests" / "p18e_p9_validation.json"
    )
    _run(
        [
            sys.executable,
            str(canary_root / "scripts" / "build_archetypes.py"),
            "--config",
            str(config_path),
            "--all-scenarios",
            "--report",
            str(isolated_build),
        ],
        canary_root,
    )
    _run(
        [
            sys.executable,
            str(canary_root / "scripts" / "validate_archetypes.py"),
            "--config",
            str(config_path),
            "--all-scenarios",
            "--report",
            str(isolated_validation),
        ],
        canary_root,
    )

    build = json.loads(isolated_build.read_text(encoding="utf-8"))
    validation = json.loads(isolated_validation.read_text(encoding="utf-8"))
    reports = {item["scenario"]: item for item in build["reports"]}
    validations = {item["scenario"]: item for item in validation["reports"]}
    if build["status"] != "PASS" or validation["status"] != "PASS":
        raise RuntimeError("P18E build or independent validation failed.")
    if set(reports) != set(SCENARIOS) or set(validations) != set(SCENARIOS):
        raise RuntimeError("P18E scenario coverage is incomplete.")

    summary_rows = []
    profile_frames = []
    output_records = []
    for scenario in SCENARIOS:
        construction = reports[scenario]["construction"]
        checked = validations[scenario]
        if construction["synthetic_truth_read"] is not False:
            raise RuntimeError(f"P18E truth-read flag failed for {scenario}.")
        if construction["return_models_refit"] is not False:
            raise RuntimeError(f"P18E return-refit flag failed for {scenario}.")
        if checked["synthetic_truth_read"] is not False:
            raise RuntimeError(f"P18E validation truth flag failed for {scenario}.")
        p9_dir = (
            canary_root
            / "data"
            / "processed"
            / "synthetic"
            / scenario
            / "p9"
        )
        stability = pd.read_csv(p9_dir / "cluster_stability.csv")
        successful = stability.loc[stability["fit_status"].eq("PASS")]
        summary_rows.append(
            {
                "scenario": scenario,
                "assignment_rows": construction["assignment_rows"],
                "fit_rows": construction["fit_rows"],
                "cluster_count": construction["cluster_count"],
                "noise_or_unassigned_rate": (
                    construction["noise_or_unassigned_rate"]
                ),
                "successful_stability_refits": int(len(successful)),
                "stability_repetitions": int(len(stability)),
                "mean_stability_ari": checked["mean_stability_ari"],
                "validation_status": checked["status"],
                "validation_warning": " | ".join(checked["warnings"]),
            }
        )
        profiles = pd.read_csv(p9_dir / "archetype_profiles.csv")
        profiles.insert(0, "scenario", scenario)
        profile_frames.append(profiles)
        output_records.extend(
            [
                _file_record(p9_dir / "p9_manifest.json", canary_root),
                _file_record(
                    p9_dir / "clustering_metadata.json", canary_root
                ),
            ]
        )

    after = _main_manifest_hashes()
    frozen_unchanged = before == after
    if not frozen_unchanged:
        raise RuntimeError("Main manifests changed during isolated P18E.")
    p10_outputs = list(
        (canary_root / "data" / "processed" / "synthetic").glob("*/p10/*")
    )
    if p10_outputs:
        raise RuntimeError("P18E crossed the hard stop and created P10 outputs.")

    args.summary_table.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summary_rows).to_csv(
        args.summary_table, index=False, lineterminator="\n"
    )
    pd.concat(profile_frames, ignore_index=True).to_csv(
        args.profiles_table, index=False, lineterminator="\n"
    )
    result = {
        "schema_version": 1,
        "phase": "P18E",
        "version": "hypercube-isolated-descriptive-archetype-canary-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "isolated_canary": True,
        "scenarios": list(SCENARIOS),
        "contract": contract,
        "input_manifests": input_records,
        "isolated_reports": [
            _file_record(isolated_build, canary_root),
            _file_record(isolated_validation, canary_root),
        ],
        "isolated_outputs": output_records,
        "scenario_summaries": summary_rows,
        "main_manifest_hashes_before": before,
        "main_manifest_hashes_after": after,
        "frozen_main_manifests_unchanged": frozen_unchanged,
        "synthetic_truth_read": False,
        "return_models_refit": False,
        "performance_selection_applied": False,
        "p10_run": False,
        "real_data_run": False,
    }
    validation_result = {
        "schema_version": 1,
        "phase": "P18E",
        "status": "PASS",
        "errors": [],
        "scenario_validations": list(validations.values()),
        "canonical_code_matches": True,
        "frozen_contract_matches": True,
        "frozen_main_manifests_unchanged": frozen_unchanged,
        "synthetic_truth_read": False,
        "return_models_refit": False,
        "p10_run": False,
    }
    atomic_write_json(args.report, result)
    atomic_write_json(args.validation_report, validation_result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
