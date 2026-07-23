"""Run the synthetic P1-P10 pipeline in order with resumable validation gates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any, Sequence

import yaml

from hypercube.config import load_config
from hypercube.data import SCENARIOS, atomic_write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
P0_OUTPUT_DIRECTORIES = (
    Path("data/raw"),
    Path("data/interim"),
    Path("data/processed"),
    Path("artifacts/manifests"),
    Path("artifacts/models"),
    Path("artifacts/tables"),
    Path("artifacts/logs"),
    Path("figures"),
)


def _ensure_p0_output_directories() -> None:
    """Restore empty scaffold directories omitted by archive/copy workflows."""

    for relative in P0_OUTPUT_DIRECTORIES:
        (PROJECT_ROOT / relative).mkdir(parents=True, exist_ok=True)


def _atomic_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    except Exception:
        Path(name).unlink(missing_ok=True)
        raise


def _phase_config(base_path: Path, phase: str) -> Path:
    config = load_config(base_path).model_dump(mode="json")
    config["project"]["phase"] = phase
    config["project"]["run_name"] = f"synthetic_{phase.lower()}_full_pipeline"
    config["data"]["generate_synthetic"] = phase == "P1"
    config["data"]["require_existing_raw"] = phase != "P1"
    path = (
        PROJECT_ROOT
        / "artifacts"
        / "checkpoints"
        / "full_pipeline_configs"
        / f"{phase.lower()}.yaml"
    )
    _atomic_yaml(path, config)
    return path


def _all_exist(relative: str) -> bool:
    return all(
        (
            PROJECT_ROOT
            / "data"
            / ("raw" if relative.startswith("raw/") else "processed")
            / "synthetic"
            / scenario
            / relative.split("/", 1)[1]
        ).is_file()
        for scenario in SCENARIOS
    )


def _run(
    label: str,
    arguments: list[str],
    log_lines: list[str],
) -> None:
    command = [sys.executable, *arguments]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    block = (
        f"\n===== {label} =====\n"
        f"command: {' '.join(command)}\n"
        f"exit_code: {completed.returncode}\n"
        f"{completed.stdout}"
        f"{completed.stderr}"
    )
    log_lines.append(block)
    print(
        json.dumps(
            {
                "gate": label,
                "status": "PASS" if completed.returncode == 0 else "FAIL",
            },
            sort_keys=True,
        )
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Full-pipeline gate failed: {label}")


def main(argv: Sequence[str] | None = None) -> int:
    """Build missing phases, validate completed phases, and close P10."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic.yaml"))
    parser.add_argument(
        "--stop-after",
        choices=("P7", "P10"),
        default="P10",
        help="Stop after a validated P7 canary or continue through P10.",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if config.project.phase != "P10" or config.data.mode != "synthetic":
        raise SystemExit("Full reproduction requires the final P10 synthetic config.")
    _ensure_p0_output_directories()
    started = datetime.now(timezone.utc)
    logs: list[str] = []
    configs = {phase: _phase_config(args.config, phase) for phase in (
        "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"
    )}
    try:
        if not _all_exist("raw/data_manifest.json"):
            _run(
                "P1_BUILD",
                ["scripts/make_synthetic.py", "--config", str(configs["P1"]), "--all-scenarios"],
                logs,
            )
        _run(
            "P1_VALIDATE",
            [
                "scripts/validate_raw_inputs.py",
                "--config",
                str(configs["P1"]),
                "--all-scenarios",
                "--report",
                "artifacts/manifests/p1_validation.json",
            ],
            logs,
        )
        if not _all_exist("processed/p2_manifest.json"):
            _run(
                "P2_BUILD",
                [
                    "scripts/build_point_in_time_panel.py",
                    "--config",
                    str(configs["P2"]),
                    "--all-scenarios",
                    "--report",
                    "artifacts/manifests/p2_build.json",
                ],
                logs,
            )
        _run(
            "P2_VALIDATE",
            [
                "scripts/validate_point_in_time_panel.py",
                "--config",
                str(configs["P2"]),
                "--all-scenarios",
                "--report",
                "artifacts/manifests/p2_validation.json",
            ],
            logs,
        )
        if not _all_exist("processed/p3/p3_manifest.json"):
            _run(
                "P3_BUILD",
                [
                    "scripts/build_axis_features.py",
                    "--config",
                    str(configs["P3"]),
                    "--all-scenarios",
                    "--report",
                    "artifacts/manifests/p3_build.json",
                ],
                logs,
            )
        _run(
            "P3_VALIDATE",
            [
                "scripts/validate_axis_features.py",
                "--config",
                str(configs["P3"]),
                "--all-scenarios",
                "--report",
                "artifacts/manifests/p3_validation.json",
            ],
            logs,
        )
        build_validate = (
            (
                "P4",
                "processed/p4/p4_manifest.json",
                "scripts/build_viability_models.py",
                "scripts/validate_viability_models.py",
                "p4",
            ),
            (
                "P5",
                "processed/p5/p5_manifest.json",
                "scripts/build_dynamics.py",
                "scripts/validate_dynamics.py",
                "p5",
            ),
            (
                "P6",
                "processed/p6/p6_manifest.json",
                "scripts/build_survival_models.py",
                "scripts/validate_survival_models.py",
                "p6",
            ),
            (
                "P7",
                "processed/p7/p7_manifest.json",
                "scripts/build_return_tests.py",
                "scripts/validate_return_tests.py",
                "p7",
            ),
            (
                "P8",
                "processed/p8/p8_manifest.json",
                "scripts/build_cost_aware_portfolios.py",
                "scripts/validate_cost_aware_portfolios.py",
                "p8",
            ),
            (
                "P9",
                "processed/p9/p9_manifest.json",
                "scripts/build_archetypes.py",
                "scripts/validate_archetypes.py",
                "p9",
            ),
        )
        for phase, required, builder, validator, slug in build_validate:
            if not _all_exist(required):
                _run(
                    f"{phase}_BUILD",
                    [
                        builder,
                        "--config",
                        str(configs[phase]),
                        "--all-scenarios",
                        "--reuse-existing",
                        "--report",
                        f"artifacts/manifests/{slug}_build.json",
                    ],
                    logs,
                )
            _run(
                f"{phase}_VALIDATE",
                [
                    validator,
                    "--config",
                    str(configs[phase]),
                    "--all-scenarios",
                    "--report",
                    f"artifacts/manifests/{slug}_validation.json",
                ],
                logs,
            )
            if phase == args.stop_after:
                break
        if args.stop_after == "P7":
            _run(
                "P7_MANIFEST",
                [
                    "scripts/print_manifest.py",
                    "--config",
                    str(configs["P7"]),
                    "--output",
                    "artifacts/manifests/p7_manifest.json",
                ],
                logs,
            )
        else:
            final_receipt = PROJECT_ROOT / "artifacts" / "manifests" / "p10_build.json"
            if not final_receipt.is_file():
                _run(
                    "P10_BUILD",
                    [
                        "scripts/build_final_report.py",
                        "--config",
                        str(configs["P10"]),
                        "--report",
                        "artifacts/manifests/p10_build.json",
                    ],
                    logs,
                )
            _run(
                "P10_VALIDATE",
                [
                    "scripts/validate_final_report.py",
                    "--config",
                    str(configs["P10"]),
                    "--report",
                    "artifacts/manifests/p10_validation.json",
                ],
                logs,
            )
            _run("PYTEST", ["-m", "pytest", "-q"], logs)
            _run(
                "P10_MANIFEST",
                [
                    "scripts/print_manifest.py",
                    "--config",
                    str(configs["P10"]),
                    "--output",
                    "artifacts/manifests/p10_manifest.json",
                ],
                logs,
            )
        status = "PASS"
    except Exception:
        status = "FAIL"
        raise
    finally:
        finished = datetime.now(timezone.utc)
        log_path = PROJECT_ROOT / "artifacts" / "logs" / "full_pipeline.txt"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(
            prefix=".full-pipeline.", suffix=".tmp", dir=log_path.parent
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                handle.write("".join(logs))
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(name, log_path)
        except Exception:
            Path(name).unlink(missing_ok=True)
            raise
        receipt = {
            "schema_version": 1,
            "phase": args.stop_after,
            "status": status,
            "started_at_utc": started.isoformat(),
            "finished_at_utc": finished.isoformat(),
            "duration_seconds": (finished - started).total_seconds(),
            "config": str(args.config),
            "log": str(log_path),
            "phase_configs": {key: str(value) for key, value in configs.items()},
            "gpu_used": False,
            "network_used": False,
            "wrds_used": False,
            "real_data_run": False,
        }
        atomic_write_json(
            PROJECT_ROOT / "artifacts" / "manifests" / "full_pipeline_receipt.json",
            receipt,
        )
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
