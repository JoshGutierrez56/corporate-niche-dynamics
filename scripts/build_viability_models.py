"""Build and validate phase-P4 fixed-horizon viability models."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Sequence, cast

import pandas as pd

from hypercube.axes import scenario_p3_dir
from hypercube.config import load_config
from hypercube.data import SCENARIOS, Scenario, atomic_write_json, scenario_output_dir
from hypercube.viability import (
    build_viability_bundle,
    scenario_p4_dir,
    validate_p4_directory,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        frame.to_csv(name, index=False, lineterminator="\n")
        os.replace(name, path)
    except Exception:
        Path(name).unlink(missing_ok=True)
        raise


def _aggregate(outputs: list[tuple[str, Path]]) -> list[str]:
    mappings = {
        "fold_metrics.csv": "p4_fold_metrics.csv",
        "benchmark_comparisons.csv": "p4_benchmark_comparisons.csv",
        "label_summary.csv": "p4_label_summary.csv",
        "hyperparameter_trials.csv": "p4_hyperparameter_trials.csv",
    }
    written: list[str] = []
    for source, target in mappings.items():
        pieces = []
        for scenario, output in outputs:
            piece = pd.read_csv(output / source)
            piece.insert(0, "scenario", scenario)
            pieces.append(piece)
        path = PROJECT_ROOT / "artifacts" / "tables" / target
        _atomic_csv(pd.concat(pieces, ignore_index=True), path)
        written.append(str(path))
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """Build one real or one/all synthetic P4 model bundles."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic.yaml"))
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--p2-dir", type=Path)
    parser.add_argument("--p3-dir", type=Path)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Revalidate immutable completed bundles instead of overwriting them.",
    )
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if config.project.phase != "P4":
        raise SystemExit("Viability modeling requires a P4 config.")
    explicit = (args.p2_dir, args.p3_dir, args.raw_dir, args.output_dir)
    if args.all_scenarios and any(item is not None for item in explicit):
        raise SystemExit("Explicit directories cannot be combined with --all-scenarios.")
    if config.data.mode == "synthetic":
        selected: tuple[str, ...] = (
            tuple(SCENARIOS)
            if args.all_scenarios
            else (cast(Scenario, args.scenario or config.data.scenario),)
        )
    else:
        if args.scenario:
            raise SystemExit("Real mode does not accept synthetic scenarios.")
        selected = ("real",)

    reports: list[dict[str, object]] = []
    outputs: list[tuple[str, Path]] = []
    for scenario in selected:
        if scenario == "real":
            p2_dir = args.p2_dir or Path(config.paths.processed_dir)
            p3_dir = args.p3_dir or p2_dir / "p3"
            raw_dir = args.raw_dir or Path(config.paths.raw_dir)
            output_dir = args.output_dir or p2_dir / "p4"
        else:
            typed = cast(Scenario, scenario)
            p3_dir = args.p3_dir or scenario_p3_dir(config, typed)
            p2_dir = args.p2_dir or p3_dir.parent
            raw_dir = args.raw_dir or scenario_output_dir(config, typed)
            output_dir = args.output_dir or scenario_p4_dir(config, typed)
        reused = output_dir.exists() and any(output_dir.iterdir())
        if reused:
            if not args.reuse_existing:
                raise FileExistsError(
                    f"Completed P4 output exists; use --reuse-existing to revalidate: {output_dir}"
                )
            validation = validate_p4_directory(
                output_dir, p2_dir, p3_dir, raw_dir, config
            )
            if validation["status"] != "PASS":
                raise RuntimeError(f"Existing P4 bundle failed validation: {validation['errors']}")
            diagnostics = json.loads(
                (output_dir / "p4_diagnostics.json").read_text(encoding="utf-8")
            )
        else:
            diagnostics = build_viability_bundle(
                p2_dir,
                p3_dir,
                raw_dir,
                output_dir,
                config,
                scenario=scenario,
            )
            validation = validate_p4_directory(
                output_dir, p2_dir, p3_dir, raw_dir, config
            )
        reports.append(
            {
                "scenario": scenario,
                "output_dir": str(output_dir),
                "construction": diagnostics,
                "validation": validation,
                "reused_existing": reused,
            }
        )
        outputs.append((scenario, output_dir))
        print(
            json.dumps(
                {
                    "status": validation["status"],
                    "scenario": scenario,
                    "output_dir": str(output_dir),
                    "rows": validation.get("rows", {}),
                    "errors": validation.get("errors", []),
                    "models_fitted": validation.get("models_fitted", []),
                    "return_test_run": False,
                },
                sort_keys=True,
            )
        )
    tables = _aggregate(outputs)
    payload = {
        "schema_version": 1,
        "phase": "P4",
        "config": str(args.config),
        "status": "PASS"
        if all(item["validation"]["status"] == "PASS" for item in reports)
        else "FAIL",
        "reports": reports,
        "aggregate_tables": tables,
        "return_test_run": False,
        "portfolio_run": False,
    }
    if args.report:
        atomic_write_json(args.report, payload)
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
