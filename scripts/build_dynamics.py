"""Build and validate phase-P5 frontier and niche-dynamics features."""

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
from hypercube.dynamics import (
    build_dynamics_bundle,
    scenario_p5_dir,
    validate_p5_directory,
)
from hypercube.viability import scenario_p4_dir


PROJECT_ROOT = Path(__file__).resolve().parents[1]


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


def _aggregate(outputs: list[tuple[str, Path]]) -> list[str]:
    summary_rows = []
    model_rows = []
    for scenario, output in outputs:
        diagnostics = json.loads(
            (output / "p5_diagnostics.json").read_text(encoding="utf-8")
        )
        summary_rows.append(diagnostics)
        models = pd.read_csv(output / "migration_model_diagnostics.csv")
        models.insert(0, "scenario", scenario)
        model_rows.append(models)
    summary_path = PROJECT_ROOT / "artifacts" / "tables" / "p5_feature_summary.csv"
    model_path = (
        PROJECT_ROOT / "artifacts" / "tables" / "p5_migration_model_diagnostics.csv"
    )
    _atomic_csv(pd.DataFrame(summary_rows), summary_path)
    _atomic_csv(pd.concat(model_rows, ignore_index=True), model_path)
    return [str(summary_path), str(model_path)]


def main(argv: Sequence[str] | None = None) -> int:
    """Build one real or one/all synthetic P5 dynamics bundles."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic.yaml"))
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--p2-dir", type=Path)
    parser.add_argument("--p3-dir", type=Path)
    parser.add_argument("--p4-dir", type=Path)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if config.project.phase != "P5":
        raise SystemExit("Dynamics construction requires a P5 config.")
    explicit = (args.p2_dir, args.p3_dir, args.p4_dir, args.raw_dir, args.output_dir)
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
            p4_dir = args.p4_dir or p2_dir / "p4"
            raw_dir = args.raw_dir or Path(config.paths.raw_dir)
            output_dir = args.output_dir or p2_dir / "p5"
            truth_path = None
        else:
            typed = cast(Scenario, scenario)
            p3_dir = args.p3_dir or scenario_p3_dir(config, typed)
            p2_dir = args.p2_dir or p3_dir.parent
            p4_dir = args.p4_dir or scenario_p4_dir(config, typed)
            raw_dir = args.raw_dir or scenario_output_dir(config, typed)
            output_dir = args.output_dir or scenario_p5_dir(config, typed)
            truth_path = raw_dir / "synthetic_truth.parquet"
        reused = output_dir.exists() and any(output_dir.iterdir())
        if reused:
            if not args.reuse_existing:
                raise FileExistsError(
                    f"Completed P5 output exists; use --reuse-existing: {output_dir}"
                )
            diagnostics = json.loads(
                (output_dir / "p5_diagnostics.json").read_text(encoding="utf-8")
            )
        else:
            diagnostics = build_dynamics_bundle(
                p2_dir,
                p3_dir,
                p4_dir,
                raw_dir,
                output_dir,
                config,
                scenario=scenario,
            )
        validation = validate_p5_directory(
            output_dir,
            p2_dir,
            p3_dir,
            p4_dir,
            raw_dir,
            config,
            synthetic_truth_path=truth_path,
        )
        if validation["status"] != "PASS":
            raise RuntimeError(f"P5 bundle failed validation: {validation['errors']}")
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
                    "recovery": validation.get("recovery"),
                    "return_test_run": False,
                },
                sort_keys=True,
            )
        )
    tables = _aggregate(outputs)
    payload = {
        "schema_version": 1,
        "phase": "P5",
        "config": str(args.config),
        "status": "PASS",
        "reports": reports,
        "aggregate_tables": tables,
        "return_test_run": False,
        "portfolio_run": False,
        "clustering_run": False,
    }
    if args.report:
        atomic_write_json(args.report, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
