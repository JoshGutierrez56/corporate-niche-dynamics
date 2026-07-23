"""Build and validate phase-P8 cost-aware portfolio bundles."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Sequence, cast

import pandas as pd

from hypercube.config import load_config
from hypercube.data import SCENARIOS, Scenario, atomic_write_json, scenario_output_dir
from hypercube.dynamics import scenario_p5_dir
from hypercube.portfolio import (
    build_cost_aware_bundle,
    scenario_p8_dir,
    validate_p8_directory,
)
from hypercube.returns import scenario_p7_dir
from hypercube.universe import scenario_processed_dir


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
    mappings = {
        "cost_aware_summary.csv": "p8_cost_aware_summary.csv",
        "capacity_diagnostics.csv": "p8_capacity_diagnostics.csv",
        "cost_waterfall.csv": "p8_cost_waterfall.csv",
        "delayed_execution_sensitivity.csv": "p8_delayed_execution_sensitivity.csv",
    }
    written: list[str] = []
    for source, target in mappings.items():
        pieces = []
        for scenario, output in outputs:
            frame = pd.read_csv(output / source)
            frame.insert(0, "scenario", scenario)
            pieces.append(frame)
        path = PROJECT_ROOT / "artifacts" / "tables" / target
        _atomic_csv(pd.concat(pieces, ignore_index=True), path)
        written.append(str(path))
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """Build one real or one/all synthetic P8 bundles."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic.yaml"))
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--p2-dir", type=Path)
    parser.add_argument("--p5-dir", type=Path)
    parser.add_argument("--p7-dir", type=Path)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if config.project.phase != "P8":
        raise SystemExit("Cost-aware portfolio construction requires a P8 config.")
    explicit = (
        args.p2_dir,
        args.p5_dir,
        args.p7_dir,
        args.raw_dir,
        args.output_dir,
    )
    if args.all_scenarios and any(item is not None for item in explicit):
        raise SystemExit("Explicit directories cannot be combined with --all-scenarios.")
    if config.data.mode == "synthetic":
        selected: tuple[str, ...] = (
            tuple(SCENARIOS)
            if args.all_scenarios
            else (cast(Scenario, args.scenario or config.data.scenario),)
        )
    else:
        selected = ("real",)
    reports: list[dict[str, object]] = []
    outputs: list[tuple[str, Path]] = []
    for scenario in selected:
        if scenario == "real":
            p2_dir = args.p2_dir or Path(config.paths.processed_dir)
            p5_dir = args.p5_dir or p2_dir / "p5"
            p7_dir = args.p7_dir or p2_dir / "p7"
            raw_dir = args.raw_dir or Path(config.paths.raw_dir)
            output_dir = args.output_dir or p2_dir / "p8"
        else:
            typed = cast(Scenario, scenario)
            p2_dir = args.p2_dir or scenario_processed_dir(config, typed)
            p5_dir = args.p5_dir or scenario_p5_dir(config, typed)
            p7_dir = args.p7_dir or scenario_p7_dir(config, typed)
            raw_dir = args.raw_dir or scenario_output_dir(config, typed)
            output_dir = args.output_dir or scenario_p8_dir(config, typed)
        reused = output_dir.exists() and any(output_dir.iterdir())
        if reused:
            if not args.reuse_existing:
                raise FileExistsError(
                    f"Completed P8 output exists; use --reuse-existing: {output_dir}"
                )
            construction = json.loads(
                (output_dir / "cost_metadata.json").read_text(encoding="utf-8")
            )
        else:
            construction = build_cost_aware_bundle(
                p2_dir,
                p5_dir,
                p7_dir,
                raw_dir,
                output_dir,
                config,
                scenario=scenario,
            )
        validation = validate_p8_directory(
            output_dir,
            p7_dir,
            config,
            scenario=scenario,
        )
        if validation["status"] != "PASS":
            raise RuntimeError(f"P8 bundle failed validation: {validation['errors']}")
        reports.append(
            {
                "scenario": scenario,
                "output_dir": str(output_dir),
                "construction": construction,
                "validation": validation,
                "reused_existing": reused,
            }
        )
        outputs.append((scenario, output_dir))
        print(json.dumps({"scenario": scenario, **validation}, sort_keys=True))
    tables = _aggregate(outputs)
    payload = {
        "schema_version": 1,
        "phase": "P8",
        "config": str(args.config),
        "status": "PASS",
        "reports": reports,
        "aggregate_tables": tables,
        "costs_applied": True,
        "capacity_applied": True,
        "borrow_applied": True,
        "return_test_rerun": False,
    }
    if args.report:
        atomic_write_json(args.report, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
