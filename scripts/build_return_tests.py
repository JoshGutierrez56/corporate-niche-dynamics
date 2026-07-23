"""Build and validate phase-P7 point-in-time return-test bundles."""

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
from hypercube.returns import (
    build_return_bundle,
    scenario_p7_dir,
    validate_p7_directory,
)
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
        "fmb_summary.csv": "p7_fmb_summary.csv",
        "portfolio_factor_results.csv": "p7_portfolio_factor_results.csv",
        "portfolio_exposures.csv": "p7_portfolio_exposures.csv",
        "subsample_results.csv": "p7_subsample_results.csv",
        "target_attrition.csv": "p7_target_attrition.csv",
    }
    written = []
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
    """Build one real or one/all synthetic P7 bundles."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic.yaml"))
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--p2-dir", type=Path)
    parser.add_argument("--p5-dir", type=Path)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if config.project.phase != "P7":
        raise SystemExit("Return testing requires a P7 config.")
    explicit = (args.p2_dir, args.p5_dir, args.raw_dir, args.output_dir)
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
    reports = []
    outputs = []
    for scenario in selected:
        if scenario == "real":
            p2_dir = args.p2_dir or Path(config.paths.processed_dir)
            p5_dir = args.p5_dir or p2_dir / "p5"
            raw_dir = args.raw_dir or Path(config.paths.raw_dir)
            output_dir = args.output_dir or p2_dir / "p7"
            truth_path = None
        else:
            typed = cast(Scenario, scenario)
            p2_dir = args.p2_dir or scenario_processed_dir(config, typed)
            p5_dir = args.p5_dir or scenario_p5_dir(config, typed)
            raw_dir = args.raw_dir or scenario_output_dir(config, typed)
            output_dir = args.output_dir or scenario_p7_dir(config, typed)
            truth_path = raw_dir / "synthetic_truth.parquet"
        reused = output_dir.exists() and any(output_dir.iterdir())
        if reused:
            if not args.reuse_existing:
                raise FileExistsError(
                    f"Completed P7 output exists; use --reuse-existing: {output_dir}"
                )
            construction = json.loads(
                (output_dir / "return_metadata.json").read_text(encoding="utf-8")
            )
        else:
            construction = build_return_bundle(
                p2_dir,
                p5_dir,
                raw_dir,
                output_dir,
                config,
                scenario=scenario,
            )
        validation = validate_p7_directory(
            output_dir,
            p2_dir,
            p5_dir,
            raw_dir,
            config,
            scenario=scenario,
            synthetic_truth_path=truth_path,
        )
        if validation["status"] != "PASS":
            raise RuntimeError(f"P7 bundle failed validation: {validation['errors']}")
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
        "phase": "P7",
        "config": str(args.config),
        "status": "PASS",
        "reports": reports,
        "aggregate_tables": tables,
        "return_test_run": True,
        "portfolio_run": True,
        "costs_applied": False,
    }
    if args.report:
        atomic_write_json(args.report, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
