"""Independently validate saved phase-P5 dynamics bundles."""

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
from hypercube.dynamics import scenario_p5_dir, validate_p5_directory
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


def main(argv: Sequence[str] | None = None) -> int:
    """Validate one real or one/all synthetic P5 output directories."""

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
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if config.project.phase not in {"P5", "P6", "P7", "P8", "P9", "P10"}:
        raise SystemExit("Dynamics validation requires a P5-P7 config.")
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
        selected = ("real",)

    reports = []
    recovery_rows = []
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
        validation = validate_p5_directory(
            output_dir,
            p2_dir,
            p3_dir,
            p4_dir,
            raw_dir,
            config,
            synthetic_truth_path=truth_path,
        )
        reports.append({"scenario": scenario, "output_dir": str(output_dir), **validation})
        if validation.get("recovery"):
            recovery_rows.append({"scenario": scenario, **validation["recovery"]})
        print(json.dumps({"scenario": scenario, **validation}, sort_keys=True))
    recovery_path = PROJECT_ROOT / "artifacts" / "tables" / "p5_recovery_metrics.csv"
    if recovery_rows:
        _atomic_csv(pd.json_normalize(recovery_rows, sep="_"), recovery_path)
    payload = {
        "schema_version": 1,
        "phase": "P5",
        "status": "PASS" if all(item["status"] == "PASS" for item in reports) else "FAIL",
        "reports": reports,
        "recovery_table": str(recovery_path) if recovery_rows else None,
        "return_test_run": False,
        "portfolio_run": False,
        "clustering_run": False,
    }
    if args.report:
        atomic_write_json(args.report, payload)
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
