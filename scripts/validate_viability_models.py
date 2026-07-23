"""Independently validate saved phase-P4 viability bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence, cast

from hypercube.axes import scenario_p3_dir
from hypercube.config import load_config
from hypercube.data import SCENARIOS, Scenario, atomic_write_json, scenario_output_dir
from hypercube.viability import scenario_p4_dir, validate_p4_directory


def main(argv: Sequence[str] | None = None) -> int:
    """Validate one real or one/all synthetic P4 output directories."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic.yaml"))
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--p2-dir", type=Path)
    parser.add_argument("--p3-dir", type=Path)
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if config.project.phase not in {"P4", "P5", "P6", "P7", "P8", "P9", "P10"}:
        raise SystemExit("Viability validation requires a P4-P7 config.")
    if args.all_scenarios and any(
        item is not None for item in (args.p2_dir, args.p3_dir, args.raw_dir, args.output_dir)
    ):
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
        report = validate_p4_directory(output_dir, p2_dir, p3_dir, raw_dir, config)
        report["scenario"] = scenario
        report["output_dir"] = str(output_dir)
        reports.append(report)
        print(json.dumps(report, sort_keys=True))
    payload = {
        "schema_version": 1,
        "phase": "P4",
        "status": "PASS" if all(item["status"] == "PASS" for item in reports) else "FAIL",
        "reports": reports,
        "return_test_run": False,
        "portfolio_run": False,
    }
    if args.report:
        atomic_write_json(args.report, payload)
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
