"""Independently validate saved phase-P2 point-in-time outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence, cast

from hypercube.config import load_config
from hypercube.data import SCENARIOS, Scenario, atomic_write_json
from hypercube.universe import scenario_processed_dir, validate_p2_directory


def main(argv: Sequence[str] | None = None) -> int:
    """Validate one or all configured P2 output directories."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic.yaml"))
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if config.project.phase not in {"P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"}:
        raise SystemExit("Panel validation requires a P2-or-later panel config.")
    if args.output_dir is not None and args.all_scenarios:
        raise SystemExit("--output-dir cannot be combined with --all-scenarios.")
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
            output_dir = args.output_dir or Path(config.paths.processed_dir)
        else:
            output_dir = args.output_dir or scenario_processed_dir(
                config, cast(Scenario, scenario)
            )
        report = validate_p2_directory(output_dir, config)
        report["scenario"] = scenario
        report["output_dir"] = str(output_dir)
        reports.append(report)
        print(json.dumps(report, sort_keys=True))
    payload = {
        "schema_version": 1,
        "phase": "P2",
        "status": "PASS" if all(item["status"] == "PASS" for item in reports) else "FAIL",
        "reports": reports,
        "models_fitted": [],
    }
    if args.report:
        atomic_write_json(args.report, payload)
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
