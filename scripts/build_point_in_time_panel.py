"""Build and validate the phase-P2 point-in-time panel bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence, cast

from hypercube.config import load_config
from hypercube.data import SCENARIOS, Scenario, atomic_write_json, scenario_output_dir
from hypercube.universe import (
    build_point_in_time_panel,
    scenario_processed_dir,
    validate_p2_directory,
)


def build_parser() -> argparse.ArgumentParser:
    """Create the phase-bounded P2 command parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic.yaml"))
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Build one or all synthetic P2 panels and independently validate them."""

    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if config.project.phase != "P2":
        raise SystemExit("Panel construction requires a P2 config.")
    if args.all_scenarios and (args.raw_dir is not None or args.output_dir is not None):
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
    for scenario in selected:
        if scenario == "real":
            raw_dir = args.raw_dir or Path(config.paths.raw_dir)
            output_dir = args.output_dir or Path(config.paths.processed_dir)
        else:
            typed_scenario = cast(Scenario, scenario)
            raw_dir = args.raw_dir or scenario_output_dir(config, typed_scenario)
            output_dir = args.output_dir or scenario_processed_dir(config, typed_scenario)
        diagnostics = build_point_in_time_panel(
            raw_dir, output_dir, config, scenario=scenario
        )
        validation = validate_p2_directory(output_dir, config)
        report = {
            "scenario": scenario,
            "raw_dir": str(raw_dir),
            "output_dir": str(output_dir),
            "construction": diagnostics,
            "validation": validation,
            "models_fitted": [],
        }
        reports.append(report)
        print(
            json.dumps(
                {
                    "status": validation["status"],
                    "scenario": scenario,
                    "output_dir": str(output_dir),
                    "rows": validation.get("rows", {}),
                    "availability_sources": validation.get(
                        "availability_sources", {}
                    ),
                    "delisting_events": validation.get("delisting_events", 0),
                    "errors": validation.get("errors", []),
                    "models_fitted": [],
                },
                sort_keys=True,
            )
        )
    payload = {
        "schema_version": 1,
        "phase": "P2",
        "config": str(args.config),
        "status": "PASS"
        if all(item["validation"]["status"] == "PASS" for item in reports)
        else "FAIL",
        "reports": reports,
        "models_fitted": [],
    }
    if args.report:
        atomic_write_json(args.report, payload)
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
