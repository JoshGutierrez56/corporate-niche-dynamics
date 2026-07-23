"""Validate P1 raw schemas, keys, dates, and cross-file relationships."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence, cast

from hypercube.config import load_config
from hypercube.data import (
    SCENARIOS,
    Scenario,
    atomic_write_json,
    scenario_output_dir,
    validate_raw_directory,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the raw-input validation command parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic.yaml"))
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--raw-dir", type=Path)
    parser.add_argument("--report", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate configured local inputs without fitting any model."""

    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.raw_dir is not None and args.all_scenarios:
        raise SystemExit("--raw-dir cannot be combined with --all-scenarios.")
    if config.data.mode == "synthetic":
        selected = (
            SCENARIOS
            if args.all_scenarios
            else (cast(Scenario, args.scenario or config.data.scenario),)
        )
        locations = [args.raw_dir or scenario_output_dir(config, item) for item in selected]
    else:
        selected = ("real",)
        locations = [args.raw_dir or Path(config.paths.raw_dir)]

    reports = []
    for label, raw_dir in zip(selected, locations, strict=True):
        report = validate_raw_directory(raw_dir)
        report["scenario"] = label
        reports.append(report)
        print(
            json.dumps(
                {
                    "status": report["status"],
                    "scenario": label,
                    "raw_dir": str(raw_dir),
                    "row_counts": {
                        name: details["rows"]
                        for name, details in report.get("tables", {}).items()
                    },
                    "errors": report.get("errors", []),
                    "warnings": report.get("warnings", []),
                    "models_fitted": [],
                },
                sort_keys=True,
            )
        )
    payload = {
        "schema_version": 1,
        "phase": "P1",
        "config": str(args.config),
        "status": "PASS" if all(item["status"] == "PASS" for item in reports) else "FAIL",
        "reports": reports,
        "models_fitted": [],
    }
    if args.report:
        atomic_write_json(args.report, payload)
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
