"""Independently validate saved phase-P3 component and axis bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence, cast

from hypercube.axes import scenario_p3_dir, validate_p3_directory
from hypercube.config import load_config
from hypercube.data import SCENARIOS, Scenario, atomic_write_json


def main(argv: Sequence[str] | None = None) -> int:
    """Validate one real or one/all synthetic P3 output directories."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic.yaml"))
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if config.project.phase not in {"P3", "P4", "P5", "P6", "P7", "P8", "P9", "P10"}:
        raise SystemExit("Axis validation requires a P3-or-later feature config.")
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
        output = (
            args.output_dir or Path(config.paths.processed_dir) / "p3"
            if scenario == "real"
            else args.output_dir
            or scenario_p3_dir(config, cast(Scenario, scenario))
        )
        report = validate_p3_directory(output, config)
        report["scenario"] = scenario
        report["output_dir"] = str(output)
        reports.append(report)
        print(json.dumps(report, sort_keys=True))
    payload = {
        "schema_version": 1,
        "phase": "P3",
        "status": "PASS" if all(item["status"] == "PASS" for item in reports) else "FAIL",
        "reports": reports,
        "models_fitted": [],
    }
    if args.report:
        atomic_write_json(args.report, payload)
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
