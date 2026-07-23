"""Independently validate saved phase-P8 cost-aware portfolio bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence, cast

from hypercube.config import load_config
from hypercube.data import SCENARIOS, Scenario, atomic_write_json
from hypercube.portfolio import scenario_p8_dir, validate_p8_directory
from hypercube.returns import scenario_p7_dir
from hypercube.universe import scenario_processed_dir


def main(argv: Sequence[str] | None = None) -> int:
    """Validate one real or one/all synthetic P8 output directories."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic.yaml"))
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--p2-dir", type=Path)
    parser.add_argument("--p7-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if config.project.phase not in {"P8", "P9", "P10"}:
        raise SystemExit("Cost-aware portfolio validation requires a P8 or later config.")
    if args.all_scenarios and any(
        item is not None for item in (args.p2_dir, args.p7_dir, args.output_dir)
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
            p7_dir = args.p7_dir or p2_dir / "p7"
            output_dir = args.output_dir or p2_dir / "p8"
        else:
            typed = cast(Scenario, scenario)
            p2_dir = args.p2_dir or scenario_processed_dir(config, typed)
            p7_dir = args.p7_dir or scenario_p7_dir(config, typed)
            output_dir = args.output_dir or scenario_p8_dir(config, typed)
        validation = validate_p8_directory(
            output_dir,
            p7_dir,
            config,
            scenario=scenario,
        )
        reports.append({"scenario": scenario, "output_dir": str(output_dir), **validation})
        print(json.dumps({"scenario": scenario, **validation}, sort_keys=True))
    payload = {
        "schema_version": 1,
        "phase": "P8",
        "status": "PASS" if all(item["status"] == "PASS" for item in reports) else "FAIL",
        "reports": reports,
        "costs_applied": True,
    }
    if args.report:
        atomic_write_json(args.report, payload)
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
