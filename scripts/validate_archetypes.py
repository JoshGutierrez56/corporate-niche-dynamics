"""Independently validate saved phase-P9 archetype bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence, cast

from hypercube.clustering import scenario_p9_dir, validate_p9_directory
from hypercube.config import load_config
from hypercube.data import SCENARIOS, Scenario, atomic_write_json
from hypercube.universe import scenario_processed_dir


def main(argv: Sequence[str] | None = None) -> int:
    """Validate one real or one/all synthetic P9 output directories."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic.yaml"))
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--p2-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if config.project.phase not in {"P9", "P10"}:
        raise SystemExit("Archetype validation requires a P9 or P10 config.")
    if args.all_scenarios and any(
        item is not None for item in (args.p2_dir, args.output_dir)
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
            output_dir = args.output_dir or p2_dir / "p9"
        else:
            typed = cast(Scenario, scenario)
            p2_dir = args.p2_dir or scenario_processed_dir(config, typed)
            output_dir = args.output_dir or scenario_p9_dir(config, typed)
        validation = validate_p9_directory(
            output_dir,
            p2_dir / "p3",
            config,
            scenario=scenario,
        )
        reports.append({"scenario": scenario, "output_dir": str(output_dir), **validation})
        print(json.dumps({"scenario": scenario, **validation}, sort_keys=True))
    payload = {
        "schema_version": 1,
        "phase": "P9",
        "status": "PASS" if all(item["status"] == "PASS" for item in reports) else "FAIL",
        "reports": reports,
        "descriptive_only": True,
    }
    if args.report:
        atomic_write_json(args.report, payload)
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
