"""Generate deterministic P1 synthetic WRDS-shaped raw parquets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence, cast

from hypercube.config import load_config
from hypercube.data import SCENARIOS, Scenario, generate_synthetic_bundle, scenario_output_dir


def build_parser() -> argparse.ArgumentParser:
    """Build the P1 synthetic-generation command parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic.yaml"))
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--n-firms", type=int)
    parser.add_argument("--start-year", type=int)
    parser.add_argument("--end-year", type=int)
    parser.add_argument("--seed", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Generate one scenario or the three predeclared P1 scenarios."""

    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if config.project.phase != "P1" or config.data.mode != "synthetic":
        raise SystemExit("Synthetic generation requires a P1 synthetic config.")
    if args.output_dir is not None and args.all_scenarios:
        raise SystemExit("--output-dir cannot be combined with --all-scenarios.")
    selected = (
        SCENARIOS
        if args.all_scenarios
        else (cast(Scenario, args.scenario or config.data.scenario),)
    )
    results = []
    for offset, scenario in enumerate(selected):
        output_dir = args.output_dir or scenario_output_dir(config, scenario)
        result = generate_synthetic_bundle(
            config,
            scenario,
            output_dir,
            n_firms=args.n_firms,
            start_year=args.start_year,
            end_year=args.end_year,
            seed=(args.seed + offset if args.seed is not None else config.project.seed + offset),
        )
        results.append(result)
        print(
            json.dumps(
                {
                    "status": "P1_SYNTHETIC_GENERATED",
                    "scenario": scenario,
                    "output_dir": result["output_dir"],
                    "row_counts": result["row_counts"],
                    "injection": result["injection"],
                    "expected_signs": result["expected_signs"],
                    "models_fitted": [],
                },
                sort_keys=True,
            )
        )
    print(
        json.dumps(
            {
                "status": "P1_GENERATION_COMPLETE",
                "scenarios": [result["scenario"] for result in results],
                "models_fitted": [],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
