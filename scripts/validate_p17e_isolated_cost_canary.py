"""Independently validate both saved P17E cost-canary scenarios."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from hypercube.canary_costs import (
    SCENARIOS,
    validate_locked_proxy_cost_scenario,
)
from hypercube.config import load_config
from hypercube.data import atomic_write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANARY_ROOT = Path(
    os.environ.get(
        "HYPERCUBE_CANARY_ROOT",
        PROJECT_ROOT.parent / ".p16-hypercube-canary",
    )
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--canary-root",
        type=Path,
        default=DEFAULT_CANARY_ROOT,
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "manifests"
        / "p17e_cost_canary_validation.json",
    )
    args = parser.parse_args()
    config = load_config(args.canary_root / "configs" / "synthetic.yaml")
    reports = []
    errors = []
    for scenario in SCENARIOS:
        output_dir = (
            args.canary_root
            / "data"
            / "processed"
            / "synthetic"
            / scenario
            / "p17e_p8"
        )
        report = validate_locked_proxy_cost_scenario(output_dir, config)
        reports.append(report)
        errors.extend(
            f"{scenario}: {error}" for error in report.get("errors", [])
        )
    payload = {
        "schema_version": 1,
        "phase": "P17E",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "scenario_validations": reports,
        "synthetic_truth_read": False,
        "performance_gate_applied": False,
    }
    atomic_write_json(args.report, payload)
    if errors:
        raise SystemExit("\n".join(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
