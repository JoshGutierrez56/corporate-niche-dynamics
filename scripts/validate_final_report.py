"""Validate P10 figures, report, README, and traceability receipts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from hypercube.config import load_config
from hypercube.data import atomic_write_json
from hypercube.visualization import validate_visualization_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    """Validate final P10 outputs without regenerating them."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic.yaml"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if config.project.phase != "P10":
        raise SystemExit("Final validation requires a P10 config.")
    validation = validate_visualization_bundle(PROJECT_ROOT)
    errors = list(validation.get("errors", []))
    for relative in ("README.md", "report.md", "docs/reproduction_guide.md"):
        path = PROJECT_ROOT / relative
        if not path.is_file() or path.stat().st_size < 200:
            errors.append(f"Missing or empty P10 document: {relative}")
    payload = {
        **validation,
        "phase": "P10",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "scientific_status": "SIGNAL_RECOVERY_FAILED",
        "real_data_run": False,
    }
    if args.report:
        atomic_write_json(args.report, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
