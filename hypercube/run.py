"""Phase-bounded command-line entry point for configuration validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from hypercube.config import load_config


def build_parser() -> argparse.ArgumentParser:
    """Create the phase-bounded validation-only command parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate one config without starting a downstream empirical phase."""

    args = build_parser().parse_args(argv)
    if not args.validate_only:
        raise SystemExit("Use --validate-only; empirical execution is phase-gated.")
    config = load_config(args.config)
    print(
        json.dumps(
            {
                "status": f"{config.project.phase}_CONFIG_VALID",
                "config": str(args.config),
                "phase": config.project.phase,
                "data_mode": config.data.mode,
                "run_name": config.project.run_name,
                "seed": config.project.seed,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
