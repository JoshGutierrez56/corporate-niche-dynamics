"""Build a compact hash manifest for a post-P10 exploratory phase."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from hypercube.data import atomic_write_json, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--gate-result", required=True)
    parser.add_argument("--selected-candidate")
    parser.add_argument("--selected-multiplier", type=float)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args(argv)

    records = []
    for supplied in args.files:
        path = supplied if supplied.is_absolute() else PROJECT_ROOT / supplied
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = {
        "schema_version": 1,
        "phase": args.phase,
        "version": args.version,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": args.status,
        "gate_result": args.gate_result,
        "selected_candidate": args.selected_candidate,
        "selected_multiplier": args.selected_multiplier,
        "new_synthetic_scenario_generated": False,
        "frozen_p0_p10_outputs_modified": False,
        "real_data_run": False,
        "files": records,
    }
    output = args.output if args.output.is_absolute() else PROJECT_ROOT / args.output
    atomic_write_json(output, payload)
    print(f"{args.phase}_MANIFEST_WRITTEN {output} {len(records)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
