"""Independently validate the corrected P14F eligibility audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from hypercube.data import atomic_write_json
from hypercube.proxy_redesign import validate_p7_eligibility_audit


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("artifacts/tables/p13f_proxy_candidates.parquet"),
    )
    parser.add_argument(
        "--targets",
        type=Path,
        default=Path(
            "data/processed/synthetic/migration_alpha/p7/"
            "forward_return_targets.parquet"
        ),
    )
    parser.add_argument(
        "--truth",
        type=Path,
        default=Path("data/raw/synthetic/migration_alpha/synthetic_truth.parquet"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("artifacts/manifests/p14f_p7_eligibility_audit.json"),
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=Path("artifacts/tables/p14f_p7_eligibility_audit.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/manifests/p14f_p7_eligibility_validation.json"),
    )
    args = parser.parse_args(argv)
    resolve = lambda path: path if path.is_absolute() else PROJECT_ROOT / path
    report = validate_p7_eligibility_audit(
        resolve(args.candidates),
        resolve(args.targets),
        resolve(args.truth),
        resolve(args.summary),
        resolve(args.table),
    )
    atomic_write_json(resolve(args.report), report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
