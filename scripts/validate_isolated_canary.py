"""Independently validate P16E realized-canary outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from hypercube.canary import validate_realized_canary_outputs
from hypercube.data import atomic_write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANARY_ROOT = PROJECT_ROOT.parent / ".p16-hypercube-canary"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canary-root", type=Path, default=DEFAULT_CANARY_ROOT)
    parser.add_argument(
        "--migration-candidates",
        type=Path,
        default=Path("artifacts/tables/p16e_migration_proxy_candidates.parquet"),
    )
    parser.add_argument(
        "--null-candidates",
        type=Path,
        default=Path("artifacts/tables/p16e_null_proxy_candidates.parquet"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("artifacts/manifests/p16e_canary_evaluation.json"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("artifacts/tables/p16e_canary_metrics.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/manifests/p16e_canary_validation.json"),
    )
    args = parser.parse_args(argv)
    resolve = lambda path: path if path.is_absolute() else PROJECT_ROOT / path
    canary = args.canary_root.resolve()
    report = validate_realized_canary_outputs(
        canary
        / "data/processed/synthetic/migration_alpha/p7/"
        "forward_return_targets.parquet",
        canary / "data/raw/synthetic/migration_alpha/synthetic_truth.parquet",
        resolve(args.migration_candidates),
        canary
        / "data/processed/synthetic/null_alpha/p7/"
        "forward_return_targets.parquet",
        resolve(args.null_candidates),
        resolve(args.summary),
        resolve(args.metrics),
    )
    atomic_write_json(resolve(args.report), report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
