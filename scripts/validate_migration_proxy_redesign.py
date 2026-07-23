"""Independently validate the corrected P13F proxy redesign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from hypercube.data import atomic_write_json
from hypercube.proxy_redesign import validate_proxy_redesign_outputs


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--p5",
        type=Path,
        default=Path(
            "data/processed/synthetic/migration_alpha/p5/"
            "frontier_dynamics.parquet"
        ),
    )
    parser.add_argument(
        "--truth",
        type=Path,
        default=Path("data/raw/synthetic/migration_alpha/synthetic_truth.parquet"),
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        default=Path("artifacts/tables/p13f_proxy_candidates.parquet"),
    )
    parser.add_argument(
        "--coefficients",
        type=Path,
        default=Path("artifacts/tables/p13f_proxy_ar_coefficients.csv"),
    )
    parser.add_argument(
        "--build-receipt",
        type=Path,
        default=Path("artifacts/manifests/p13f_proxy_build.json"),
    )
    parser.add_argument(
        "--evaluation",
        type=Path,
        default=Path("artifacts/manifests/p13f_proxy_evaluation.json"),
    )
    parser.add_argument(
        "--metrics",
        type=Path,
        default=Path("artifacts/tables/p13f_proxy_evaluation.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/manifests/p13f_proxy_validation.json"),
    )
    args = parser.parse_args(argv)
    resolve = lambda path: path if path.is_absolute() else PROJECT_ROOT / path
    report = validate_proxy_redesign_outputs(
        resolve(args.p5),
        resolve(args.truth),
        resolve(args.candidates),
        resolve(args.coefficients),
        resolve(args.build_receipt),
        resolve(args.evaluation),
        resolve(args.metrics),
    )
    atomic_write_json(resolve(args.report), report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
