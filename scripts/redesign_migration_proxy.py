"""Build and evaluate the corrected P13F migration-proxy candidates."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

from hypercube.data import atomic_write_json, sha256_file
from hypercube.exploratory import atomic_write_csv
from hypercube.proxy_redesign import (
    atomic_write_parquet,
    build_proxy_candidates,
    evaluate_proxy_candidates,
)


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
    args = parser.parse_args(argv)
    resolve = lambda path: path if path.is_absolute() else PROJECT_ROOT / path

    candidates, coefficients, receipt = build_proxy_candidates(resolve(args.p5))
    atomic_write_parquet(candidates, resolve(args.candidates))
    atomic_write_csv(coefficients, resolve(args.coefficients))
    receipt["candidate_artifact_sha256"] = sha256_file(resolve(args.candidates))
    receipt["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(resolve(args.build_receipt), receipt)

    summary, metrics = evaluate_proxy_candidates(
        resolve(args.candidates),
        resolve(args.truth),
    )
    summary["candidate_artifact_sha256"] = receipt["candidate_artifact_sha256"]
    summary["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(resolve(args.evaluation), summary)
    atomic_write_csv(metrics, resolve(args.metrics))
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
