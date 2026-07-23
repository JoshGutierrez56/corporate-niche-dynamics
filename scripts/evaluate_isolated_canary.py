"""Construct locked proxies and evaluate the isolated P16E canary."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

from hypercube.canary import evaluate_realized_canary
from hypercube.data import atomic_write_json, sha256_file
from hypercube.exploratory import atomic_write_csv
from hypercube.proxy_redesign import atomic_write_parquet, build_proxy_candidates


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
        "--build-receipt",
        type=Path,
        default=Path("artifacts/manifests/p16e_proxy_build.json"),
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
    args = parser.parse_args(argv)
    resolve = lambda path: path if path.is_absolute() else PROJECT_ROOT / path
    canary = args.canary_root.resolve()

    candidate_paths = {}
    coefficient_rows = {}
    receipts = {}
    for scenario, output in (
        ("migration_alpha", resolve(args.migration_candidates)),
        ("null_alpha", resolve(args.null_candidates)),
    ):
        p5 = (
            canary
            / "data"
            / "processed"
            / "synthetic"
            / scenario
            / "p5"
            / "frontier_dynamics.parquet"
        )
        candidates, coefficients, receipt = build_proxy_candidates(p5)
        atomic_write_parquet(candidates, output)
        candidate_paths[scenario] = output
        coefficient_rows[scenario] = int(len(coefficients))
        receipts[scenario] = receipt
    build_receipt = {
        "schema_version": 1,
        "phase": "P16E",
        "status": "BUILT",
        "canary_root": str(canary),
        "synthetic_truth_read": False,
        "return_outcomes_read": False,
        "migration_candidate_sha256": sha256_file(
            candidate_paths["migration_alpha"]
        ),
        "null_candidate_sha256": sha256_file(candidate_paths["null_alpha"]),
        "coefficient_rows": coefficient_rows,
        "scenario_receipts": receipts,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write_json(resolve(args.build_receipt), build_receipt)

    summary, metrics = evaluate_realized_canary(
        canary
        / "data/processed/synthetic/migration_alpha/p7/"
        "forward_return_targets.parquet",
        canary / "data/raw/synthetic/migration_alpha/synthetic_truth.parquet",
        candidate_paths["migration_alpha"],
        canary
        / "data/processed/synthetic/null_alpha/p7/"
        "forward_return_targets.parquet",
        candidate_paths["null_alpha"],
    )
    summary["canary_root"] = str(canary)
    summary["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(resolve(args.summary), summary)
    atomic_write_csv(metrics, resolve(args.metrics))
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
