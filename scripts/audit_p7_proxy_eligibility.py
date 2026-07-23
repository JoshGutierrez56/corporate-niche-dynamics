"""Run the corrected P14F P7 eligibility-key proxy audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Sequence

from hypercube.data import atomic_write_json
from hypercube.exploratory import atomic_write_csv
from hypercube.proxy_redesign import audit_p7_eligibility


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
    args = parser.parse_args(argv)
    resolve = lambda path: path if path.is_absolute() else PROJECT_ROOT / path
    summary, table = audit_p7_eligibility(
        resolve(args.candidates),
        resolve(args.targets),
        resolve(args.truth),
    )
    summary["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(resolve(args.summary), summary)
    atomic_write_csv(table, resolve(args.table))
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
