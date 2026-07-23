"""Independently validate the saved P11E alpha-power diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from hypercube.data import atomic_write_json
from hypercube.exploratory import validate_alpha_power_outputs


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
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
        default=Path("artifacts/manifests/p11e_alpha_power_diagnostic.json"),
    )
    parser.add_argument(
        "--folds",
        type=Path,
        default=Path("artifacts/tables/p11e_alpha_power_by_fold.csv"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/manifests/p11e_alpha_power_validation.json"),
    )
    args = parser.parse_args(argv)
    resolve = lambda path: path if path.is_absolute() else PROJECT_ROOT / path
    report = validate_alpha_power_outputs(
        resolve(args.targets),
        resolve(args.truth),
        resolve(args.summary),
        resolve(args.folds),
    )
    atomic_write_json(resolve(args.report), report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
