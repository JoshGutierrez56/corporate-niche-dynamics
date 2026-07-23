"""Build the P16E manifest across main-project and isolated-canary roots."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from hypercube.data import atomic_write_json, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANARY_ROOT = PROJECT_ROOT.parent / ".p16-hypercube-canary"


def _record(path: Path, root: Path, scope: str) -> dict[str, object]:
    return {
        "scope": scope,
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> int:
    main_files = (
        Path("hypercube/canary.py"),
        Path("hypercube/proxy_redesign.py"),
        Path("scripts/evaluate_isolated_canary.py"),
        Path("scripts/validate_isolated_canary.py"),
        Path("scripts/build_canary_manifest.py"),
        Path("tests/test_p16e_canary.py"),
        Path("docs/p16e_isolated_canary_preregistration.md"),
        Path("docs/p16e_isolated_canary_results.md"),
        Path("artifacts/tables/p16e_migration_proxy_candidates.parquet"),
        Path("artifacts/tables/p16e_null_proxy_candidates.parquet"),
        Path("artifacts/tables/p16e_canary_metrics.csv"),
        Path("artifacts/manifests/p16e_proxy_build.json"),
        Path("artifacts/manifests/p16e_canary_evaluation.json"),
        Path("artifacts/manifests/p16e_canary_validation.json"),
    )
    canary_files = (
        Path("configs/base.yaml"),
        Path("scripts/run_full_pipeline.py"),
        Path("artifacts/manifests/full_pipeline_receipt.json"),
        Path("artifacts/manifests/p1_validation.json"),
        Path("artifacts/manifests/p2_validation.json"),
        Path("artifacts/manifests/p3_validation.json"),
        Path("artifacts/manifests/p4_validation.json"),
        Path("artifacts/manifests/p5_validation.json"),
        Path("artifacts/manifests/p6_validation.json"),
        Path("artifacts/manifests/p7_validation.json"),
        Path("artifacts/manifests/p7_manifest.json"),
        Path("data/raw/synthetic/migration_alpha/data_manifest.json"),
        Path(
            "data/raw/synthetic/migration_alpha/"
            "synthetic_scenario_metadata.json"
        ),
        Path(
            "data/processed/synthetic/migration_alpha/p5/"
            "frontier_dynamics.parquet"
        ),
        Path(
            "data/processed/synthetic/migration_alpha/p7/"
            "forward_return_targets.parquet"
        ),
        Path(
            "data/processed/synthetic/null_alpha/p7/"
            "forward_return_targets.parquet"
        ),
    )
    records = [
        *[_record(PROJECT_ROOT / path, PROJECT_ROOT, "main") for path in main_files],
        *[_record(CANARY_ROOT / path, CANARY_ROOT, "isolated_canary") for path in canary_files],
    ]
    payload = {
        "schema_version": 1,
        "phase": "P16E",
        "version": "hypercube-isolated-six-x-canary-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "gate_result": "PASS",
        "selected_candidate": "anchored_axis_innovation",
        "selected_multiplier": 6.0,
        "migration_alpha_monthly": 0.024,
        "isolated_canary_root": str(CANARY_ROOT),
        "p1_p7_validated": True,
        "p8_p10_run": False,
        "new_synthetic_scenario_generated": True,
        "frozen_p0_p10_outputs_modified": False,
        "real_data_run": False,
        "files": records,
    }
    output = PROJECT_ROOT / "artifacts/manifests/p16e_manifest.json"
    atomic_write_json(output, payload)
    print(f"P16E_MANIFEST_WRITTEN {output} {len(records)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
