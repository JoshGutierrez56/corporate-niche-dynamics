"""Write a phase-aware reproducibility manifest with atomic replacement."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any

import pyarrow.parquet as pq

from hypercube.config import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit() -> str | None:
    try:
        value = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None
    return value or None


def _project_files() -> list[dict[str, Any]]:
    excluded = {".venv", ".git", "__pycache__", ".pytest_cache"}
    records: list[dict[str, Any]] = []
    for path in sorted(PROJECT_ROOT.rglob("*")):
        if (
            not path.is_file()
            or any(part in excluded for part in path.parts)
            or any(part.endswith(".egg-info") for part in path.parts)
        ):
            continue
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        if relative.startswith("data/") and path.name != ".gitkeep":
            continue
        if relative.startswith("artifacts/tables/") and path.name != ".gitkeep":
            continue
        if relative.startswith("artifacts/logs/") or relative.startswith(
            "artifacts/manifests/"
        ):
            continue
        records.append(
            {"path": relative, "bytes": path.stat().st_size, "sha256": _sha256(path)}
        )
    return records


def _input_files(raw_dir: Path) -> list[dict[str, Any]]:
    """Inventory local raw inputs without reading observation values."""

    root = raw_dir.parent if raw_dir.name in {"null_alpha", "migration_alpha", "regime_shift"} else raw_dir
    if not root.exists():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == ".gitkeep":
            continue
        record: dict[str, Any] = {
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        if path.suffix == ".parquet":
            record["rows"] = pq.ParquetFile(path).metadata.num_rows
        records.append(record)
    return records


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic.yaml"))
    parser.add_argument(
        "--output", type=Path
    )
    parser.add_argument("--brief", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    packages = {
        name: importlib.metadata.version(name)
        for name in (
            "numpy",
            "pandas",
            "pyarrow",
            "pydantic",
            "PyYAML",
            "scipy",
            "scikit-learn",
            "statsmodels",
            "joblib",
            "pytest",
        )
    }
    input_files = _input_files(PROJECT_ROOT / config.paths.raw_dir)
    phase_slug = config.project.phase.lower()
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": config.project.phase,
        "project": config.project.name,
        "run_name": config.project.run_name,
        "seed": config.project.seed,
        "resolved_config": config.model_dump(mode="json"),
        "config_path": args.config.as_posix(),
        "config_sha256": _sha256(args.config.resolve()),
        "git_commit": _git_commit(),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "platform": platform.platform(),
        },
        "packages": packages,
        "build_brief": (
            {
                "name": args.brief.name,
                "bytes": args.brief.stat().st_size,
                "sha256": _sha256(args.brief.resolve()),
            }
            if args.brief
            else None
        ),
        "input_data": input_files,
        "row_counts": {
            record["path"]: record["rows"]
            for record in input_files
            if "rows" in record
        },
        "feature_availability_summary": {
            "status": (
                "p10_full_synthetic_pipeline_figures_and_reports_constructed"
                if config.project.phase == "P10"
                else "p9_training_only_descriptive_archetypes_constructed"
                if config.project.phase == "P9"
                else "p8_cost_capacity_borrow_and_delay_sensitivities_constructed"
                if config.project.phase == "P8"
                else "p7_point_in_time_targets_and_gross_return_tests_constructed"
                if config.project.phase == "P7"
                else "p6_competing_exit_intervals_and_time_split_models_constructed"
                if config.project.phase == "P6"
                else "p5_frontier_and_point_in_time_dynamics_constructed"
                if config.project.phase == "P5"
                else "p4_fixed_horizon_labels_and_oos_viability_scores_constructed"
                if config.project.phase == "P4"
                else "p3_components_and_axes_constructed"
                if config.project.phase == "P3"
                else "p2_accounting_availability_constructed"
                if config.project.phase == "P2"
                else "not_constructed_before_p2"
            ),
            "hierarchy": list(config.availability.hierarchy),
            "strict_post_availability_formation": config.panel.strict_post_availability_formation,
            "maximum_staleness_months": config.panel.max_staleness_months,
        },
        "training_windows": (
            [
                {
                    "horizons_months": list(config.returns.horizons_months),
                    "policy": "P5 OOS features; independent formation-month cross-sections",
                }
            ]
            if config.project.phase in {"P7", "P8", "P10"}
            else [
                {
                    "training_end_year": config.clustering.training_end_year,
                    "maximum_training_rows": config.clustering.maximum_training_rows,
                    "policy": "training-only density fit; radius-bounded later assignment",
                }
            ]
            if config.project.phase == "P9"
            else [
                {
                    "horizon_years": horizon,
                    "validation_years": config.viability.validation_years,
                    "policy": "expanding training with horizon purge",
                }
                for horizon in config.viability.horizons_years
            ]
            if config.project.phase in {"P4", "P5", "P6"}
            else []
        ),
        "test_windows": (
            (
                [fold.model_dump(mode="json") for fold in config.survival.outer_folds]
                if config.project.phase == "P6"
                else [fold.model_dump(mode="json") for fold in config.viability.outer_folds]
            )
            if config.project.phase in {"P4", "P5", "P6", "P7", "P8", "P10"}
            else [
                {
                    "start_year": None,
                    "end_year": config.clustering.training_end_year,
                    "role": "clustering_fit",
                },
                {
                    "start_year": config.clustering.training_end_year + 1,
                    "end_year": None,
                    "role": "out_of_sample_assignment",
                },
            ]
            if config.project.phase == "P9"
            else []
        ),
        "model_hyperparameters": (
            {
                "viability": config.viability.model_dump(mode="json"),
                "dynamics": config.dynamics.model_dump(mode="json"),
                "survival": config.survival.model_dump(mode="json"),
                "returns": config.returns.model_dump(mode="json"),
                "costs": config.costs.model_dump(mode="json"),
                "clustering": config.clustering.model_dump(mode="json"),
            }
            if config.project.phase == "P10"
            else {
                "clustering": config.clustering.model_dump(mode="json"),
                "descriptive_only": True,
            }
            if config.project.phase == "P9"
            else {
                "viability": config.viability.model_dump(mode="json"),
                "dynamics": config.dynamics.model_dump(mode="json"),
                "survival": config.survival.model_dump(mode="json"),
                "returns": config.returns.model_dump(mode="json"),
                "costs": config.costs.model_dump(mode="json"),
            }
            if config.project.phase == "P8"
            else {
                "viability": config.viability.model_dump(mode="json"),
                "dynamics": config.dynamics.model_dump(mode="json"),
                "survival": config.survival.model_dump(mode="json"),
                "returns": config.returns.model_dump(mode="json"),
            }
            if config.project.phase == "P7"
            else {
                "viability": config.viability.model_dump(mode="json"),
                "dynamics": config.dynamics.model_dump(mode="json"),
                "survival": config.survival.model_dump(mode="json"),
            }
            if config.project.phase == "P6"
            else {
                "viability": config.viability.model_dump(mode="json"),
                "dynamics": config.dynamics.model_dump(mode="json"),
            }
            if config.project.phase == "P5"
            else config.viability.model_dump(mode="json")
            if config.project.phase == "P4"
            else {}
        ),
        "outputs": (
            [
                "artifacts/manifests/full_pipeline_receipt.json",
                "artifacts/manifests/p10_build.json",
                "artifacts/manifests/p10_validation.json",
                "artifacts/manifests/p10_manifest.json",
                "artifacts/tables/p10_*.csv",
                "figures/01_*.png through figures/11_*.png",
                "figures/11_three_axis_projection_rotating.gif",
                "README.md",
                "report.md",
                "docs/reproduction_guide.md",
            ]
            if config.project.phase == "P10"
            else [
                "artifacts/manifests/p9_build.json",
                "artifacts/manifests/p9_validation.json",
                "artifacts/manifests/p9_manifest.json",
                "artifacts/tables/p9_*.csv",
                "data/processed/synthetic/<scenario>/p9/archetype_assignments.parquet",
                "data/processed/synthetic/<scenario>/p9/archetype_profiles.csv",
                "data/processed/synthetic/<scenario>/p9/transition_matrix.csv",
                "data/processed/synthetic/<scenario>/p9/cluster_stability.csv",
            ]
            if config.project.phase == "P9"
            else [
                "artifacts/manifests/p8_build.json",
                "artifacts/manifests/p8_validation.json",
                "artifacts/manifests/p8_manifest.json",
                "artifacts/tables/p8_*.csv",
                "data/processed/synthetic/<scenario>/p8/capacity_assignments.parquet",
                "data/processed/synthetic/<scenario>/p8/executed_positions.parquet",
                "data/processed/synthetic/<scenario>/p8/cost_aware_monthly_returns.csv",
                "data/processed/synthetic/<scenario>/p8/cost_aware_summary.csv",
                "data/processed/synthetic/<scenario>/p8/cost_waterfall.csv",
            ]
            if config.project.phase == "P8"
            else [
                "artifacts/manifests/p7_build.json",
                "artifacts/manifests/p7_validation.json",
                "artifacts/manifests/p7_manifest.json",
                "artifacts/tables/p7_*.csv",
                "data/processed/synthetic/<scenario>/p7/forward_return_targets.parquet",
                "data/processed/synthetic/<scenario>/p7/fmb_summary.csv",
                "data/processed/synthetic/<scenario>/p7/portfolio_monthly_returns.csv",
                "data/processed/synthetic/<scenario>/p7/portfolio_factor_results.csv",
                "data/processed/synthetic/<scenario>/p7/portfolio_exposures.csv",
            ]
            if config.project.phase == "P7"
            else [
                "artifacts/manifests/p6_build.json",
                "artifacts/manifests/p6_validation.json",
                "artifacts/manifests/p6_manifest.json",
                "artifacts/tables/p6_*.csv",
                "data/processed/synthetic/<scenario>/p6/survival_intervals.parquet",
                "data/processed/synthetic/<scenario>/p6/fold_predictions.parquet",
                "data/processed/synthetic/<scenario>/p6/cause_coefficients.csv",
                "data/processed/synthetic/<scenario>/p6/ph_diagnostics.csv",
                "data/processed/synthetic/<scenario>/p6/models/*.joblib",
            ]
            if config.project.phase == "P6"
            else [
                "artifacts/manifests/p5_build.json",
                "artifacts/manifests/p5_validation.json",
                "artifacts/manifests/p5_manifest.json",
                "artifacts/tables/p5_*.csv",
                "data/processed/synthetic/<scenario>/p5/frontier_dynamics.parquet",
                "data/processed/synthetic/<scenario>/p5/migration_model_diagnostics.csv",
                "data/processed/synthetic/<scenario>/p5/models/*.joblib",
            ]
            if config.project.phase == "P5"
            else
            [
                "artifacts/manifests/p4_build.json",
                "artifacts/manifests/p4_validation.json",
                "artifacts/manifests/p4_manifest.json",
                "artifacts/tables/p4_*.csv",
                "data/processed/synthetic/<scenario>/p4/viability_labels.parquet",
                "data/processed/synthetic/<scenario>/p4/oos_predictions.parquet",
                "data/processed/synthetic/<scenario>/p4/fold_metrics.csv",
                "data/processed/synthetic/<scenario>/p4/models/*.joblib",
            ]
            if config.project.phase == "P4"
            else
            [
                "artifacts/manifests/p3_build.json",
                "artifacts/manifests/p3_validation.json",
                "artifacts/manifests/p3_manifest.json",
                "artifacts/tables/p3_*.csv",
                "data/processed/synthetic/<scenario>/p3/component_features.parquet",
                "data/processed/synthetic/<scenario>/p3/axis_scores.parquet",
                "data/processed/synthetic/<scenario>/p3/p3_diagnostics.json",
            ]
            if config.project.phase == "P3"
            else
            [
                "artifacts/manifests/p2_build.json",
                "artifacts/manifests/p2_validation.json",
                "artifacts/manifests/p2_manifest.json",
                "data/processed/synthetic/<scenario>/accounting_availability.parquet",
                "data/processed/synthetic/<scenario>/universe_monthly.parquet",
                "data/processed/synthetic/<scenario>/firm_month_panel.parquet",
                "data/processed/synthetic/<scenario>/row_count_waterfall.csv",
            ]
            if config.project.phase == "P2"
            else [
                f"artifacts/manifests/{phase_slug}_manifest.json",
                f"artifacts/manifests/{phase_slug}_validation.json",
            ]
        ),
        "warnings": [
            f"{config.project.phase} inputs and outputs are synthetic and are not empirical evidence.",
            (
                "P10 closes a synthetic pipeline whose primary migration-alpha recovery failed; no real-data or investable result exists."
                if config.project.phase == "P10"
                else "P9 archetypes are unstable descriptive synthetic clusters; they are not evidence of economic types or market effects."
                if config.project.phase == "P9"
                else "P8 reports synthetic execution proxies, not executable performance; no clustering or real-data test was run."
                if config.project.phase == "P8"
                else "P7 reports gross synthetic return tests only; no costs, borrow, capacity, net-performance, clustering, or real-data test was run."
                if config.project.phase == "P7"
                else "P6 estimates cause-specific predictive associations only; no causal, return, portfolio, or clustering test was run."
                if config.project.phase == "P6"
                else "P5 fits migration expectations only; no competing-risk, return, portfolio, or clustering test was run."
                if config.project.phase == "P5"
                else "P4 fits viability models only; no return test or portfolio was run."
                if config.project.phase == "P4"
                else "No labels, fitted models, return tests, or portfolios were created in P3."
            ),
            "GNU Make was not available on the inspected Windows host.",
        ],
        "failed_assumptions": [],
        "downstream_empirical_phases_started": config.project.phase in {"P7", "P8", "P9", "P10"},
        "project_files": _project_files(),
    }
    requested_output = args.output or Path(
        f"artifacts/manifests/{phase_slug}_manifest.json"
    )
    output = (
        requested_output
        if requested_output.is_absolute()
        else PROJECT_ROOT / requested_output
    )
    _atomic_json(output, payload)
    print(
        json.dumps(
            {
                "status": f"{config.project.phase}_MANIFEST_WRITTEN",
                "path": str(output),
                "project_files": len(payload["project_files"]),
                "input_files": len(input_files),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
