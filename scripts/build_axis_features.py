"""Build and validate phase-P3 component and six-axis feature bundles."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Sequence, cast

import pandas as pd

from hypercube.axes import build_axis_bundle, scenario_p3_dir, validate_p3_directory
from hypercube.config import load_config
from hypercube.data import SCENARIOS, Scenario, atomic_write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        frame.to_csv(name, index=False, lineterminator="\n")
        os.replace(name, path)
    except Exception:
        Path(name).unlink(missing_ok=True)
        raise


def _aggregate_tables(outputs: list[tuple[str, Path]]) -> list[str]:
    mappings = {
        "component_missingness.csv": "p3_component_missingness.csv",
        "component_correlations.csv": "p3_component_correlations.csv",
        "axis_correlations.csv": "p3_axis_correlations.csv",
        "component_vif.csv": "p3_component_vif.csv",
        "ablation_catalog.csv": "p3_ablation_catalog.csv",
    }
    written: list[str] = []
    for source_name, target_name in mappings.items():
        pieces = []
        for scenario, output in outputs:
            piece = pd.read_csv(output / source_name)
            piece.insert(0, "scenario", scenario)
            pieces.append(piece)
        target = PROJECT_ROOT / "artifacts" / "tables" / target_name
        _atomic_csv(pd.concat(pieces, ignore_index=True), target)
        written.append(str(target))
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """Build one or all synthetic P3 bundles and aggregate diagnostics."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic.yaml"))
    parser.add_argument("--scenario", choices=SCENARIOS)
    parser.add_argument("--all-scenarios", action="store_true")
    parser.add_argument("--p2-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if config.project.phase != "P3":
        raise SystemExit("Axis construction requires a P3 config.")
    if args.all_scenarios and (args.p2_dir is not None or args.output_dir is not None):
        raise SystemExit("Explicit directories cannot be combined with --all-scenarios.")
    if config.data.mode == "synthetic":
        selected: tuple[str, ...] = (
            tuple(SCENARIOS)
            if args.all_scenarios
            else (cast(Scenario, args.scenario or config.data.scenario),)
        )
    else:
        if args.scenario:
            raise SystemExit("Real mode does not accept synthetic scenarios.")
        selected = ("real",)

    reports: list[dict[str, object]] = []
    outputs: list[tuple[str, Path]] = []
    for scenario in selected:
        if scenario == "real":
            p2_dir = args.p2_dir or Path(config.paths.processed_dir)
            output_dir = args.output_dir or p2_dir / "p3"
        else:
            output_dir = args.output_dir or scenario_p3_dir(
                config, cast(Scenario, scenario)
            )
            p2_dir = args.p2_dir or output_dir.parent
        diagnostics = build_axis_bundle(
            p2_dir, output_dir, config, scenario=scenario
        )
        validation = validate_p3_directory(output_dir, config)
        reports.append(
            {
                "scenario": scenario,
                "p2_dir": str(p2_dir),
                "output_dir": str(output_dir),
                "construction": diagnostics,
                "validation": validation,
                "models_fitted": [],
            }
        )
        outputs.append((scenario, output_dir))
        print(
            json.dumps(
                {
                    "status": validation["status"],
                    "scenario": scenario,
                    "output_dir": str(output_dir),
                    "rows": validation.get("rows", {}),
                    "errors": validation.get("errors", []),
                    "models_fitted": [],
                },
                sort_keys=True,
            )
        )
    tables = _aggregate_tables(outputs)
    payload = {
        "schema_version": 1,
        "phase": "P3",
        "config": str(args.config),
        "status": "PASS"
        if all(item["validation"]["status"] == "PASS" for item in reports)
        else "FAIL",
        "reports": reports,
        "aggregate_tables": tables,
        "models_fitted": [],
    }
    if args.report:
        atomic_write_json(args.report, payload)
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
