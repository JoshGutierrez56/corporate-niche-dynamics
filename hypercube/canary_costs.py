"""Isolated P17E cost-aware evaluation for the locked P16E proxy."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from hypercube.config import HypercubeConfig
from hypercube.data import atomic_write_json, sha256_file
from hypercube.portfolio import (
    _lagged_execution_liquidity,
    build_executed_positions,
    capacity_diagnostics,
    cost_waterfall,
    delayed_execution_sensitivity,
    evaluate_cost_aware_months,
    prepare_cost_assignments,
    summarize_cost_aware_months,
)
from hypercube.returns import (
    build_monthly_portfolios,
    construct_portfolio_sorts,
)


P17E_VERSION = "hypercube-isolated-locked-proxy-cost-canary-v1"
LOCKED_SIGNAL = "anchored_axis_innovation"
LOCKED_SIGNAL_LABEL = "anchored_axis_innovation"
LOCKED_HORIZON = 6
SCENARIOS = ("migration_alpha", "null_alpha")
JOIN_KEY = ["gvkey", "datadate", "fyear", "feature_date"]
STRATEGY_COLUMNS = [
    "horizon_months",
    "signal",
    "signal_label",
    "weighting",
    "industry_neutral",
]


class CanaryCostError(ValueError):
    """Raised when the isolated locked-proxy cost contract is violated."""


def locked_proxy_config(config: HypercubeConfig) -> HypercubeConfig:
    """Return a runtime-only P8 config for the preregistered locked proxy."""

    returns = config.returns.model_copy(
        update={
            "horizons_months": (LOCKED_HORIZON,),
            "primary_horizon_months": LOCKED_HORIZON,
            "signals": (LOCKED_SIGNAL,),
            "primary_signal": LOCKED_SIGNAL,
        }
    )
    project = config.project.model_copy(
        update={"phase": "P8", "run_name": "p17e_isolated_locked_proxy_cost_canary"}
    )
    return config.model_copy(update={"returns": returns, "project": project})


def _atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.to_parquet(name, index=False, compression="zstd")
        os.replace(name, path)
    except Exception:
        Path(name).unlink(missing_ok=True)
        raise


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.to_csv(name, index=False, lineterminator="\n")
        os.replace(name, path)
    except Exception:
        Path(name).unlink(missing_ok=True)
        raise


def _row_count(path: Path) -> int | None:
    if path.suffix == ".parquet":
        return int(pq.ParquetFile(path).metadata.num_rows)
    if path.suffix == ".csv":
        return max(0, sum(1 for _ in path.open("r", encoding="utf-8")) - 1)
    return None


def _input_record(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    rows = _row_count(path)
    if rows is not None:
        record["rows"] = rows
    return record


def _output_record(path: Path, root: Path) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }
    rows = _row_count(path)
    if rows is not None:
        record["rows"] = rows
    return record


def _locked_targets(p7_dir: Path, candidate_path: Path) -> pd.DataFrame:
    candidates = pd.read_parquet(candidate_path)
    required = {*JOIN_KEY, LOCKED_SIGNAL}
    missing = sorted(required - set(candidates.columns))
    if missing:
        raise CanaryCostError(f"Locked candidate is missing columns: {missing}")
    candidates = candidates[[*JOIN_KEY, LOCKED_SIGNAL]].copy()
    if candidates.duplicated(JOIN_KEY).any():
        raise CanaryCostError("Locked candidate keys are duplicated.")
    targets = pd.read_parquet(p7_dir / "forward_return_targets.parquet")
    targets = targets.merge(
        candidates,
        on=JOIN_KEY,
        how="left",
        validate="many_to_one",
    )
    if targets[LOCKED_SIGNAL].notna().sum() == 0:
        raise CanaryCostError("Locked candidate has no P7-eligible observations.")
    return targets


def build_locked_proxy_cost_scenario(
    *,
    scenario: str,
    p7_dir: Path,
    raw_dir: Path,
    candidate_path: Path,
    output_dir: Path,
    config: HypercubeConfig,
    preregistration_path: Path,
) -> dict[str, Any]:
    """Build one isolated P17E scenario without reading synthetic truth."""

    if scenario not in SCENARIOS:
        raise CanaryCostError(f"Unsupported P17E scenario: {scenario}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite P17E output: {output_dir}")
    p17_config = locked_proxy_config(config)
    targets = _locked_targets(p7_dir, candidate_path)
    quantiles, assignments, attrition = construct_portfolio_sorts(
        targets, p17_config
    )
    if set(assignments["signal"]) != {LOCKED_SIGNAL}:
        raise CanaryCostError("P17E assignments contain an unlocked signal.")
    if set(assignments["horizon_months"]) != {LOCKED_HORIZON}:
        raise CanaryCostError("P17E assignments contain an unlocked horizon.")
    paths = pd.read_parquet(p7_dir / "event_month_paths.parquet")
    factors = pd.read_parquet(raw_dir / "factor_returns.parquet")
    gross_monthly, gross_summary, gross_exposures = build_monthly_portfolios(
        assignments,
        paths,
        factors,
        p17_config,
    )

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".p17e-staging-", dir=output_dir.parent))
    try:
        bridge = staging / "locked_proxy_p7_bridge"
        bridge.mkdir(parents=True)
        _atomic_parquet(
            pd.read_parquet(p7_dir / "return_events.parquet"),
            bridge / "return_events.parquet",
        )
        _atomic_parquet(paths, bridge / "event_month_paths.parquet")
        _atomic_parquet(targets, bridge / "forward_return_targets.parquet")
        _atomic_csv(quantiles, bridge / "portfolio_quantile_returns.csv")
        _atomic_parquet(assignments, bridge / "portfolio_assignments.parquet")
        _atomic_csv(gross_monthly, bridge / "portfolio_monthly_returns.csv")
        _atomic_csv(gross_summary, bridge / "portfolio_factor_results.csv")
        _atomic_csv(gross_exposures, bridge / "portfolio_exposures.csv")
        _atomic_csv(attrition, bridge / "portfolio_sort_attrition.csv")

        cost_assignments = prepare_cost_assignments(bridge, raw_dir, p17_config)
        lagged = _lagged_execution_liquidity(raw_dir, p17_config)
        position_frames: list[pd.DataFrame] = []
        calendar_frames: list[pd.DataFrame] = []
        for delay in (0, p17_config.costs.delayed_execution_months):
            positions, calendars = build_executed_positions(
                cost_assignments,
                paths,
                lagged,
                p17_config,
                delay_months=delay,
            )
            position_frames.append(positions)
            calendar_frames.append(calendars)
        positions = pd.concat(position_frames, ignore_index=True)
        calendars = pd.concat(calendar_frames, ignore_index=True)
        monthly = evaluate_cost_aware_months(
            positions, calendars, factors, p17_config
        )
        summary = summarize_cost_aware_months(monthly, p17_config)
        capacity = capacity_diagnostics(cost_assignments)
        waterfall = cost_waterfall(summary, gross_monthly, p17_config)
        delayed = delayed_execution_sensitivity(summary, p17_config)

        primary_mask = (
            summary["signal"].eq(LOCKED_SIGNAL)
            & summary["weighting"].eq(p17_config.costs.primary_weighting)
            & summary["industry_neutral"].eq(
                p17_config.costs.primary_industry_neutral
            )
            & summary["cost_scenario"].eq(
                p17_config.costs.primary_cost_scenario
            )
            & summary["delay_months"].eq(0)
        )
        if int(primary_mask.sum()) != 1:
            raise CanaryCostError("P17E primary portfolio is not unique.")
        primary = summary.loc[primary_mask].iloc[0]

        _atomic_parquet(
            cost_assignments, staging / "capacity_assignments.parquet"
        )
        _atomic_parquet(positions, staging / "executed_positions.parquet")
        _atomic_csv(monthly, staging / "cost_aware_monthly_returns.csv")
        _atomic_csv(summary, staging / "cost_aware_summary.csv")
        _atomic_csv(capacity, staging / "capacity_diagnostics.csv")
        _atomic_csv(waterfall, staging / "cost_waterfall.csv")
        _atomic_csv(delayed, staging / "delayed_execution_sensitivity.csv")
        atomic_write_json(
            staging / "cost_metadata.json",
            {
                "schema_version": 1,
                "phase": "P17E",
                "version": P17E_VERSION,
                "scenario": scenario,
                "isolated_canary": True,
                "selected_candidate": LOCKED_SIGNAL,
                "primary_horizon_months": LOCKED_HORIZON,
                "primary_weighting": p17_config.costs.primary_weighting,
                "primary_industry_neutral": (
                    p17_config.costs.primary_industry_neutral
                ),
                "primary_cost_scenario": (
                    p17_config.costs.primary_cost_scenario
                ),
                "synthetic_truth_read": False,
                "costs_applied": True,
                "borrow_model_applied": True,
                "capacity_model_applied": True,
                "performance_gate_applied": False,
                "primary_result": {
                    "annualized_gross_capacity_return": float(
                        primary["annualized_gross_capacity_return"]
                    ),
                    "annualized_net_return": float(
                        primary["annualized_net_return"]
                    ),
                    "annualized_net_sharpe": float(
                        primary["annualized_net_sharpe"]
                    ),
                    "net_factor_alpha_monthly": float(
                        primary["net_factor_alpha_monthly"]
                    ),
                    "net_factor_alpha_p_value": float(
                        primary["net_factor_alpha_p_value"]
                    ),
                    "net_maximum_drawdown": float(
                        primary["net_maximum_drawdown"]
                    ),
                    "average_turnover": float(primary["average_turnover"]),
                    "average_capacity_fill_ratio": float(
                        primary["average_capacity_fill_ratio"]
                    ),
                },
            },
        )
        atomic_write_json(
            staging / "resolved_config.json",
            p17_config.model_dump(mode="json"),
        )
        input_paths = [
            preregistration_path,
            candidate_path,
            p7_dir / "p7_manifest.json",
            p7_dir / "return_events.parquet",
            p7_dir / "event_month_paths.parquet",
            p7_dir / "forward_return_targets.parquet",
            raw_dir / "crsp_monthly.parquet",
            raw_dir / "factor_returns.parquet",
        ]
        output_paths = sorted(
            (
                path
                for path in staging.rglob("*")
                if path.is_file() and path.name != "p17e_manifest.json"
            ),
            key=lambda path: path.as_posix(),
        )
        manifest = {
            "schema_version": 1,
            "phase": "P17E",
            "version": P17E_VERSION,
            "scenario": scenario,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "isolated_canary": True,
            "selected_candidate": LOCKED_SIGNAL,
            "primary_horizon_months": LOCKED_HORIZON,
            "inputs": [_input_record(path) for path in input_paths],
            "outputs": [
                _output_record(path, staging) for path in output_paths
            ],
            "rows": {
                "locked_targets": int(len(targets)),
                "portfolio_assignments": int(len(assignments)),
                "capacity_assignments": int(len(cost_assignments)),
                "executed_positions": int(len(positions)),
                "monthly_cost_rows": int(len(monthly)),
                "summary_rows": int(len(summary)),
            },
            "synthetic_truth_read": False,
            "performance_gate_applied": False,
            "p9_p10_run": False,
            "real_data_run": False,
        }
        atomic_write_json(staging / "p17e_manifest.json", manifest)
        os.replace(staging, output_dir)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "scenario": scenario,
        "status": "PASS",
        "output_dir": str(output_dir),
        "selected_candidate": LOCKED_SIGNAL,
        "primary_result": json.loads(
            (output_dir / "cost_metadata.json").read_text(encoding="utf-8")
        )["primary_result"],
        "summary_rows": int(len(summary)),
        "synthetic_truth_read": False,
        "performance_gate_applied": False,
    }


def validate_locked_proxy_cost_scenario(
    output_dir: Path,
    config: HypercubeConfig,
) -> dict[str, Any]:
    """Independently validate one saved P17E scenario."""

    required = (
        "capacity_assignments.parquet",
        "executed_positions.parquet",
        "cost_aware_monthly_returns.csv",
        "cost_aware_summary.csv",
        "capacity_diagnostics.csv",
        "cost_waterfall.csv",
        "delayed_execution_sensitivity.csv",
        "cost_metadata.json",
        "resolved_config.json",
        "p17e_manifest.json",
    )
    errors = [
        f"Missing P17E file: {name}"
        for name in required
        if not (output_dir / name).is_file()
    ]
    if errors:
        return {"status": "FAIL", "errors": errors}
    manifest = json.loads(
        (output_dir / "p17e_manifest.json").read_text(encoding="utf-8")
    )
    metadata = json.loads(
        (output_dir / "cost_metadata.json").read_text(encoding="utf-8")
    )
    for record in manifest.get("inputs", []):
        path = Path(record["path"])
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            errors.append(f"P17E input mismatch: {path.name}")
        if "truth" in path.name.lower() or "truth" in path.parent.name.lower():
            errors.append(f"P17E manifest includes truth input: {path}")
    for record in manifest.get("outputs", []):
        path = output_dir / record["path"]
        if (
            not path.is_file()
            or path.stat().st_size != record["bytes"]
            or sha256_file(path) != record["sha256"]
        ):
            errors.append(f"P17E output mismatch: {record['path']}")
            continue
        rows = _row_count(path)
        if rows is not None and rows != record.get("rows"):
            errors.append(f"P17E row-count mismatch: {record['path']}")
    if metadata.get("synthetic_truth_read") is not False:
        errors.append("P17E construction read synthetic truth.")
    if metadata.get("selected_candidate") != LOCKED_SIGNAL:
        errors.append("P17E selected candidate is not locked.")
    if metadata.get("performance_gate_applied") is not False:
        errors.append("P17E improperly applies a performance gate.")
    for flag in ("costs_applied", "borrow_model_applied", "capacity_model_applied"):
        if metadata.get(flag) is not True:
            errors.append(f"P17E metadata flag is false: {flag}")

    assignments = pd.read_parquet(
        output_dir / "capacity_assignments.parquet"
    )
    positions = pd.read_parquet(output_dir / "executed_positions.parquet")
    monthly = pd.read_csv(
        output_dir / "cost_aware_monthly_returns.csv",
        parse_dates=["holding_date"],
    )
    summary = pd.read_csv(
        output_dir / "cost_aware_summary.csv",
        parse_dates=["first_month", "last_month"],
    )
    waterfall = pd.read_csv(output_dir / "cost_waterfall.csv")
    delayed = pd.read_csv(output_dir / "delayed_execution_sensitivity.csv")
    if set(assignments["signal"]) != {LOCKED_SIGNAL}:
        errors.append("P17E assignments contain an unlocked signal.")
    if set(assignments["horizon_months"]) != {LOCKED_HORIZON}:
        errors.append("P17E assignments contain an unlocked horizon.")
    if (
        assignments["capacity_weight"] - assignments["weight"] > 1e-12
    ).any():
        errors.append("P17E capacity weight exceeds frozen gross weight.")
    blocked = assignments["leg"].eq("short") & ~assignments[
        "short_available"
    ].astype(bool)
    if assignments.loc[blocked, "capacity_weight"].ne(0.0).any():
        errors.append("P17E retains an unavailable short.")
    if (
        positions["actual_signed_weight"].abs()
        - positions["capacity_weight_limit"]
        > 1e-12
    ).any():
        errors.append("P17E position breaches its capacity limit.")
    if (
        pd.to_datetime(positions["liquidity_date"])
        >= pd.to_datetime(positions["holding_date"])
    ).any():
        errors.append("P17E uses non-lagged execution liquidity.")
    recomputed_net = (
        monthly["gross_capacity_return"]
        - monthly["transaction_cost"]
        - monthly["borrow_cost"]
    )
    if not np.allclose(
        monthly["net_return"], recomputed_net, rtol=1e-12, atol=1e-12
    ):
        errors.append("P17E monthly net-return equation does not reconcile.")
    expected_summary = (
        len(config.returns.weighting_schemes)
        * len(config.returns.industry_neutral_modes)
        * len(config.costs.scenarios)
        * 2
    )
    if len(summary) != expected_summary:
        errors.append("P17E strategy/cost/delay ladder is incomplete.")
    expected_delayed = (
        len(config.returns.weighting_schemes)
        * len(config.returns.industry_neutral_modes)
    )
    if len(delayed) != expected_delayed:
        errors.append("P17E delayed-execution ladder is incomplete.")
    if not np.allclose(
        waterfall["reconciliation_error"], 0.0, rtol=0.0, atol=1e-12
    ):
        errors.append("P17E cost waterfall does not reconcile.")
    for _, saved in summary.iterrows():
        mask = pd.Series(True, index=monthly.index)
        for column in (
            *STRATEGY_COLUMNS,
            "delay_months",
            "cost_scenario",
        ):
            left = monthly[column]
            value = saved[column]
            if column == "industry_neutral":
                left = left.astype(str).str.lower().eq("true")
                value = str(value).lower() == "true"
            mask &= left.eq(value)
        values = monthly.loc[mask, "net_return"].to_numpy(float)
        if not len(values):
            errors.append("P17E saved summary has no monthly rows.")
            break
        annualized = 12.0 * float(np.mean(values))
        if not math.isclose(
            float(saved["annualized_net_return"]),
            annualized,
            rel_tol=1e-10,
            abs_tol=1e-12,
        ):
            errors.append("P17E annualized net return does not recompute.")
            break
    primary = summary.loc[
        summary["signal"].eq(LOCKED_SIGNAL)
        & summary["weighting"].eq(config.costs.primary_weighting)
        & summary["industry_neutral"].astype(str).str.lower().eq(
            str(config.costs.primary_industry_neutral).lower()
        )
        & summary["cost_scenario"].eq(config.costs.primary_cost_scenario)
        & summary["delay_months"].eq(0)
    ]
    if len(primary) != 1:
        errors.append("P17E primary portfolio is not unique.")
    return {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "scenario": metadata.get("scenario"),
        "selected_candidate": metadata.get("selected_candidate"),
        "summary_rows": int(len(summary)),
        "synthetic_truth_read": False,
        "performance_gate_applied": False,
        "primary_result": metadata.get("primary_result"),
    }
