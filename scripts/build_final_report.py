"""Generate P10 figures, final report, README, and closeout receipt."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Sequence

import pandas as pd

from hypercube.config import load_config
from hypercube.data import atomic_write_json
from hypercube.visualization import (
    build_visualization_bundle,
    validate_visualization_bundle,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text.rstrip() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(name, path)
    except Exception:
        Path(name).unlink(missing_ok=True)
        raise


def _percent(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _report(statistics: pd.DataFrame) -> str:
    migration = statistics.loc[statistics["scenario"].eq("migration_alpha")].iloc[0]
    null = statistics.loc[statistics["scenario"].eq("null_alpha")].iloc[0]
    regime = statistics.loc[statistics["scenario"].eq("regime_shift")].iloc[0]
    return f"""# Research Report

## Final status

The synthetic P0-P10 research pipeline is complete. Its engineering and
point-in-time gates pass, but the predeclared scientific alpha-recovery test
fails. No real-data run occurred, and this report contains no claim about
actual markets or investable performance.

## Main result

The continuous state representation did not broadly beat simple profitability
and distress baselines, niche velocity/acceleration signs were not stable, and
the observable migration-surprise signal did not recover the injected return
effect. In the migration-alpha scenario, the six-month rank IC was
{migration['p7_primary_ic']:.4f}; the controlled coefficient was
{migration['p7_primary_coefficient']:.4f} with p =
{migration['p7_primary_p_value']:.3f} and Holm p =
{migration['p7_primary_holm_p_value']:.3f}. The hidden oracle component was
recoverable in P7, so the failure lies in the observable feature rather than
the return-path timing.

The null-alpha control behaved correctly: IC {null['p7_primary_ic']:.4f},
coefficient {null['p7_primary_coefficient']:.4f}, p =
{null['p7_primary_p_value']:.3f}. This prevents a false positive but does not
rescue the failed injected-alpha recovery.

## Implementability

Under the frozen conservative execution proxy, the primary migration-alpha
portfolio earned {_percent(migration['p8_annualized_gross_capacity_return'])}
annualized after capacity but before costs and
{_percent(migration['p8_annualized_net_return'])} after spread, slippage, and
borrow assumptions. Net Sharpe was {migration['p8_annualized_net_sharpe']:.2f},
maximum drawdown {_percent(migration['p8_net_maximum_drawdown'])}, average
monthly turnover {migration['p8_average_turnover']:.2f}, and modeled capacity
fill {migration['p8_average_capacity_fill_ratio']:.2f}. The null and
regime-shift scenarios were also negative after costs
({_percent(null['p8_annualized_net_return'])} and
{_percent(regime['p8_annualized_net_return'])}).

This supports only the modest H6 statement that gross results materially
overstate implementation. It does not support a trade.

## Archetypes

Training-only density clustering was not stable. Noise/unassigned rates were
{_percent(null['p9_noise_or_unassigned_rate'])} in null alpha,
{_percent(migration['p9_noise_or_unassigned_rate'])} in migration alpha, and
{_percent(regime['p9_noise_or_unassigned_rate'])} in regime shift. Mean
stability ARI was approximately zero where calculable; every bounded
regime-shift stability refit failed to recover two clusters. Neutral labels
are retained for audit, but they should not be interpreted as robust business
archetypes.

## Hypothesis scorecard

- H1: not broadly recovered; combined axes did not consistently beat simple
  profitability and distress baselines.
- H2: not broadly recovered; velocity and acceleration signs varied.
- H3: consistent with weak/ambiguous static return predictability, but the
  synthetic exercise is not market evidence.
- H4: not supported; migration surprise failed injected-alpha recovery.
- H5: not promoted because the primary return signal failed.
- H6: gross materially exceeded net, but net performance was negative.

## Evidence boundary

All accounting transformations use SEC filing time, RDQ, or a conservative
180-day fallback; all formation dates are strictly post-availability. CCM,
duplicates, staleness, delistings, overlapping horizons, costs, lagged
liquidity, hashes, and row counts are independently audited.

The 729-cell cube is used only as a secondary visualization. P9 clustering is
descriptive. All figures trace to saved tables listed in
`artifacts/tables/p10_figure_registry.csv`.

## Conclusion

The repository is a reproducible negative research result: it validates the
infrastructure, rejects the current observable migration-alpha construction,
and documents what would need redesign before any real-data confirmatory
study. Replacing synthetic parquets with licensed WRDS-shaped inputs is
structurally supported, but no real-data study should be run under the failed
alpha-recovery specification without a versioned exploratory redesign.
"""


def _readme() -> str:
    return """# Business Niche Hypercube

Corporate Niche Dynamics and Asset Prices

This repository implements a point-in-time, synthetic-first empirical-finance
pipeline for six-dimensional corporate viability, dynamics, survival, return
tests, implementation costs, and descriptive archetypes.

## Result

P0-P10 are implemented and audited on the three frozen synthetic scenarios.
The engineering gate passes; the primary scientific signal-recovery gate does
not. Migration surprise did not recover the injected return effect, net
portfolios were negative after conservative costs, and density archetypes were
mostly noise and unstable. These are synthetic falsification results, not
claims about markets.

## Reproduce

Python 3.11+ is required. The frozen environment is in `uv.lock`.

```powershell
uv sync --extra dev
make all CONFIG=configs/synthetic.yaml PYTHON=.venv/Scripts/python.exe
```

On Windows hosts without GNU Make, run the exact target directly:

```powershell
.\\.venv\\Scripts\\python.exe scripts\\run_full_pipeline.py --config configs\\synthetic.yaml
```

The command validates or builds every phase in order, refuses silent
overwrites, renders all figures, runs the complete test suite, and writes the
final receipts. See `docs/reproduction_guide.md`.

## Data contract and safeguards

Real-data mode uses the same local parquet contracts for fundamentals, SEC
filing dates, CRSP monthly returns, delistings, CCM links, and factors. Network
and WRDS access are disabled by configuration. Licensed raw data, generated
models, and large artifacts are ignored by Git.

SEC/RDQ/180-day availability, dated CCM links, strictly later formation,
delistings, missing paths, lagged execution inputs, training-only transforms,
and training-only clustering are independently tested.

## Key outputs

- `report.md`: final negative-result research report.
- `artifacts/manifests/`: phase receipts and hashes.
- `artifacts/tables/p10_report_statistics.csv`: report statistics.
- `artifacts/tables/p10_figure_registry.csv`: figure-to-table lineage.
- `figures/`: 11 required topics, including the secondary 729-cell view and
  rotating three-axis projection.

The project is research code, not investment advice, a live strategy, or a
claim of production readiness.
"""


def _reproduction_guide() -> str:
    return """# Reproduction guide

1. Install Python 3.11 or newer and `uv`.
2. Run `uv sync --extra dev`.
3. Run `make all CONFIG=configs/synthetic.yaml
   PYTHON=.venv/Scripts/python.exe`, or the documented Python command on
   Windows without GNU Make.
4. Confirm `artifacts/manifests/p10_validation.json` is `PASS`.
5. Confirm the complete pytest receipt and every figure hash in
   `artifacts/tables/p10_figure_registry.csv`.

The orchestrator is phase-ordered and resumable. Completed atomic phase
bundles are independently revalidated and reused; incomplete or hash-invalid
bundles stop the run. A genuinely fresh analytical run can begin with only
the deterministic raw synthetic scenario directories. Real-data mode requires
local licensed parquets matching `docs/data_dictionary.md`; it never enables
network or WRDS access.

Synthetic truth is read only by independent recovery audits. Feature, model,
portfolio, cost, clustering, and figure construction never use hidden truth.
"""


def main(argv: Sequence[str] | None = None) -> int:
    """Build final figures and reports, then independently validate images."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/synthetic.yaml"))
    parser.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if config.project.phase != "P10":
        raise SystemExit("Final reporting requires a P10 config.")
    metadata = build_visualization_bundle(PROJECT_ROOT, config)
    statistics = pd.read_csv(
        PROJECT_ROOT / "artifacts" / "tables" / "p10_report_statistics.csv"
    )
    _atomic_text(PROJECT_ROOT / "report.md", _report(statistics))
    _atomic_text(PROJECT_ROOT / "README.md", _readme())
    _atomic_text(
        PROJECT_ROOT / "docs" / "reproduction_guide.md",
        _reproduction_guide(),
    )
    validation = validate_visualization_bundle(PROJECT_ROOT)
    if validation["status"] != "PASS":
        raise RuntimeError(f"P10 visualization validation failed: {validation}")
    payload = {
        "schema_version": 1,
        "phase": "P10",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "metadata": metadata,
        "validation": validation,
        "scientific_status": "SIGNAL_RECOVERY_FAILED",
        "implementation_status": "NEGATIVE_AFTER_COSTS",
        "archetype_status": "UNSTABLE_MOSTLY_NOISE",
        "real_data_run": False,
    }
    if args.report:
        atomic_write_json(args.report, payload)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
