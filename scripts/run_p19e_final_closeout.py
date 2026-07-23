"""Build the P19E final repository closeout without rerunning analytics."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from hypercube.data import atomic_write_json, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_NAME = "corporate-niche-dynamics"


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text.rstrip() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(PROJECT_ROOT)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _phase_manifest_hashes() -> dict[str, str]:
    root = PROJECT_ROOT / "artifacts" / "manifests"
    return {
        path.name: sha256_file(path)
        for path in sorted(root.glob("*.json"))
        if not path.name.startswith("p19e_")
    }


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    process_query_limited_information = 0x1000
    handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
        process_query_limited_information, False, pid
    )
    if not handle:
        return False
    ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
    return True


def _archive_p10_documents() -> list[dict[str, Any]]:
    p10_path = PROJECT_ROOT / "artifacts" / "manifests" / "p10_manifest.json"
    p10 = json.loads(p10_path.read_text(encoding="utf-8"))
    expected = {
        record["path"]: record
        for record in p10["project_files"]
        if record["path"] in {"README.md", "report.md"}
    }
    if set(expected) != {"README.md", "report.md"}:
        raise RuntimeError("P10 manifest does not freeze README.md and report.md.")
    archives = []
    for source_name, archive_name in (
        ("README.md", "p10_README.md"),
        ("report.md", "p10_report.md"),
    ):
        source = PROJECT_ROOT / source_name
        record = expected[source_name]
        payload = source.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        if digest != record["sha256"] or len(payload) != record["bytes"]:
            raise RuntimeError(f"Current {source_name} no longer matches P10.")
        archive = PROJECT_ROOT / "docs" / "archive" / archive_name
        if archive.exists():
            if archive.read_bytes() != payload:
                raise RuntimeError(f"Conflicting P10 archive exists: {archive}")
        else:
            _atomic_bytes(archive, payload)
        archives.append(
            {
                **_record(archive),
                "p10_original_path": source_name,
                "p10_expected_sha256": record["sha256"],
                "byte_identical": True,
            }
        )
    return archives


def _claim_ledger(
    p10: pd.DataFrame,
    p16: pd.DataFrame,
    p17: pd.DataFrame,
    p18: pd.DataFrame,
) -> pd.DataFrame:
    original = p10.loc[p10["scenario"].eq("migration_alpha")].iloc[0]
    migration16 = p16.loc[p16["scenario"].eq("migration_alpha")].iloc[0]
    null16 = p16.loc[p16["scenario"].eq("null_alpha")].iloc[0]
    migration17 = p17.loc[p17["scenario"].eq("migration_alpha")].iloc[0]
    return pd.DataFrame(
        [
            {
                "claim_id": "engineering",
                "domain": "reproducibility",
                "verdict": "PASS",
                "scope": "synthetic pipeline",
                "primary_metric": "P13F-P18E signed gates",
                "value": 6.0,
                "evidence": "artifacts/manifests/p13f_manifest.json through p18e_manifest.json",
            },
            {
                "claim_id": "original_signal",
                "domain": "measurement",
                "verdict": "REJECT_ORIGINAL_SPECIFICATION",
                "scope": "original 1x synthetic migration scenario",
                "primary_metric": "six-month rank IC",
                "value": float(original["p7_primary_ic"]),
                "evidence": "artifacts/tables/p10_report_statistics.csv",
            },
            {
                "claim_id": "redesigned_signal",
                "domain": "measurement",
                "verdict": "PASS_SYNTHETIC_6X_ONLY",
                "scope": "isolated amplified synthetic canary",
                "primary_metric": "six-month rank IC",
                "value": float(migration16["candidate_return_spearman"]),
                "evidence": "artifacts/tables/p16e_canary_metrics.csv",
            },
            {
                "claim_id": "redesigned_null_control",
                "domain": "measurement",
                "verdict": "PASS",
                "scope": "isolated synthetic null control",
                "primary_metric": "clustered slope p-value",
                "value": float(null16["candidate_return_p_value"]),
                "evidence": "artifacts/tables/p16e_canary_metrics.csv",
            },
            {
                "claim_id": "implementation",
                "domain": "portfolio",
                "verdict": "REJECT_PRIMARY_IMPLEMENTATION",
                "scope": "conservative value-weighted SIC1-neutral 6m",
                "primary_metric": "annualized net return",
                "value": float(migration17["annualized_net_return"]),
                "evidence": "artifacts/tables/p17e_primary_cost_results.csv",
            },
            {
                "claim_id": "taxonomy",
                "domain": "clustering",
                "verdict": "REJECT_CURRENT_TAXONOMY",
                "scope": "three isolated synthetic scenarios",
                "primary_metric": "minimum noise/unassigned rate",
                "value": float(p18["noise_or_unassigned_rate"].min()),
                "evidence": "artifacts/tables/p18e_archetype_summary.csv",
            },
            {
                "claim_id": "real_market_existence",
                "domain": "external validity",
                "verdict": "NOT_TESTED",
                "scope": "real data",
                "primary_metric": "real-data runs",
                "value": 0.0,
                "evidence": "artifacts/manifests/p19e_final_closeout.json",
            },
            {
                "claim_id": "causality",
                "domain": "causal inference",
                "verdict": "NOT_TESTED",
                "scope": "all phases",
                "primary_metric": "causal designs",
                "value": 0.0,
                "evidence": "docs/p19e_final_closeout_preregistration.md",
            },
            {
                "claim_id": "project_status",
                "domain": "closeout",
                "verdict": "COMPLETE_MIXED_NEGATIVE",
                "scope": "full research program",
                "primary_metric": "open analytical phases",
                "value": 0.0,
                "evidence": "report.md",
            },
        ]
    )


def _readme(
    original_ic: float,
    migration_ic: float,
    migration_p: float,
    net_return: float,
    minimum_noise: float,
) -> str:
    return f"""# Corporate Niche Dynamics

*Corporate Niche Dynamics and Asset Prices*

This repository is the completed, synthetic-first Business Niche Hypercube
research program. It tests whether point-in-time corporate operating states
and changes can measure migration, predict survival and returns, survive
implementation costs, and form stable descriptive archetypes.

## Final verdict

The project is a reproducible mixed/negative result:

- Engineering and point-in-time safeguards: **PASS**
- Original observable migration signal: **REJECTED** (six-month IC
  `{original_ic:.4f}`)
- Locked redesign: **PASS_SYNTHETIC_6X_ONLY** (IC
  `{migration_ic:.4f}`, clustered p=`{migration_p:.4f}`)
- Primary cost-aware implementation: **REJECT_PRIMARY_IMPLEMENTATION**
  (`{100.0 * net_return:.2f}%` annualized net)
- Current archetype taxonomy: **REJECT_CURRENT_TAXONOMY** (at least
  `{100.0 * minimum_noise:.2f}%` noise/unassigned)
- Real-market existence, causality, and investability: **NOT TESTED**

The redesigned measure detects a deliberately amplified synthetic channel, but
costs erase the primary portfolio and clustering does not yield stable,
broadly covering business types. No real-data run occurred.

## Repository identity

Recommended repository name: **`{REPOSITORY_NAME}`**

The Python distribution and historical local folder retain
`business-niche-hypercube` so frozen receipts remain reproducible.

## Evidence map

- `report.md` — final research closeout and interpretation
- `artifacts/tables/p19e_claim_ledger.csv` — machine-readable claim ledger
- `artifacts/manifests/p19e_final_closeout.json` — final provenance receipt
- `docs/archive/p10_report.md` — byte-identical original P10 report
- `docs/p16e_isolated_canary_results.md` — repaired-signal canary
- `docs/p17e_isolated_cost_canary_results.md` — cost/capacity result
- `docs/p18e_isolated_archetype_canary_results.md` — clustering result

## Reproduce

Python 3.11+ is required. The frozen environment is in `uv.lock`.

```powershell
uv sync --extra dev
.\\.venv\\Scripts\\python.exe scripts\\run_full_pipeline.py --config configs\\synthetic.yaml
.\\.venv\\Scripts\\python.exe scripts\\run_p19e_final_closeout.py
.\\.venv\\Scripts\\python.exe scripts\\validate_p19e_final_closeout.py
```

The first command reproduces the original P0-P10 synthetic pipeline. The P19E
commands rebuild and validate the repository-facing closeout from signed
post-P10 artifacts; they fit no model and read no truth table.

This is research code, not investment advice, a live strategy, or a claim of
production readiness.
"""


def _report(
    p10: pd.Series,
    migration16: pd.Series,
    null16: pd.Series,
    migration17: pd.Series,
    null17: pd.Series,
    p18: pd.DataFrame,
) -> str:
    regime18 = p18.loc[p18["scenario"].eq("regime_shift")].iloc[0]
    return f"""# Corporate Niche Dynamics — Final Research Closeout

## Executive verdict

The Business Niche Hypercube research program is complete. Its infrastructure
is reproducible, but the economically relevant conclusions are mixed and
mostly negative. The original signal fails, the locked redesign works only as
a deliberately amplified synthetic proof of concept, the primary portfolio is
negative after costs, and the current clustering design does not produce a
stable taxonomy.

No real-data study occurred. Nothing in this repository establishes a
real-market effect, causality, or investability.

## Original preregistered result

The original `migration_surprise` specification failed the synthetic recovery
gate. In the original migration scenario its six-month rank IC was
`{p10['p7_primary_ic']:.4f}` and its controlled coefficient was
`{p10['p7_primary_coefficient']:.4f}` (p=`{p10['p7_primary_p_value']:.3f}`,
Holm p=`{p10['p7_primary_holm_p_value']:.3f}`). The hidden oracle remained
recoverable. The original P10 conclusion is therefore retained as a valid
negative result and archived byte-identically under `docs/archive/`.

## Locked exploratory redesign

The outcome-blind redesign selected `anchored_axis_innovation`, then evaluated
it only on the isolated 6x synthetic canary. It covered
`{100.0 * migration16['coverage_vs_benchmark']:.2f}%` of benchmark rows,
correlated `{migration16['candidate_truth_spearman']:.4f}` with the injected
migration state, and produced a six-month return IC of
`{migration16['candidate_return_spearman']:.4f}`. The clustered slope was
`{migration16['candidate_return_slope']:.4f}` with p=
`{migration16['candidate_return_p_value']:.4f}`. The null-control IC was
`{null16['candidate_return_spearman']:.4f}` and its clustered p-value was
`{null16['candidate_return_p_value']:.3f}`.

This proves that the repaired measurement pipeline can detect a sufficiently
strong channel that was deliberately injected into synthetic returns. It does
not prove that the channel exists in real data, and it does not convert the
original confirmatory failure into a positive finding.

## Implementability

Under the frozen conservative, value-weighted, dated-SIC1-neutral primary
portfolio, the migration canary earned
`{100.0 * migration17['annualized_gross_capacity_return']:.2f}%` annualized
gross after capacity but `{100.0 * migration17['annualized_net_return']:.2f}%`
net. Net Sharpe was `{migration17['annualized_net_sharpe']:.2f}`, average
turnover `{100.0 * migration17['average_turnover']:.2f}%`, capacity fill
`{100.0 * migration17['average_capacity_fill_ratio']:.2f}%`, and maximum
drawdown `{100.0 * migration17['net_maximum_drawdown']:.2f}%`. The null
control was also negative at
`{100.0 * null17['annualized_net_return']:.2f}%` net.

The primary implementation is rejected. Positive gross separation from the
null control is insufficient when spread/slippage, borrow, turnover, and
capacity erase it.

## Archetypes

All three frozen P9 canaries formed two training clusters, but overall
noise/unassigned rates were:

- null alpha: `{100.0 * p18.loc[p18['scenario'].eq('null_alpha'), 'noise_or_unassigned_rate'].iloc[0]:.2f}%`
- migration alpha: `{100.0 * p18.loc[p18['scenario'].eq('migration_alpha'), 'noise_or_unassigned_rate'].iloc[0]:.2f}%`
- regime shift: `{100.0 * regime18['noise_or_unassigned_rate']:.2f}%`

Null and migration stability ARIs were effectively zero. Every regime-shift
stability refit failed to recover two clusters. The small Archetype B pocket
is primarily low innovation, but coverage, persistence, and outcome ordering
are too weak for a defensible business taxonomy. The current taxonomy is
rejected without tuning it after the fact.

## Claim boundary

- Engineering reproducibility: **PASS**
- Synthetic measurement proof of concept: **PASS at 6x only**
- Primary implementation: **REJECT**
- Current taxonomy: **REJECT**
- Real-market existence: **NOT TESTED**
- Causality: **NOT TESTED**
- Investability: **NOT SUPPORTED**

The machine-readable version is
`artifacts/tables/p19e_claim_ledger.csv`.

## Final decision

Archive this repository as a completed synthetic research program. Do not run
real data under the current specification and do not present the P9 labels as
economic types. Any future real-data or alternative-signal study must begin
under a new, versioned protocol rather than extending this result informally.

Recommended repository name: **`{REPOSITORY_NAME}`**.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embedding-pid", type=int, default=5600)
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "manifests"
        / "p19e_final_closeout.json",
    )
    parser.add_argument(
        "--claim-ledger",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "tables"
        / "p19e_claim_ledger.csv",
    )
    args = parser.parse_args()

    if args.report.exists() or args.claim_ledger.exists():
        raise FileExistsError("Refusing to overwrite completed P19E outputs.")
    preregistration = (
        PROJECT_ROOT / "docs" / "p19e_final_closeout_preregistration.md"
    )
    if not preregistration.is_file():
        raise FileNotFoundError("P19E preregistration is missing.")

    before = _phase_manifest_hashes()
    embedding_alive_before = _pid_alive(args.embedding_pid)
    if not embedding_alive_before:
        raise RuntimeError("Authorized embedding worker is not alive at closeout start.")
    archives = _archive_p10_documents()

    p10_path = PROJECT_ROOT / "artifacts" / "tables" / "p10_report_statistics.csv"
    p16_path = PROJECT_ROOT / "artifacts" / "tables" / "p16e_canary_metrics.csv"
    p17_path = (
        PROJECT_ROOT / "artifacts" / "tables" / "p17e_primary_cost_results.csv"
    )
    p18_path = (
        PROJECT_ROOT / "artifacts" / "tables" / "p18e_archetype_summary.csv"
    )
    p10 = pd.read_csv(p10_path)
    p16 = pd.read_csv(p16_path)
    p17 = pd.read_csv(p17_path)
    p18 = pd.read_csv(p18_path)

    claims = _claim_ledger(p10, p16, p17, p18)
    args.claim_ledger.parent.mkdir(parents=True, exist_ok=True)
    claims.to_csv(args.claim_ledger, index=False, lineterminator="\n")

    original = p10.loc[p10["scenario"].eq("migration_alpha")].iloc[0]
    migration16 = p16.loc[p16["scenario"].eq("migration_alpha")].iloc[0]
    null16 = p16.loc[p16["scenario"].eq("null_alpha")].iloc[0]
    migration17 = p17.loc[p17["scenario"].eq("migration_alpha")].iloc[0]
    null17 = p17.loc[p17["scenario"].eq("null_alpha")].iloc[0]
    _atomic_text(
        PROJECT_ROOT / "README.md",
        _readme(
            float(original["p7_primary_ic"]),
            float(migration16["candidate_return_spearman"]),
            float(migration16["candidate_return_p_value"]),
            float(migration17["annualized_net_return"]),
            float(p18["noise_or_unassigned_rate"].min()),
        ),
    )
    _atomic_text(
        PROJECT_ROOT / "report.md",
        _report(
            original,
            migration16,
            null16,
            migration17,
            null17,
            p18,
        ),
    )

    after = _phase_manifest_hashes()
    if before != after:
        raise RuntimeError("An existing phase manifest changed during P19E.")
    embedding_alive_after = _pid_alive(args.embedding_pid)
    if not embedding_alive_after:
        raise RuntimeError("Embedding worker exited during P19E closeout.")

    source_paths = [
        preregistration,
        p10_path,
        p16_path,
        p17_path,
        p18_path,
        *[
            PROJECT_ROOT / "artifacts" / "manifests" / f"{phase}_manifest.json"
            for phase in ("p13f", "p14f", "p15e", "p16e", "p17e", "p18e")
        ],
    ]
    outputs = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "report.md",
        args.claim_ledger,
    ]
    payload = {
        "schema_version": 1,
        "phase": "P19E",
        "version": "hypercube-final-repository-closeout-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS",
        "project_status": "COMPLETE_MIXED_NEGATIVE",
        "recommended_repository_name": REPOSITORY_NAME,
        "paper_subtitle": "Corporate Niche Dynamics and Asset Prices",
        "frozen_verdicts": {
            "engineering": "PASS",
            "redesigned_measurement": "PASS_SYNTHETIC_6X_ONLY",
            "primary_implementation": "REJECT_PRIMARY_IMPLEMENTATION",
            "archetypes": "REJECT_CURRENT_TAXONOMY",
            "real_market_existence": "NOT_TESTED",
            "causality": "NOT_TESTED",
            "investability": "NOT_SUPPORTED",
        },
        "source_records": [_record(path) for path in source_paths],
        "output_records": [_record(path) for path in outputs],
        "p10_archives": archives,
        "existing_manifest_hashes_before": before,
        "existing_manifest_hashes_after": after,
        "existing_manifests_unchanged": True,
        "analytical_model_fit": False,
        "synthetic_truth_read": False,
        "real_data_run": False,
        "p10_rerun": False,
        "embedding_isolation": {
            "pid": args.embedding_pid,
            "alive_before": embedding_alive_before,
            "alive_after": embedding_alive_after,
            "stopped": False,
            "duplicated": False,
            "reconfigured": False,
        },
    }
    atomic_write_json(args.report, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
