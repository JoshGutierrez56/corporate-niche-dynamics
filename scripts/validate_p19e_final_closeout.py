"""Independently validate the P19E final repository closeout."""

from __future__ import annotations

import argparse
import codecs
import ctypes
import json
import os
import re
from pathlib import Path
from typing import Any

import pandas as pd

from hypercube.data import atomic_write_json, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ISOLATED_ROOT = Path(
    os.environ.get(
        "HYPERCUBE_CANARY_ROOT",
        PROJECT_ROOT.parent / ".p16-hypercube-canary",
    )
)


def _read_text_auto(path: Path) -> str:
    payload = path.read_bytes()
    if payload.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        return payload.decode("utf-16")
    if payload.startswith(codecs.BOM_UTF8):
        return payload.decode("utf-8-sig")
    return payload.decode("utf-8")


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name != "nt":
        try:
            os.kill(pid, 0)
        except OSError:
            return False
        return True
    handle = ctypes.windll.kernel32.OpenProcess(  # type: ignore[attr-defined]
        0x1000, False, pid
    )
    if not handle:
        return False
    ctypes.windll.kernel32.CloseHandle(handle)  # type: ignore[attr-defined]
    return True


def _check_record(
    record: dict[str, Any],
    root: Path,
    errors: list[str],
    label: str,
) -> None:
    path = root / record["path"]
    if not path.is_file():
        errors.append(f"{label} missing: {record['path']}")
        return
    if path.stat().st_size != record["bytes"]:
        errors.append(f"{label} byte mismatch: {record['path']}")
    if sha256_file(path) != record["sha256"]:
        errors.append(f"{label} hash mismatch: {record['path']}")


def _phase_manifest_hashes() -> dict[str, str]:
    root = PROJECT_ROOT / "artifacts" / "manifests"
    return {
        path.name: sha256_file(path)
        for path in sorted(root.glob("*.json"))
        if not path.name.startswith("p19e_")
    }


def _audit_postcloseout_manifests(errors: list[str]) -> int:
    checked = 0
    manifest_root = PROJECT_ROOT / "artifacts" / "manifests"
    for phase in ("p13f", "p14f", "p15e", "p16e", "p17e", "p18e"):
        manifest = json.loads(
            (manifest_root / f"{phase}_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        for record in manifest["files"]:
            root = (
                ISOLATED_ROOT
                if record.get("scope") == "isolated_canary"
                else PROJECT_ROOT
            )
            _check_record(record, root, errors, f"{phase} manifest")
            checked += 1
    return checked


def _claim(
    claims: pd.DataFrame,
    claim_id: str,
    verdict: str,
    value: float,
    errors: list[str],
) -> None:
    selected = claims.loc[claims["claim_id"].eq(claim_id)]
    if len(selected) != 1:
        errors.append(f"Claim ledger has duplicate/missing {claim_id}.")
        return
    row = selected.iloc[0]
    if row["verdict"] != verdict:
        errors.append(f"Claim verdict mismatch: {claim_id}.")
    if abs(float(row["value"]) - float(value)) > 1e-12:
        errors.append(f"Claim value mismatch: {claim_id}.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-report",
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
    parser.add_argument(
        "--test-log",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "logs" / "p19e_pytest.txt",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT
        / "artifacts"
        / "manifests"
        / "p19e_final_closeout_validation.json",
    )
    args = parser.parse_args()

    result = json.loads(args.input_report.read_text(encoding="utf-8"))
    errors: list[str] = []
    for record in result["source_records"]:
        _check_record(record, PROJECT_ROOT, errors, "P19E source")
    for record in result["output_records"]:
        _check_record(record, PROJECT_ROOT, errors, "P19E output")
    for record in result["p10_archives"]:
        _check_record(record, PROJECT_ROOT, errors, "P10 archive")
        if record["sha256"] != record["p10_expected_sha256"]:
            errors.append(
                f"P10 archive differs from P10 receipt: {record['path']}"
            )

    if result["existing_manifest_hashes_before"] != result[
        "existing_manifest_hashes_after"
    ]:
        errors.append("Existing manifest hashes changed during P19E.")
    if _phase_manifest_hashes() != result["existing_manifest_hashes_after"]:
        errors.append("Existing manifest hashes changed after P19E.")
    if result["recommended_repository_name"] != "corporate-niche-dynamics":
        errors.append("Recommended repository name changed.")
    expected_verdicts = {
        "engineering": "PASS",
        "redesigned_measurement": "PASS_SYNTHETIC_6X_ONLY",
        "primary_implementation": "REJECT_PRIMARY_IMPLEMENTATION",
        "archetypes": "REJECT_CURRENT_TAXONOMY",
        "real_market_existence": "NOT_TESTED",
        "causality": "NOT_TESTED",
        "investability": "NOT_SUPPORTED",
    }
    if result["frozen_verdicts"] != expected_verdicts:
        errors.append("P19E frozen verdicts changed.")
    for flag in (
        "analytical_model_fit",
        "synthetic_truth_read",
        "real_data_run",
        "p10_rerun",
    ):
        if result.get(flag) is not False:
            errors.append(f"P19E boundary flag failed: {flag}.")

    p10 = pd.read_csv(
        PROJECT_ROOT / "artifacts" / "tables" / "p10_report_statistics.csv"
    )
    p16 = pd.read_csv(
        PROJECT_ROOT / "artifacts" / "tables" / "p16e_canary_metrics.csv"
    )
    p17 = pd.read_csv(
        PROJECT_ROOT
        / "artifacts"
        / "tables"
        / "p17e_primary_cost_results.csv"
    )
    p18 = pd.read_csv(
        PROJECT_ROOT / "artifacts" / "tables" / "p18e_archetype_summary.csv"
    )
    claims = pd.read_csv(args.claim_ledger)
    original = p10.loc[p10["scenario"].eq("migration_alpha")].iloc[0]
    migration16 = p16.loc[p16["scenario"].eq("migration_alpha")].iloc[0]
    null16 = p16.loc[p16["scenario"].eq("null_alpha")].iloc[0]
    migration17 = p17.loc[p17["scenario"].eq("migration_alpha")].iloc[0]
    _claim(
        claims,
        "original_signal",
        "REJECT_ORIGINAL_SPECIFICATION",
        float(original["p7_primary_ic"]),
        errors,
    )
    _claim(
        claims,
        "redesigned_signal",
        "PASS_SYNTHETIC_6X_ONLY",
        float(migration16["candidate_return_spearman"]),
        errors,
    )
    _claim(
        claims,
        "redesigned_null_control",
        "PASS",
        float(null16["candidate_return_p_value"]),
        errors,
    )
    _claim(
        claims,
        "implementation",
        "REJECT_PRIMARY_IMPLEMENTATION",
        float(migration17["annualized_net_return"]),
        errors,
    )
    _claim(
        claims,
        "taxonomy",
        "REJECT_CURRENT_TAXONOMY",
        float(p18["noise_or_unassigned_rate"].min()),
        errors,
    )
    _claim(
        claims,
        "real_market_existence",
        "NOT_TESTED",
        0.0,
        errors,
    )
    _claim(claims, "causality", "NOT_TESTED", 0.0, errors)
    _claim(
        claims,
        "project_status",
        "COMPLETE_MIXED_NEGATIVE",
        0.0,
        errors,
    )

    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    report_text = (PROJECT_ROOT / "report.md").read_text(encoding="utf-8")
    required_readme = (
        "# Corporate Niche Dynamics",
        "corporate-niche-dynamics",
        "PASS_SYNTHETIC_6X_ONLY",
        "REJECTED",
        "NOT TESTED",
    )
    for phrase in required_readme:
        if phrase not in readme:
            errors.append(f"README missing required phrase: {phrase}")
    required_report = (
        "Original preregistered result",
        "Locked exploratory redesign",
        "Implementability",
        "Archetypes",
        "No real-data study occurred",
        "corporate-niche-dynamics",
    )
    for phrase in required_report:
        if phrase not in report_text:
            errors.append(f"Final report missing required phrase: {phrase}")

    tests_passed = 0
    if not args.test_log.is_file():
        errors.append("P19E pytest receipt is missing.")
    else:
        match = re.search(
            r"(\d+) passed", _read_text_auto(args.test_log)
        )
        tests_passed = int(match.group(1)) if match else 0
        if tests_passed != 83:
            errors.append(f"Expected 83 tests; observed {tests_passed}.")

    manifest_records_checked = _audit_postcloseout_manifests(errors)
    embedding = result["embedding_isolation"]
    if not embedding["alive_before"] or not embedding["alive_after"]:
        errors.append("Embedding worker was not preserved during P19E.")
    for flag in ("stopped", "duplicated", "reconfigured"):
        if embedding[flag] is not False:
            errors.append(f"Embedding isolation flag failed: {flag}.")
    embedding_alive_now = _pid_alive(int(embedding["pid"]))

    payload = {
        "schema_version": 1,
        "phase": "P19E",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "claim_rows": int(len(claims)),
        "tests_passed": tests_passed,
        "postcloseout_manifest_records_checked": manifest_records_checked,
        "p10_archives_byte_identical": not any(
            "P10 archive" in error for error in errors
        ),
        "existing_manifests_unchanged": not any(
            "manifest hashes" in error for error in errors
        ),
        "embedding_pid": int(embedding["pid"]),
        "embedding_alive_during_closeout": True,
        "embedding_alive_at_validation": embedding_alive_now,
        "analytical_model_fit": False,
        "synthetic_truth_read": False,
        "real_data_run": False,
    }
    atomic_write_json(args.report, payload)
    if errors:
        raise SystemExit("\n".join(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
