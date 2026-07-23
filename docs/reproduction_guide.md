# Reproduction guide

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

Before phase execution, the orchestrator restores the empty P0 output
directories that archive and copy workflows may omit. This behavior is covered
by `tests/test_full_pipeline_scaffold.py`. The 2026-07-23 cold-reproduction
evidence is recorded in
`docs/p10_reproducibility_followup_2026-07-23.md`.

Synthetic truth is read only by independent recovery audits. Feature, model,
portfolio, cost, clustering, and figure construction never use hidden truth.
