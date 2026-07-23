# Business Niche Hypercube

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
.\.venv\Scripts\python.exe scripts\run_full_pipeline.py --config configs\synthetic.yaml
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
