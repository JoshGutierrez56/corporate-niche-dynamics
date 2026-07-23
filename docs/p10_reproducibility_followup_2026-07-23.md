# P10 Reproducibility Follow-up — 2026-07-23

## Status

The frozen synthetic P0-P10 analytical result is unchanged:

- primary statistical conclusion: `MIGRATION_SIGNAL_NOT_RECOVERED`;
- primary implementation conclusion: `NEGATIVE_AFTER_COSTS`;
- archetype conclusion: `UNSTABLE_MOSTLY_NOISE`;
- real-data run: `false`.

No result table, processed analytical bundle, model, figure, hypothesis, cost
assumption, or scientific conclusion was changed during this follow-up.

## Independent checks

- The primary repository passed `64` tests after adding the fresh-copy
  regression test.
- The cold reproduction rebuilt and validated every analytical phase P1-P10.
- After restoring the empty P0 output-directory contract, the cold repository
  passed its original `63` tests.
- `58` substantive compact tables and figures were byte-identical between the
  primary and cold runs.
- `p10_figure_registry.csv` differed only because it stores the absolute
  repository root. Replacing each repository root with the same placeholder
  made the registries identical.

The cold run's historical `full_pipeline_receipt.json` remains `FAIL` because
its final test gate ran before the omitted empty directories were restored.
That receipt is preserved rather than rewritten.

## Reproducibility defect and fix

Archive and copy workflows can omit empty directories. The cold tree therefore
lacked `data/interim` and `artifacts/models`, even though both are part of the
P0 repository contract. Every numerical build and validator passed; only the
final tree-contract smoke test failed.

`scripts/run_full_pipeline.py` now recreates all empty P0 output directories
before phase execution. `tests/test_full_pipeline_scaffold.py` independently
tests that behavior against a new temporary root.

## Scientific interpretation

The saved negative result is reproducible and begins before implementation
costs:

- migration-alpha primary P7 gross return: `0.9607%` annualized;
- gross return after capacity: `1.2212%`;
- conservative transaction cost: `5.8164%`;
- borrow cost: `2.0162%`;
- net return: `-6.6114%`;
- net Sharpe: `-1.2274`;
- maximum drawdown: `-56.4694%`.

One-month delayed execution improved the primary migration-alpha net return to
`-4.9339%`, but it remained negative. Even the best saved exploratory
specification under the low-cost case remained negative in all three
scenarios.

P11E subsequently found that the hidden oracle itself was not statistically
detectable with issuer-clustered uncertainty. Its slope point estimate passed
the original magnitude-only bound, but its rank correlation and explained
return variance were near zero. P7 therefore does not cleanly distinguish an
inadequate observable feature from an underpowered injected-alpha experiment.
See `docs/p11e_alpha_power_diagnostic.md`.

## Next valid research action

Do not run a confirmatory real-data study under the failed specification.
Any continuation must be a separately versioned exploratory redesign with
oracle detectability and Monte Carlo power gates frozen before generating a
new synthetic scenario or inspecting real outcomes.
