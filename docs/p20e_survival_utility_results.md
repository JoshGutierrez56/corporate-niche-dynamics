# P20E Survival Utility Results

## Verdict

**NO_ROBUST_INCREMENTAL_UTILITY**

The six-dimensional niche representation is not a robust general-purpose
predictor of firm survival in the existing synthetic experiments. It may have
limited value as a stress-regime feature, but that narrower result requires
independent confirmation.

## Primary incremental test

P20E compared a conventional financial ridge-logit model with the same model
augmented by all six relative and all six anchored Hypercube axes. Both used
the original expanding, horizon-purged 3/5-year folds, inner-validation
hyperparameter selection, and validation-only Platt calibration.

Across 24 outer folds:

- mean AUC improvement: `-0.002732`;
- median AUC improvement: `+0.001085`;
- AUC wins: `13 / 24`;
- mean Brier improvement: `-0.00000411`;
- Brier wins: `11 / 24`;
- positive scenario-horizon cells: `2 / 6`.

None of the four frozen robustness gates passed.

Weighted across all scenarios and horizons:

- financial baseline AUC: `0.5464`;
- financial-plus-Hypercube AUC: `0.5443`;
- financial baseline average precision: `0.03514`;
- financial-plus-Hypercube average precision: `0.03438`.

The augmented model therefore did not improve the already modest absolute
predictive performance.

## Where the axes did and did not help

Mean AUC lift by scenario and horizon:

| Scenario | Horizon | Mean AUC lift | Winning folds |
|---|---:|---:|---:|
| Null alpha | 3 years | `-0.00882` | 1 / 4 |
| Null alpha | 5 years | `-0.01951` | 1 / 4 |
| Migration alpha | 3 years | `-0.00846` | 2 / 4 |
| Migration alpha | 5 years | `-0.00596` | 2 / 4 |
| Regime shift | 3 years | `+0.00915` | 3 / 4 |
| Regime shift | 5 years | `+0.01721` | 4 / 4 |

The stress-regime pattern is the only encouraging result: seven of eight
regime-shift folds improved and the five-year lift exceeded the `0.01`
materiality threshold. Because this is one pre-existing synthetic scenario
and the other four scenario-horizon cells were negative, it is exploratory
evidence rather than a promoted claim.

## Existing model evidence

The pre-existing Hypercube-only `combined_axes_logit` beat coarse historical
benchmarks:

- versus industry failure rates: mean AUC lift `+0.0442`, 20/24 wins;
- versus occupied-cell failure rates: mean AUC lift `+0.0321`, 21/24 wins.

It did not beat stronger financial benchmarks:

- versus profitability logit: mean AUC lift `-0.00830`, 8/24 wins;
- versus distress logit: mean AUC lift `-0.00168`, 12/24 wins.

The existing P6 time-varying performance-failure model was similarly modest:

- mean cause-specific concordance: `0.5674`;
- mean ROC AUC: `0.5625`;
- ROC-AUC range: `0.5016` to `0.6242`;
- latest regime-shift fold AUC: `0.5016`, with calibration slope `0.0925`.

That final stress fold is effectively nondiscriminating and poorly calibrated.

## Interpretation

The defensible conclusion is:

> Hypercube coordinates contain more survival information than a coarse
> industry or occupied-cell lookup, and they may detect survival risk during
> structural regime changes. They do not add reliable predictive value beyond
> standard profitability and distress variables across normal synthetic
> environments.

The repository should remain a research framework, not be marketed as a firm
survival predictor. A next study would need a separately frozen regime-aware
interaction hypothesis and real point-in-time firm-exit data.

## Validation and claim boundary

- `376,196` saved outer-test predictions were independently reopened.
- All 48 metric rows and 24 paired comparisons recomputed exactly.
- Seven frozen sources and seven outputs passed byte/hash checks.
- No return, portfolio, synthetic-truth, real-data, or embedding artifact was
  read.
- This is synthetic predictive evidence only; it is not causal and has not
  been validated on real firms.
