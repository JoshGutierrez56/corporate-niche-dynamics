# P20E Survival Utility Audit Protocol

## Purpose

P20E asks whether the six-dimensional niche representation adds useful
out-of-sample information for predicting firm survival. It replaces the return
question with a failure-prediction question; it does not revise or reinterpret
the rejected return results.

This is a **retrospective extension**, not a blind preregistration. The frozen
P4 and P6 metrics had already been generated and inspected before P20E. The
new comparison below had not been fit: a conventional financial baseline
versus the same baseline augmented with all twelve Hypercube axes.

## Frozen inputs

P20E may read only:

- each synthetic scenario's immutable P4 `model_matrix.parquet`;
- the signed P4 fold definitions and validation receipts;
- the signed aggregate P4 and P6 metric tables; and
- the frozen P4 configuration.

It may not read synthetic truth, returns, portfolios, transaction costs, P7-P8
artifacts, real data, or the separate EDGAR embedding corpus.

## Outcome and splits

- Outcome: point-in-time `failure_within_horizon`.
- Horizons: three and five years.
- Scenarios: null alpha, migration alpha, and regime shift.
- Splits: the original four expanding calendar folds for each horizon.
- Purge: equal to the prediction horizon.
- Training and validation labels must be observable before the next split.
- Missing values, scaling, tuning, and calibration are fit using training or
  validation data only, never outer-test data.

## Frozen models

Both models are ridge logistic regressions with the existing P4 grid
`C in {0.01, 0.1, 1, 10}` selected by inner-validation Brier score and the
existing validation-only Platt calibration.

1. `financial_baseline_logit`
   - operating margin;
   - gross profitability;
   - market capitalization;
   - book leverage;
   - working capital/assets;
   - operating income/assets;
   - equity/liabilities; and
   - sales/assets.
2. `financial_plus_hypercube_logit`
   - every financial-baseline feature; plus
   - all six relative niche axes; and
   - all six anchored niche axes.

No candidate features, models, hyperparameters, horizons, or scenarios may be
added after the run begins.

## Primary decision gate

`ROBUST_INCREMENTAL_UTILITY` requires all of:

1. mean paired outer-fold ROC-AUC improvement of at least `0.01`;
2. positive AUC improvement in at least 16 of 24 folds;
3. positive mean paired Brier-score improvement;
4. positive mean AUC improvement in every scenario-horizon cell (6 of 6).

Otherwise the primary verdict is `NO_ROBUST_INCREMENTAL_UTILITY`.

Secondary evidence may describe:

- absolute AUC, average precision, Brier score, and log loss;
- performance of the pre-existing Hypercube-only P4 model against coarse
  industry and occupied-cell rates;
- P6 cause-specific performance-failure concordance and AUC; and
- stress-scenario deterioration.

Secondary evidence cannot override the primary gate.

## Claim boundary

P20E can establish only whether Hypercube axes add predictive information in
the existing synthetic data-generating processes. It cannot establish
real-market validity, causality, production readiness, or investability.
