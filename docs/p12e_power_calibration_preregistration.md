# P12E — Exploratory Power-Calibration Preregistration

## Boundary

P12E is a CPU-only, post-closeout exploratory study. It does not regenerate a
synthetic scenario, change P0-P10, inspect real data, or authorize a real-data
run. The protocol below was frozen before executing P12E.

## Inputs

- frozen migration-alpha six-month P7 targets;
- frozen synthetic truth sidecar;
- the original `0.004` monthly migration-alpha injection;
- the original 12-month exponential decay and six-month primary horizon.

## Linearized counterfactual

The current six-month return is decomposed into a residual plus the expected
six-month fraction of the injected alpha. Candidate multipliers
`1, 2, 4, 6, 8, 10` scale only the injection. This is a power-calibration
approximation; it does not replace a full generator canary because it does not
replay monthly clipping and compounding.

## Resampling

- issuer-cluster bootstrap;
- 100 calibration draws with seeds beginning at `12000`;
- 50 held-out evaluation draws with seeds beginning at `22000`;
- duplicate sampled issuers receive distinct bootstrap cluster identifiers;
- issuer-clustered slope inference in every draw.

## Frozen selection gates

Select the smallest multiplier satisfying all calibration gates:

- positive oracle slope with clustered `p < 0.05` in at least 80% of draws;
- positive observable-signal slope with clustered `p < 0.05` in at least 80%;
- null observable false-positive rate no greater than 10%;
- median oracle rank IC at least 0.04;
- median observable rank IC at least 0.01.

The selected multiplier must pass the same gates on the held-out evaluation
seeds. Otherwise P12E returns `NO_GO`.

## Stop rule

P12E stops before generating a new synthetic corpus. A passing multiplier is
only a proposed canary parameter. A later versioned canary must replay the full
monthly return generator and pass its own oracle/null gates before any
observable-feature comparison or real-data study.
