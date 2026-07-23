# P13E — Outcome-Blind Migration-Proxy Redesign

## Boundary

P13E is a post-closeout synthetic diagnostic. It does not alter P0-P10,
inspect real data, read return outcomes while constructing candidates, change
the synthetic return injection, or generate a new corpus. Synthetic truth may
be opened only after the candidate artifact and its construction receipt have
been written.

## Motivation

P12E showed that increasing the return injection repairs oracle power but not
the observable `migration_surprise` proxy. The redesign therefore targets the
observable state measurement rather than return strength.

## Frozen population

- Scenario: `migration_alpha`
- Source: the three-year P5 OOS surface, one row per accounting event
- Required history: a prior observation for the same issuer
- Candidate construction may read P5 axes, dates, identifiers, and the existing
  benchmark only.
- Candidate construction must not read synthetic truth, injected alpha, P7
  targets, returns, or portfolio results.

## Frozen candidates

Every P3 axis already follows the convention that a larger value is more
viable or defensible.

1. `relative_axis_innovation`: equal-weight mean of the six relative axes,
   minus its expanding prior-year AR(1) expectation.
2. `anchored_axis_innovation`: equal-weight mean of the six anchored axes,
   minus its expanding prior-year AR(1) expectation.
3. `blended_axis_innovation`: a 50/50 mean of the relative and anchored levels,
   minus its expanding prior-year AR(1) expectation.
4. `migration_surprise`: the existing P5 feature, retained only as the frozen
   benchmark.

A level requires at least four of its six axes. For each prediction year, the
intercept and persistence coefficient are estimated from issuer transitions
whose current observation is in an earlier calendar year. At least 1,000
prior transitions and two prior calendar years are required. Coefficients are
clipped to `[0, 1]`; no industry, return, or truth variable enters the fit.

## Frozen evaluation

Candidate selection uses synthetic migration truth, never returns:

- calibration years: 2002-2012
- held-out years: 2013-2018
- selection: highest calibration Spearman correlation, deterministic
  alphabetical tie break
- held-out gate: Spearman at least `0.35`
- stability gate: Spearman at least `0.20` in each of the held-out blocks
  2013-2015 and 2016-2018
- coverage gate: held-out nonmissing rows at least `95%` of the benchmark's
  nonmissing rows
- improvement gate: held-out Spearman at least `0.10` above the benchmark

The result is `GO` only if one frozen candidate passes every held-out gate.
Otherwise it is `NO_GO`, and no new synthetic corpus or return calibration may
be started from P13E.

## Validation

An independent validator must reconstruct the candidate table from P5, confirm
its byte-independent values, reopen truth, and recompute selection and every
gate. P0-P10 outputs remain immutable.
