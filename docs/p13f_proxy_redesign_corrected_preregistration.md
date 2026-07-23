# P13F — Corrected Outcome-Blind Migration-Proxy Redesign

## Correction record

P13E was frozen and run on the three-year P5 surface. That population is valid
for the P5 synthetic recovery diagnostic but not for the P7 return experiment,
which consumes the five-year P5 surface. The P13E artifacts are preserved under
`artifacts/archive/p13e_v1_invalid_three_year_population/` and are excluded
from all decisions.

The invalid run disclosed aggregate candidate correlations. P13F is therefore
an exploratory locked-formula correction, not a pristine confirmatory
holdout. No P13E formula or gate is changed in response to those correlations.

## Boundary

P13F does not alter P0-P10, inspect real data, read return outcomes during
candidate construction, change the synthetic return injection, or generate a
new corpus. Synthetic truth may be opened only after the five-year candidate
artifact and construction receipt have been written.

## Corrected population

- Scenario: `migration_alpha`
- Source: the **five-year** P5 OOS surface used by P7
- Required history: a consecutive-fiscal-year prior observation for the issuer
- Candidate construction reads only P5 axes, dates, identifiers, and the
  existing benchmark.
- Candidate construction must not read synthetic truth, injected alpha, P7
  targets, returns, or portfolio results.

## Locked candidates and gates

P13F carries forward the P13E definitions unchanged:

1. equal-weight relative-axis expanding AR(1) innovation;
2. equal-weight anchored-axis expanding AR(1) innovation;
3. 50/50 blended-axis expanding AR(1) innovation; and
4. existing P5 `migration_surprise` as the benchmark.

At least four of six axes are required. Each prediction-year AR(1) fit uses at
least 1,000 transitions from at least two earlier calendar years, with
persistence clipped to `[0, 1]`.

The unchanged evaluation protocol is:

- calibration years 2002-2012;
- locked evaluation years 2013-2018;
- select the highest calibration Spearman correlation, alphabetical tie break;
- evaluation Spearman at least `0.35`;
- Spearman at least `0.20` in both 2013-2015 and 2016-2018;
- nonmissing coverage at least `95%` of the benchmark; and
- evaluation Spearman improvement at least `0.10` over the benchmark.

`GO` authorizes only a separate, preregistered power calibration using the
selected proxy. It does not authorize new corpus generation, real-data work,
or performance promotion.
