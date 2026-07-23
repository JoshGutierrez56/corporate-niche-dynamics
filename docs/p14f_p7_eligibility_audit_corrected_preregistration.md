# P14F — Corrected P7 Eligibility-Sample Proxy Audit

## Correction record

P14E defined calibration years 2002-2012 and evaluation years 2013-2018 but
mistakenly included 2001 in its "overall" correlation and coverage gates.
Every 2001 row lacked the two prior calendar years required by the locked
candidate, mechanically lowering coverage. P14E remains archived under
`artifacts/archive/p14e_v1_included_precalibration_2001/` and is excluded from
the decision.

P14F changes no candidate, threshold, or in-window observation. It applies all
overall metrics to the explicitly frozen union of calibration and evaluation
years, 2002-2018.

## Locked protocol

- Candidate: P13F `anchored_axis_innovation`
- P7 fields read: keys, `horizon_months`, and `target_valid` only
- Primary horizon: six months
- Population: target-valid rows with a nonmissing frozen benchmark
- Analysis years: 2002-2018
- Calibration years: 2002-2012
- Evaluation years: 2013-2018
- No forward returns, injected alpha, costs, portfolios, or real data read

The P14E gates remain unchanged:

1. overall Spearman at least `0.35`;
2. improvement over the benchmark at least `0.10`;
3. calibration and evaluation Spearman each at least `0.30`;
4. 2013-2015 and 2016-2018 Spearman each at least `0.20`; and
5. nonmissing coverage at least `95%` of the benchmark.

`GO` authorizes only a separately frozen proxy-aware power calibration.
