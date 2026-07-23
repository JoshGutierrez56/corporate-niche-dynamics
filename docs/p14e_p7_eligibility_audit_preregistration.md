# P14E — P7 Eligibility-Sample Proxy Audit

## Question

P11E measured only `0.204` rank correlation between the P7 observable proxy
and synthetic migration truth, while P13F found much stronger late-period
alignment on the full P5 surface. P14E tests whether P7 return-path eligibility
and signal availability explain that downstream loss of measurement quality.

## Boundary

- The candidate is locked to P13F's selected
  `anchored_axis_innovation`; no candidate is reselected.
- The audit reads the P13F candidate artifact, synthetic migration truth, and
  only the P7 target key fields `gvkey`, `datadate`, `fyear`,
  `horizon_months`, and `target_valid`.
- It must not read forward returns, injected alpha, costs, portfolios, or real
  data.
- It does not change P0-P10 or generate a new synthetic corpus.

## Population and gates

- Primary horizon: six months
- Require `target_valid == true`
- Require the frozen benchmark to be nonmissing, matching the P11E population
- Calibration years: 2002-2012
- Locked evaluation years: 2013-2018
- Report the selected proxy and existing P5 benchmark on identical rows.

The audit returns `GO` only if the locked proxy:

1. has overall P7-eligible Spearman correlation at least `0.35`;
2. improves overall Spearman correlation over the benchmark by at least `0.10`;
3. has Spearman correlation at least `0.30` in both calibration and locked
   evaluation years;
4. has Spearman correlation at least `0.20` in both evaluation blocks,
   2013-2015 and 2016-2018; and
5. has nonmissing eligible coverage at least `95%` of the benchmark.

`GO` authorizes only a separately frozen proxy-aware power calibration. It
does not authorize a new corpus, real-data run, or trading-performance claim.
