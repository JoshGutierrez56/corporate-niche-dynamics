# P15E — Locked-Proxy Alpha-Power Calibration

## Authorization boundary

P14F qualified the locked `anchored_axis_innovation` proxy on the exact P7
eligibility contract. P15E may now read the existing six-month forward return
and injected-alpha fields solely for power calibration. It must not generate a
new synthetic corpus, change P0-P10, inspect real data, or run portfolio
selection.

## Frozen inputs and population

- P13F five-year candidate artifact
- migration-alpha P7 six-month target-valid rows
- nonmissing existing P5 benchmark, matching P11E/P14F
- fiscal years 2002-2018
- nonmissing locked `anchored_axis_innovation`
- existing migration-alpha truth sidecar

## Frozen calibration

- injection multipliers: `1, 2, 4, 6, 8, 10`
- issuer-cluster bootstrap
- calibration: 100 replicates with seeds `32000` through `32099`
- held-out evaluation: 50 replicates with seeds `42000` through `42049`
- null return: observed forward excess return less the expected six-month
  share of the existing injected alpha
- selected multiplier: smallest candidate passing every calibration gate

## Gates

The P12E thresholds are carried forward unchanged:

- oracle positive clustered slope with `p < 0.05`: at least `80%`
- locked-proxy positive clustered slope with `p < 0.05`: at least `80%`
- null locked-proxy false-positive rate: at most `10%`
- median oracle rank IC: at least `0.04`
- median locked-proxy rank IC: at least `0.01`

The selected multiplier must pass the same gates on the disjoint held-out
seeds. `GO` authorizes only a new-scenario canary proposal at the selected
monthly alpha; it does not itself generate that scenario.
