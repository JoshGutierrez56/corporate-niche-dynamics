# P12E — Power-Calibration Results

## Gate result

`NO_GO`

The independent P12E validator passed, but no candidate injection multiplier
passed all frozen calibration gates. No held-out multiplier was selected and no
new synthetic scenario was generated.

| Multiplier | Oracle detection | Observable detection | Median oracle IC | Median observable IC |
|---:|---:|---:|---:|---:|
| 1x | 0.24 | 0.00 | 0.0088 | -0.0187 |
| 2x | 0.78 | 0.00 | 0.0190 | -0.0167 |
| 4x | 1.00 | 0.00 | 0.0395 | -0.0124 |
| 6x | 1.00 | 0.01 | 0.0599 | -0.0082 |
| 8x | 1.00 | 0.05 | 0.0800 | -0.0041 |
| 10x | 1.00 | 0.32 | 0.1000 | 0.0001 |

The null observable false-positive rate was zero in the calibration draws.

## Interpretation

The oracle becomes reliably detectable around 4x-6x, but strengthening the
return injection alone does not repair the current observable
`migration_surprise` feature. At 10x, the oracle is strong while the observable
feature still fails the 80% detection and 0.01 median-rank-IC gates.

P12E therefore rejects a simple "turn up the alpha" repair. The next redesign
must improve the observable state/migration measurement itself and then repeat
power calibration on disjoint seeds.

P13F-P15E completed that sequence. The locked anchored-axis innovation passed
the exact P7 eligibility audit, and a separate power calibration selected a
`6x` canary setting. See `docs/p15e_proxy_power_calibration_results.md`.

## Stop state

- new synthetic scenario generated: **NO**
- real data inspected: **NO**
- frozen P0-P10 outputs changed: **NO**
- proposed monthly injection selected: **NO**
- next authorized action: outcome-blind observable-feature redesign only
