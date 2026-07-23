# P15E — Locked-Proxy Power-Calibration Results

## Gate result

`GO` at `6x`.

| Multiplier | Oracle detection | Proxy detection | Median oracle IC | Median proxy IC | All gates |
|---:|---:|---:|---:|---:|:---:|
| 1x | 0.25 | 0.00 | 0.0089 | -0.0047 | No |
| 2x | 0.79 | 0.02 | 0.0194 | 0.0004 | No |
| 4x | 1.00 | 0.39 | 0.0406 | 0.0103 | No |
| 6x | 1.00 | 0.92 | 0.0618 | 0.0203 | Yes |
| 8x | 1.00 | 0.99 | 0.0828 | 0.0302 | Yes |
| 10x | 1.00 | 0.99 | 0.1036 | 0.0401 | Yes |

The smallest calibration-qualified multiplier, `6x`, also passed the disjoint
50-draw evaluation:

- oracle detection: `1.00`
- locked-proxy detection: `0.88`
- median oracle rank IC: `0.0617`
- median proxy rank IC: `0.0202`
- null proxy false-positive rate: `0.00`

The independent validator recomputed the complete bootstrap protocol and
passed with zero errors.

## Decision

The canary proposal is an anchored-axis innovation with synthetic
`migration_alpha_monthly = 0.024`, six times the original `0.004`. This is a
power-calibration setting, not an economic estimate or an investable return
claim.

No new synthetic scenario has been generated. P0-P10 remain frozen, no real
data was opened, and the next action requires a separately isolated canary
build that cannot overwrite the original scenarios.
