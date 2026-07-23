# P16E — Isolated 6× Canary Results

## Result

`PASS`

The isolated canary regenerated all three synthetic scenarios and completed
P1-P7 in `1,226.5` seconds. Every phase build and independent validator
passed. The run used no GPU, network, WRDS, or real data.

The original P5 `migration_surprise` still failed its scientific recovery
check at the stronger injection, with six-month IC `-0.0086`. Its oracle slope
was healthy at `0.7717`, confirming that the regenerated return injection was
present and correctly scaled.

## Locked-proxy gates

Applying the preregistered `anchored_axis_innovation` to the regenerated P7
paths produced:

- migration rows / firms: `22,083` / `2,227`
- candidate-to-truth Spearman: `0.46165`
- coverage versus benchmark: `96.31%`
- candidate/return Spearman: `0.01968`
- candidate return slope: `0.01298`
- issuer-clustered p-value: `0.00211`
- oracle return slope: `0.75718`
- oracle issuer-clustered p-value: `< 7e-19`

The regenerated null control retained:

- null candidate/return Spearman: `0.00793`
- null candidate slope p-value: `0.2541`

All eight frozen gates passed. The independent P16E validator recomputed the
saved result with zero errors.

## Interpretation

The Hypercube signal-recovery failure has been repaired in an isolated
synthetic canary. The repair requires both:

1. replacing the unstable failure-model migration residual with the
   transparent anchored-axis innovation; and
2. using the power-qualified 6× synthetic injection (`0.024` monthly).

This remains synthetic validation. P8-P10 were not run on the canary, the main
P0-P10 outputs were not modified, and no real-data or investable-performance
claim follows.
