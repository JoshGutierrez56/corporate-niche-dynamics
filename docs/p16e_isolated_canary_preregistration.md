# P16E — Isolated 6× Synthetic Canary

## Isolation

The canary runs from the directory supplied by `--canary-root`, or by the
`HYPERCUBE_CANARY_ROOT` environment variable. If neither is supplied, it uses
the adjacent `.p16-hypercube-canary` directory. This must be a separate source
and data root. It may not read or write the main project's frozen synthetic
directories. The main P0-P10 artifacts remain immutable.

## Frozen design

- source seed: `20260722`
- scenarios: `null_alpha`, `migration_alpha`, and `regime_shift`
- phases: P1 through validated P7
- migration monthly alpha: `0.024` (`6x` the original `0.004`)
- decay: the existing 12-month exponential schedule
- locked observable: P13F `anchored_axis_innovation`
- analysis years: 2002-2018
- primary return horizon: six months
- no P8 costs, P9 clustering, P10 reporting, real data, or portfolio
  specification search

The candidate is reconstructed from each canary's five-year P5 surface using
the unchanged P13F formulas. Evaluation uses the P7 target-valid,
benchmark-nonmissing population and then requires the candidate to be
nonmissing.

## Frozen realized-canary gates

For `migration_alpha`:

1. candidate-to-truth Spearman at least `0.35`;
2. candidate coverage versus the benchmark at least `0.95`;
3. oracle injected-alpha slope between `0.40` and `1.30`, positive with
   issuer-clustered `p < 0.05`;
4. candidate/forward-return Spearman at least `0.01`; and
5. candidate slope positive with issuer-clustered `p < 0.05`.

For `null_alpha`:

6. absolute candidate/forward-return Spearman at most `0.03`; and
7. candidate slope must not be positive with issuer-clustered `p < 0.05`.

`PASS` means the isolated regenerated paths recover the intended signal and
retain the null control. It is still synthetic evidence and does not authorize
real-data or investable-performance claims.
