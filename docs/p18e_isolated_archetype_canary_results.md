# P18E — Isolated Descriptive-Archetype Canary Results

## Result

`PASS` on the preregistered engineering, isolation, and traceability gates.
The substantive descriptive result is weak: the frozen P9 method does not
recover a stable, broadly covering archetype taxonomy in this synthetic
population.

All three scenarios built and independently validated. Every recorded input
and output hash recomputed, all fit rows obeyed the 2004 cutoff and 25,000-row
cap, no truth sidecar was read, no return model was refit, no P10 output was
created, and the main manifests remained byte-identical. The full suite passed
83 tests.

## Coverage and stability

| Scenario | Fit rows | Training clusters | Overall noise | OOS assigned | Stability refits | Mean ARI |
|---|---:|---:|---:|---:|---:|---:|
| null alpha | 25,000 | 2 | 94.68% | 6.66% | 4 / 5 | -0.0016 |
| migration alpha | 24,275 | 2 | 96.50% | 3.90% | 4 / 5 | -0.0005 |
| regime shift | 23,892 | 2 | 95.44% | 5.02% | 0 / 5 | undefined |

The two bounded-refit ARIs are effectively zero, while every regime-shift
refit failed to recover two clusters. Noise/unassigned status is highly
persistent out of sample (`95.45%` to `97.59%`). Substantive label persistence
is low: `10.36%` to `16.52%` for Archetype A and only `0.70%` to `2.49%` for
Archetype B.

The preregistration explicitly allowed high noise and weak stability without
tuning. These diagnostics therefore do not invalidate the build, but they do
preclude treating the labels as a robust firm taxonomy.

## Descriptive profiles

Across scenarios, Archetype A contains 1,217 to 2,037 out-of-sample
observations. Archetype B is much smaller, with 134 to 284 observations. The
main geometric distinction is innovation intensity:

- Archetype A mean anchored innovation: `-0.03` to `0.08`
- Archetype B mean anchored innovation: `-1.60` to `-1.51`

Archetype B also has somewhat weaker competitive defensibility, while the
other four axis means remain comparatively close to the anchored center. The
neutral labels were retained; no outcome-driven renaming was performed.

Five-year failure rates and mean six-month excess returns are descriptive
only:

| Scenario | Group | Failure rate | Mean six-month excess return |
|---|---|---:|---:|
| null alpha | A | 3.37% | 4.60% |
| null alpha | B | 4.41% | 1.68% |
| null alpha | Noise | 4.25% | 4.02% |
| migration alpha | A | 4.43% | 5.04% |
| migration alpha | B | 6.45% | 7.05% |
| migration alpha | Noise | 4.51% | 3.88% |
| regime shift | A | 4.56% | 5.27% |
| regime shift | B | 4.46% | 2.30% |
| regime shift | Noise | 4.08% | 4.47% |

The outcome ordering is not consistent across scenarios, and the small
Archetype B samples make the raw differences especially fragile. No
performance test, significance claim, causal interpretation, or portfolio
selection was applied.

## Interpretation

P18E validates the P9 machinery but does not validate the resulting clusters
as stable economic archetypes. The frozen density rule identifies a narrow
low-innovation pocket and assigns almost everything else to noise. That is a
useful negative result: a polished P10 taxonomy would overstate what the
current clustering evidence supports.

P18E stops before P10. It makes no real-data, causal, investable-performance,
or production claim.
