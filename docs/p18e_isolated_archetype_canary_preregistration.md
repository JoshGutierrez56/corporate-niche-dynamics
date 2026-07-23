# P18E — Isolated Descriptive-Archetype Canary

## Phase boundary

P18E is the smallest phase after the validated P17E cost canary. It runs the
existing P9 descriptive-archetype contract on the isolated P16E synthetic
population and stops before P10. It may not change the representation,
training cutoff, row cap, density parameters, assignment radius, stability
design, scenarios, or labels after opening P18E outputs.

The run uses only:

- the isolated P16E P3 anchored-axis panel;
- the isolated P16E P4/P5 state and dynamics outputs;
- the isolated P16E P7 descriptive return outcomes;
- the signed P9 configuration copied into the isolated canary; and
- the existing P9 builder and independent validator.

Synthetic truth sidecars are forbidden. P18E does not read or depend on the
P17E cost results.

## Frozen design

- representation: the six historically anchored axes
- scenarios: `null_alpha`, `migration_alpha`, and `regime_shift`
- training cutoff: observations through fiscal year 2004 only
- deterministic maximum training sample: 25,000 rows
- density estimator: HDBSCAN
- `min_cluster_size`: 200
- `min_samples`: 10
- later-row assignment: nearest frozen training centroid only when inside the
  cluster's frozen 95th-percentile training radius
- otherwise: `Noise / Unassigned`
- stability: five deterministic 80% training-sample refits
- labels: neutral canonical labels (`Archetype A`, etc.)
- output root: the isolated canary only
- no P10 report build, real data, truth access, clustering-parameter search,
  return-model refit, or portfolio/specification selection

All saved survival and return characteristics are descriptive profiles of
clusters frozen without those outcomes. They cannot alter P7, P8, P17E, or the
locked `anchored_axis_innovation`.

## Acceptance gates

`PASS` is an engineering, isolation, and descriptive-traceability result. All
of the following must hold:

1. all three preregistered scenarios build and independently validate;
2. every fit row is dated no later than 2004 and the fit-row cap is respected;
3. every post-training assignment follows the frozen centroid/radius rule;
4. noise rows retain the neutral `Noise / Unassigned` label;
5. all output hashes, assignment keys, transition probabilities, model
   metadata, and stability row counts validate;
6. no truth sidecar or real data is read;
7. no P7/P8 return model or portfolio is refit or selected;
8. cluster counts, noise rates, stability diagnostics, transitions, and
   descriptive outcome profiles are reported for every scenario without
   selecting among them; and
9. the main P0-P10 artifacts and all earlier post-closeout manifests remain
   byte-identical.

Fewer than two training clusters is a valid hold at the existing P9 density
gate. High noise, weak stability, unfavorable returns, or unfavorable survival
profiles are admissible and must be reported without tuning. P18E remains
synthetic descriptive evidence and cannot support a real-data, causal,
investable, or production claim.
