# P21E Measurement-Utility Results

## Final verdict

`RESEARCH_BENCHMARK_ONLY`

The Hypercube supports one new synthetic measurement use:
`competitive_crowding_state`. Strategic-drift alerts and exact structural-peer
retrieval are promising diagnostics, but neither cleared its frozen primary
gate. No threshold was changed after results were opened.

P21E fit no model and read no returns, injected return alpha, survival labels,
exit categories, true viability, real data, portfolios, or EDGAR embeddings.
Synthetic truth access was limited to the six latent axes and migration state.

## Competitive-crowding state: supported

The anchored-axis local-density measure recovered the latent crowding state:

- pooled latent-density Spearman: `0.3215`;
- null-alpha Spearman: `0.3349`;
- migration-alpha Spearman: `0.3384`;
- regime-shift Spearman: `0.2939`; and
- complete-case coverage: `93.33%`.

All frozen gates passed: pooled correlation exceeded `0.20`, every scenario
exceeded `0.10`, and coverage exceeded `85%`.

This supports a synthetic competitive-density monitor: a firm can be located
in a relatively sparse or crowded part of the six-dimensional business-model
space. It does not establish that crowding predicts returns, failure, or a
specific competitive event.

## Strategic-drift alerts: not supported

The locked `anchored_axis_innovation` measure remained directionally useful:

- extreme-event precision: `0.2421`;
- extreme-event recall: `0.2339`;
- sign accuracy among alerts: `0.8585`;
- coverage: `96.63%`;
- existing-benchmark precision: `0.1864`; and
- precision improvement: `+0.0557`.

Precision was stable across 2013-2015 (`0.2420`) and 2016-2018 (`0.2421`).
Four of five frozen gates passed. The use case failed because overall
precision was below the preregistered `0.25` threshold by `0.0079`.

The score may be used as an exploratory change-ranking diagnostic, but not as
a validated extreme-drift alert.

## Structural peer discovery: not supported

The anchored-axis peer map recovered:

- Recall@20: `0.04838`;
- analytical random recall: `0.01145`;
- lift over random: `4.23x`;
- scenario Recall@20 range: `0.04720` to `0.04910`; and
- complete-case coverage: `93.33%`.

Two of four frozen gates passed. It missed the `0.05` pooled-recall threshold
by `0.00162` and the `5x` random-lift threshold by `0.77x`.

Observed peers are materially better than random in the synthetic data, but
the result is not strong enough to validate comparable-company selection.

## Relation to prior phases

- P5 already showed that level, velocity, and migration measurements recover
  their synthetic states.
- P14F showed strong rank recovery for the locked migration proxy
  (`0.4617` Spearman), but P21E shows that this does not automatically imply
  sufficiently precise top-decile alerts.
- P20E found no general incremental survival utility, while regime-shift
  folds showed exploratory improvement in 7 of 8 comparisons.
- P18E rejected the density-cluster archetype taxonomy. P21E does not revive
  clustering: local crowding is a continuous measurement, not a discrete
  economic archetype.

## Defensible conclusion

The Corporate Niche Dynamics system is best treated as an auditable synthetic
research framework and a candidate competitive-crowding monitor. It is not a
return model, survival model, taxonomy, or validated peer-selection engine.

The next credible research phase would freeze a point-in-time real-data test
of crowding measurement stability before computing any outcome relationship.
That is outside P21E and was not run.
