# P21E Measurement-Utility and Product-Closeout Protocol

## Purpose

P21E asks what the Corporate Niche Dynamics system can support after the
return implementation, general survival extension, and archetype taxonomy
failed their primary gates. It evaluates the Hypercube as a measurement and
monitoring system rather than fitting another outcome model.

P21E is a retrospective extension, not a blind preregistration. The P5, P14F,
and P20E results were known before this protocol was written. The peer-map,
crowding-state, and extreme-drift retrieval metrics below had not been
computed.

## Permitted inputs

P21E may read:

- the frozen P3 axis scores and P5 frontier-dynamics surfaces for all three
  synthetic scenarios;
- the frozen synthetic latent axes and migration state;
- the frozen P13F proxy candidates;
- the signed P5, P14F, and P20E summary artifacts; and
- the existing source code, configuration, and validation receipts.

Synthetic truth access is limited to `latent_axis_1` through
`latent_axis_6` and `migration_surprise`.

P21E may not read or use:

- forward or contemporaneous returns;
- injected return alpha;
- survival or failure labels;
- exit categories;
- true viability;
- portfolios, costs, or trading results;
- real data; or
- the separate EDGAR embedding corpus.

No model is fitted and no Hypercube axis or proxy is redesigned in P21E.

## Frozen evaluation window

- Evaluation years: 2013-2018.
- Stability blocks: 2013-2015 and 2016-2018.
- Scenarios: null alpha, migration alpha, and regime shift.
- Event key: `gvkey`, `datadate`, `fyear`.

## Use case 1: strategic-drift alerts

The locked `anchored_axis_innovation` is reconstructed for each scenario with
the exact frozen P13F constructor and compared with the existing P5
`migration_surprise` benchmark. This construction is outcome-blind and may
not be modified by P21E.

Within each scenario-year:

1. rank firms by the absolute value of each observable score;
2. flag the largest 10 percent, using descending absolute score and ascending
   `gvkey` as the deterministic tie break;
3. independently flag the largest 10 percent of absolute synthetic migration
   truth; and
4. measure extreme-event precision/recall and sign accuracy.

The strategic-drift use case is `SUPPORTED` only if the locked candidate:

1. has overall extreme-event precision of at least `0.25`;
2. has precision of at least `0.20` in both stability blocks;
3. has sign accuracy of at least `0.75` among issued alerts;
4. covers at least `0.95` of comparable benchmark events; and
5. exceeds benchmark precision by at least `0.05`.

## Use case 2: structural peer discovery

For every scenario-year, firms complete on all twelve observed axes and all
six latent axes are retained. Anchored, relative, and latent vectors are each
standardized cross-sectionally with their own mean and population standard
deviation. Zero-variance dimensions are assigned scale one.

For each firm, the 20 nearest other firms in Euclidean latent space define the
truth peer set. The 20 nearest firms in anchored or relative observed space
define candidate peer sets. Recall@20 is the share of latent peers recovered.
The analytical random recall is `20 / (n - 1)` in a group of size `n`.

The anchored peer map is `SUPPORTED` only if:

1. pooled evaluation Recall@20 is at least `0.05`;
2. Recall@20 is at least `0.03` in every scenario;
3. pooled recall is at least five times analytical random recall; and
4. complete-case coverage is at least `0.85` of P3 evaluation rows.

Relative-axis peer metrics are a frozen secondary comparison and cannot
replace the anchored primary map.

## Use case 3: competitive-crowding state

For each firm-year, latent crowding is the negative mean distance to its 20
nearest latent peers. Observed crowding is defined identically in anchored or
relative observed space. Higher values mean a denser local neighborhood.

The anchored crowding state is `SUPPORTED` only if:

1. pooled row-level Spearman correlation with latent crowding is at least
   `0.20`;
2. correlation is at least `0.10` in every scenario; and
3. complete-case coverage is at least `0.85`.

Relative-axis crowding metrics are secondary.

## Overall product gate

`MEASUREMENT_MONITOR_SUPPORTED` requires:

- the strategic-drift use case to pass; and
- at least one of structural peer discovery or competitive-crowding state to
  pass.

Otherwise the verdict is `RESEARCH_BENCHMARK_ONLY`.

Each use case retains its own verdict regardless of the overall gate.

## Product and claim boundary

P21E can establish only whether frozen Hypercube measurements recover known
structure in the existing synthetic data-generating processes. Passing a gate
supports a research monitor, not a prediction product.

The final repository must explicitly distinguish:

- supported synthetic measurement uses;
- exploratory uses requiring a separately frozen real-data evaluation; and
- rejected uses: return prediction, general survival prediction, and stable
  economic archetypes.

P21E cannot establish real-market validity, causality, investment value,
default risk, or production readiness.
