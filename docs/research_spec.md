# Research Specification

## Status and versioning

This is the pre-analysis specification for **Business Niche Hypercube: Corporate Niche Dynamics and Asset Prices**. P1 added synthetic raw inputs, P2 added point-in-time panels, P3 added accounting components and axes, P4 added synthetic viability models, P5 added synthetic OOS dynamics, P6 added synthetic competing-exit survival models, and P7 added synthetic point-in-time return tests. None is a real-data market result. Hypotheses are confirmatory candidates and must not be changed after real outcomes are inspected unless the revision is versioned and labeled exploratory.

## Mission

Estimate an evolving continuous viability surface in a six-dimensional, point-in-time firm state space. Measure firm position and movement relative to that surface, then separately evaluate business viability, valuation, and tradable alpha. The 729-cell grid is an interpretability layer, not the primary model.

## Research questions

1. **Viability surface:** Does the six-dimensional state predict three- and five-year survival better than standard distress, profitability, industry-year survival, and occupied-cell baselines?
2. **Niche dynamics:** Do viability level, velocity, acceleration, crowding, and frontier margin predict operating deterioration or performance failure?
3. **Asset pricing:** Does unexpected niche migration predict future returns after standard characteristic and industry controls?
4. **Implementability:** Does any return signal survive conservative lags, delisting returns, costs, turnover, borrow rules, and capacity constraints?
5. **Economic interpretation:** Which measured archetypes and transitions are stable or dangerous, and how does the historically anchored viable state evolve?

## Predeclared hypotheses

- **H1:** A continuous viability model outperforms simple cell occupancy, Altman-style distress variables, and static profitability measures in predicting three- and five-year survival.
- **H2:** Negative niche velocity and acceleration predict future operating deterioration and performance delisting.
- **H3:** Static viability level has weak or ambiguous return predictability because strong firms may already be expensive.
- **H4:** Residualized niche migration surprise has stronger return predictability than static niche level.
- **H5:** Return predictability, if present, is stronger near the viability frontier.
- **H6:** Gross returns materially overstate implementable returns; a credible strategy must remain positive under conservative execution assumptions.

## Primary outcomes and tests

- Primary viability outcomes: performance-related failure within three years and five years.
- Separate competing outcome: merger or acquisition.
- Primary viability comparison: calibrated continuous model versus static profitability, leverage/size, constructible Altman-style variables, naive industry-year survival, and occupied-cell density.
- Primary alpha candidate: strictly out-of-sample residualized niche migration surprise.
- Primary return horizon: six months after the first eligible formation month; 1-, 3-, and 12-month horizons are secondary.
- Primary portfolio evidence: factor-adjusted, value-weighted long-short spread net of the predeclared conservative cost scenario.
- Secondary results require multiplicity control and may not replace a failed primary result.

The exact real-data performance-failure code mapping and cost parameters remain unresolved. P2 froze the baseline universe thresholds before constructing the synthetic panels: $1 absolute price, $10 million market capitalization, and monthly volume of 100 in source units. P5 froze the viability frontier at 95% calibrated horizon survival and measures its margin in survival log odds. Remaining decisions must be frozen before the phase that uses them and before inspecting the corresponding outcomes.

## Anti-overfitting and evidence rules

1. Use expanding or rolling time splits; never use a random primary split.
2. Fit transformations, scalers, clusters, residualization, calibration, and hyperparameters on training information only.
3. Keep a simple transparent benchmark beside every sophisticated model.
4. Do not select specifications, samples, horizons, costs, or signs to improve Sharpe or significance.
5. Separate confirmatory primary tests from exploratory secondary tests.
6. Preserve null and negative findings.
7. Test the null-alpha synthetic scenario before interpreting injected-alpha recovery.
8. Do not promote a real-data result unless point-in-time, delisting, cost, and audit gates pass.
9. LLMs may draft prose only from saved numerical artifacts; they may not compute observations or statistics.
10. Archive or version completed real-data results rather than silently overwriting them.

## Phase boundary

P0 created the scaffold, policies, configs, and smoke gates. P1 added only
scenario-driven synthetic raw parquets, truth metadata, and raw-input
validation. P2 added point-in-time availability, universe, link, delisting, and
staleness transformations. P3 added outcome-blind components and axes. P4
added fixed-horizon synthetic labels and calibrated OOS viability models. P5
added only frontier/dynamics features and prior-year migration expectations.
P6 added delayed-entry, time-varying cause-specific models for performance
failure and merger, with other exits censored explicitly. Return tests,
portfolios, clustering, and visualization remained unauthorized through P6.
P7 adds gross Fama-MacBeth and portfolio tests with factor, exposure, turnover,
subsample, monotonicity, delisting, and attrition diagnostics. Costs, borrow,
capacity, net promotion, clustering, and visualization remain unauthorized
through P7.
