# Modeling Decisions

## Decisions fixed in P0

- The primary representation is a continuous calibrated viability surface.
- The 729-cell grid is secondary and descriptive.
- Business viability, valuation, and alpha are distinct empirical questions.
- All six axis scores use the convention **higher means more viable or defensible**. The competitive axis is therefore named Competitive Defensibility even when its raw inputs measure pressure.
- Relative and historically anchored representations must coexist.
- The primary alpha candidate is residualized niche migration surprise; static viability is a benchmark.
- The primary return horizon is six months; other declared horizons are secondary.
- The core baseline cannot depend on scikit-survival.
- Dependencies are staged by phase. P0 installs only configuration and testing dependencies; numerical packages are added and pinned when first required.

## Deferred decisions

These are unresolved, not silently assumed:

- alternative real-data industry definitions beyond the frozen SIC2/SIC1 fallback;
- exact performance-failure and merger code mappings;
- alternative real-data fiscal revision hierarchy if rejection proves unusable;
- stricter P3 rolling-history requirements and later capacity thresholds;
- Altman-style benchmark definition given available fields;
- any change to the frozen P3 missing-R&D treatment after real-data inspection;
- conservative spread, borrow, capacity, and delayed-execution parameters;
- exact multiplicity and deflated-Sharpe procedures;
- verified literature claims and novelty assessment.

Each decision must be frozen before the phase that uses it and before inspecting the relevant outcomes.

## P0 non-decisions

No formula, feature, label, model, hyperparameter, portfolio, cluster, or result is implemented in P0.

## Decisions fixed in P1

- The generator version is `hypercube-synthetic-v1` and every run is seeded.
- Synthetic scenarios use identical public raw schemas and differ only through
  disclosed data-generating-process parameters.
- All scenarios inject a negative relationship between true viability and the
  hazard of performance failure.
- Only `migration_alpha` injects a return effect from migration surprise.
- Only `regime_shift` changes the viability weights at a known date.
- Synthetic truth is stored separately and is never required by real-data code.
- P1 performs no feature construction, label construction, estimator fitting,
  statistical inference, portfolio formation, or parameter recovery test.

## Decisions fixed in P2

- Monthly formation is the first calendar month-end strictly after the selected public timestamp; same-day execution is prohibited.
- Fiscal duplicates and equal-priority CCM mappings to different firms are rejected instead of guessed away.
- CCM allowlists are LC/LU/LS and P/C, prioritized P before C and LC before LU before LS.
- Baseline filters are common shares 10/11, exchanges 1/2/3, absolute price at least $1, market capitalization at least $10 million, and monthly volume at least 100 in source units.
- A prior-eligible delisting-month override prevents a final distressed month from disappearing solely because that month fails a filter.
- Returns compound ordinary and delisting components. Missing delisting returns remain explicitly missing and are not assigned a performance-based replacement.
- Fundamentals expire 18 calendar months after formation. Minimum reporting history is one report in P2; P3 components may require more.
- The synthetic truth table is excluded from P2 construction and validation.
- P2 builds no axes, labels, fitted estimators, return tests, or portfolios.

Real-source exit-code mappings and any alternative real-data duplicate/revision policy remain deferred until a real extract is inspected, and must be frozen before P6.

## Decisions fixed in P3

- The feature unit is one accounting update when it first reaches the eligible P2 universe, not every monthly repetition of that update.
- Raw components use only current and earlier firm releases. Consecutive-history components are unavailable across fiscal-year gaps.
- Relative scores use the current feature-month peer snapshot and SIC2, SIC1, then market fallbacks with at least 20 observations.
- Historically anchored scores use expanding observations from strictly prior calendar years, with at least 100 observations. They are not annually recentered using future observations.
- Both representations use 1%/99% reference winsorization, median/MAD robust scores, a standard-deviation fallback for degenerate MAD, and a plus/minus-five clip.
- Missing R&D is zero with a separate indicator in the baseline. Observed-only and contemporaneous peer-median variants are retained for sensitivity. Capitalized R&D stock depreciates at 20% per year.
- NOPAT uses a fixed 25% tax proxy; this is documented as a limitation rather than estimated from synthetic outcomes.
- Accounting-industry competition uses current SIC2 sales HHI and active-firm count. The baseline treats concentration as defensibility and stores a reversed-HHI sensitivity.
- The cash-conversion proxy remains a diagnostic/ablation component but is excluded from the baseline unit-economics average after outcome-blind diagnostics showed approximately 0.98 correlation with ROIC and VIFs above 30. The remaining baseline retains operating margin, ROIC, and gross profitability; residual collinearity is reported rather than hidden.
- Advertising is never zero-imputed. Minimum component-coverage rules allow the go-to-market axis to remain available when advertising is unreported.
- P3 fits no viability, survival, return, or portfolio model and never reads synthetic truth.

## Decisions fixed in P4

- Three- and five-year performance-failure probabilities use four purged,
  expanding outer calendar folds. Training labels must be observable before
  validation, and validation labels before the outer test window.
- The primary continuous surface is the calibrated `combined_axes_logit`.
  Profitability, distress, industry-rate, occupied-cell, relative-axis, and one
  constrained histogram-gradient model remain frozen comparisons.
- Ridge strength is selected by inner-validation Brier score, and Platt
  calibration is fitted only on inner validation. The nonlinear model has no
  search in P4.
- P4 reads no synthetic truth and performs no return or portfolio test.

## Decisions fixed in P5

- P5 uses only out-of-sample probabilities from P4's frozen
  `combined_axes_logit`; it does not select a better model after seeing P4
  outcomes. Each frozen fold artifact is applied to every eligible event in its
  outer test years, including censored and competing-exit rows. P4's
  observed-label prediction table is used only as an exact reproduction check,
  never as the P5 population filter.
- The fixed frontier is 95% calibrated horizon survival for both horizons.
  Frontier margin is measured in survival log odds, not claimed as Euclidean
  distance. A constant annualized hazard proxy is stored as
  `-log(survival_probability) / horizon_years` and is not an instantaneous
  hazard estimate.
- Level is calibrated survival probability and survival log odds. Velocity is
  annualized change in survival log odds; acceleration is annualized change in
  velocity. Model-refit boundaries are not treated as firm movement: level
  changes, acceleration, and frontier crossings are missing when consecutive
  observations come from different outer folds.
- Cross-sectional percentile uses only firms sharing the exact feature date.
  Crowding uses contemporaneous six-axis neighbors, preferring SIC2 and falling
  back to the full dated cohort. Historical-success density uses only prior
  events whose nonfailure outcome was observable before the current calendar
  year.
- Migration surprise is current log-odds velocity minus an expanding ridge
  expectation trained only on earlier calendar years. The frozen controls are
  lagged viability level, size, book-to-market, 12-to-2 momentum, prior-month
  return, profitability, investment, leverage/distress variables, trailing
  liquidity and volatility, market beta, and dated SIC2. Alpha is fixed at
  10.0; there is no outcome-driven search.
- Synthetic truth is forbidden in feature construction. Independent P5
  validation may read it only to test three predeclared directional recovery
  gates at the five-year horizon: level Spearman at least 0.10, velocity at
  least 0.05, and residualized migration surprise at least 0.03. Return alpha
  and injected return fields remain unread until the return phase.
- P5 does not cluster archetypes (P9), estimate competing-risk survival models
  (P6), or run return and portfolio tests (P7-P8).

## Decisions fixed in P6

- P6 uses the five-year P5 OOS surface only once per accounting event and joins
  the P2 reporting-history count. A minimum of three public reports is required.
- Survival data use nonoverlapping start/stop intervals. Covariates update at
  each feature date; an interval ends at the next feature, the dated exit, or
  last raw security observation, whichever occurs first. Absolute calendar
  entry and stop times represent delayed entry explicitly.
- Performance failure and merger are separate cause-specific outcomes. For
  either cause, every other dated exit is censoring at its actual exit date.
  Voluntary/administrative and other/unknown exits are never relabeled as
  failure.
- The frozen feature order is viability log odds, velocity, acceleration,
  crowding, historical-success density, size, leverage, and operating margin.
  Missing values are median-imputed and standardized using training data only.
- The primary estimator is an unpenalized Breslow-ties proportional-hazards
  regression with issuer-clustered uncertainty. No feature or penalty search is
  permitted. Schoenfeld-residual/time associations are diagnostics, not a gate
  selected to improve results.
- Time-split tests are 2010-2014 and 2015-2019. Training intervals and outcomes
  must end before each test window begins. A train-fitted logistic map from Cox
  risk score to interval event probability supplies AUC, Brier, average
  precision, and calibration diagnostics; Harrell-style concordance uses dated
  interval risk sets.
- P6 estimates no causal effect. It performs no return test, portfolio,
  transaction-cost analysis, clustering, or synthetic return-alpha recovery.

## Decisions fixed in P7

- P7 uses only the five-year P5 OOS surface. The primary signal is
  `migration_surprise`; static viability log odds and raw log-odds velocity are
  frozen benchmarks.
- The signal date is the first P5 feature month in which the accounting update
  is both public and attached to an eligible security; it may be later than the
  accounting-only formation month but never earlier. A signal formed at that
  month-end may first earn a return in the following calendar month. Forward
  horizons are 1, 3, 6, and 12 months, with six months primary. All ordinary
  and available delisting returns are compounded.
- A target requires a contiguous dated return path through its horizon or a
  valid delisting event. After a valid delisting, proceeds earn the observed
  risk-free rate through the remaining horizon. A missing delisting return
  makes the target missing; it is never replaced with zero or an assumed loss.
- The primary cross-sectional outcome is the forward return in excess of
  compounded risk-free returns. Fama-MacBeth regressions include size, value,
  momentum, profitability, investment, leverage, four distress variables,
  liquidity, volatility, beta, and dated SIC1 controls. Predictors are
  contemporaneously winsorized and standardized without using outcomes.
- Overlapping-horizon coefficient inference uses Newey-West lag
  `horizon - 1`. The primary six-month migration-surprise test is predeclared;
  Holm adjustment is reported across the twelve signal/horizon tests.
- Gross portfolio sorts use five quantiles and report equal/value weighting
  with and without dated SIC1 neutrality. Monthly returns aggregate overlapping
  formation cohorts; factor adjustment uses market, size, value,
  profitability, investment, and momentum factors. Turnover, exposures,
  drawdown, monotonicity, fold/subsample results, and attrition are reported.
- P7 applies no costs, borrow assumptions, capacity rules, delayed-execution
  sensitivity, or net-performance promotion. Those belong only to P8.
- Synthetic construction never reads truth. Independent validation may read
  the truth sidecar only for predeclared null-alpha, migration-sign, and oracle
  injected-magnitude recovery checks after all P7 outputs are frozen.
# P8 — Cost-aware portfolio freeze

The P8 assumptions were frozen before reading P8 net results. The primary
portfolio remains the predeclared six-month, value-weighted,
industry-neutral migration-surprise spread. P8 does not search for a new
signal or rescue P7's failed scientific recovery gate.

- Quoted half-spread is the primary spread proxy; missing/invalid quotes use
  a 50 bps half-spread fallback.
- Low, conservative, and severe one-way cases use 0.5/1.0/1.5 times the
  quoted half-spread plus 5/10/25 bps fixed slippage.
- Capacity is the minimum of 10% of dated ADV, 0.1% of dated market cap, and
  10% portfolio weight, using a $1 million notional per leg.
- Shorts require at least $25 million market cap and $0.10 million ADV.
  Borrow is 300 bps annually, or 700 bps below $100 million market cap or
  $0.50 million ADV.
- Execution inputs for a return month are lagged at least one month.
- The delayed-execution sensitivity adds one full month; it is not selected
  after observing performance.
- No-trade/excluded capacity stays excluded. Delisting returns already
  embedded in P7 are retained. Forced delisting exits are not charged a
  fabricated closing trade.

These are synthetic execution proxies. They are not evidence that a live
order could have been filled at the modeled price or capacity.

# P9 — Descriptive archetype freeze

P9 uses the six historically anchored axes. Robust location and scale are
estimated from observations through 2004 only. A deterministic maximum of
25,000 training rows is fitted with HDBSCAN (`min_cluster_size=200`,
`min_samples=10`). Later rows are mapped only to a training centroid and only
inside that cluster's frozen 95th-percentile training radius; all other rows
remain `Noise / Unassigned`.

The initial infrastructure canary using `min_cluster_size=300` and
`min_samples=30` returned all noise. Before any profile, survival, or return
characteristic was opened, the density requirement was relaxed once to the
values above and recorded here. High remaining noise is a valid result and
will not be tuned away. Labels remain neutral (`Archetype A`, etc.); outcome
tables are descriptive and cannot alter P7 or P8.
