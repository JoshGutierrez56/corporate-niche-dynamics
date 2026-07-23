# Data Dictionary

## P2 status

P1 generated synthetic raw parquets with the same file names and required columns
expected from later WRDS-style extracts. `hypercube.data.TABLE_SPECS` is the
machine-enforced source of truth for required columns, primary keys, and date
fields. P2 consumes those files without changing their schemas.

## Required raw files

| File | Primary key | Purpose |
|---|---|---|
| funda.parquet | gvkey, datadate, fyear | Annual Compustat-style fundamentals and reporting dates |
| crsp_monthly.parquet | permno, date | Monthly returns, prices, shares, volume, spread inputs, and security codes |
| crsp_delist.parquet | permno, dlstdt | Delisting returns, codes, and exit reasons |
| ccm_link.parquet | link-history rows | Point-in-time GVKEY-PERMNO mapping |
| factor_returns.parquet | date | Market, SMB, HML, RMW, CMA, momentum, and risk-free returns |

All required keys must be non-null and unique. CRSP dates must be month-end,
factor returns must cover every CRSP month, CRSP and delisting PERMNOs must be
present in CCM, and primary CCM links for a GVKEY may not overlap.

## Optional raw files

- sec_filing_dates.parquet: verified public filing timestamps.
- Product-market text measures, analyst variables, ownership, credit measures, and filing embeddings through modular adapters only.

Synthetic runs additionally write two explicitly non-real sidecars:

- `synthetic_truth.parquet`: latent axes, true viability, migration surprise,
  availability date/source, injected return contribution, regime marker, and
  exit category. Downstream real-data code may not require this file.
- `synthetic_scenario_metadata.json`: seed, injected-effect contract,
  missingness, exit counts, and a declaration that no model was fitted.

## Required derived availability fields

| Field | Meaning |
|---|---|
| availability_date | Earliest conservative public availability date used by the pipeline |
| availability_source | SEC timestamp, earnings announcement, or 180-day fallback |
| availability_confidence | Source-quality category defined before panel construction |

## P2 processed files

| File | Key | Purpose |
|---|---|---|
| accounting_availability.parquet | gvkey, datadate, fyear | Raw fundamentals plus selected public date, source, confidence, strict formation date, invalid-date flags, and reporting-history count |
| universe_monthly.parquet | permno, date | CRSP month with market cap, universe eligibility, delisting metadata, and total return after the frozen delisting rule |
| firm_month_panel.parquet | permno, date | Point-in-time CCM-linked security month joined to the latest eligible, non-stale accounting observation |
| row_count_waterfall.csv | stage | Cumulative observations retained and removed at every universe, link, availability, history, and staleness gate |
| p2_diagnostics.json | one run | Availability-source counts, link diagnostics, delisting reconciliation, timing violations, duplicate counts, and output rows |
| resolved_config.json | one run | Exact frozen configuration used for the bundle |
| p2_manifest.json | one run | Output sizes, SHA-256 hashes, row counts, seed, paths, and timestamp |

Additional P2 columns include `formation_date`, `reporting_history`,
`staleness_months`, `market_cap_millions`, `ret_total`, `has_delist_event`,
`delist_return_missing`, `universe_eligible`, and `delist_month_override`.

## Boundary

Real and synthetic parquets continue to share the same validated raw contracts.
The synthetic truth sidecar is not read by P2 panel construction.

## P3 processed files

Each scenario writes the following below `data/processed/synthetic/<scenario>/p3/`:

| File | Key | Purpose |
|---|---|---|
| component_features.parquet | gvkey, datadate, fyear | Raw accounting components, relative scores, anchored scores, peer levels/counts, anchor counts, and R&D missing-data variants |
| axis_scores.parquet | gvkey, datadate, fyear | Six relative axes, six anchored axes, component coverage, and the predeclared HHI/R&D sensitivities |
| component_missingness.csv | section, variable | Missing and nonmissing counts and rates |
| component_correlations.csv | section, left, right | Pairwise Spearman component correlations and sample counts |
| axis_correlations.csv | section, left, right | Relative and anchored axis correlations |
| component_vif.csv | section, variable | Descriptive VIFs for the unit-economics baseline components |
| ablation_catalog.csv | level, variable | Component, axis, and sensitivity ablation flags |
| transformation_metadata.json | one run | Exact formulas, required inputs, signs, limitations, standardization rules, and truth-exclusion declaration |
| p3_diagnostics.json | one run | Population reconciliation, fallback counts, R&D treatment counts, timing, duplicate, and non-finite gates |
| p3_manifest.json | one run | P2 input hashes and P3 output hashes, sizes, rows, seed, and timestamp |

The machine-readable formula catalog in `hypercube.axes.COMPONENT_SPECS` is the source of truth for 19 declared components. Competitive HHI and active-firm count are constructed from the current investable peer snapshot; they are not available in the annual accounting input alone.

## P4 processed files

Each scenario writes the following below
`data/processed/synthetic/<scenario>/p4/`:

| File | Key | Purpose |
|---|---|---|
| viability_labels.parquet | gvkey, datadate, fyear, horizon_years | Point-in-time three/five-year failure labels with censoring and observation dates |
| model_matrix.parquet | label key | Frozen axes, transparent benchmark controls, labels, and fold inputs |
| oos_predictions.parquet | label key, fold, model | Calibrated outer-test failure and survival probabilities for all seven declared models |
| fold_metrics.csv | horizon, fold, model | Discrimination, calibration, Brier, log-loss, and test-sample metrics |
| hyperparameter_trials.csv | horizon, fold, model, candidate | Inner-validation ridge trials and fixed-model receipts |
| benchmark_comparisons.csv | horizon, fold, benchmark | Primary combined-axis comparison with each simple benchmark |

## P5 processed files

Each scenario writes the following below
`data/processed/synthetic/<scenario>/p5/`:

| File | Key | Purpose |
|---|---|---|
| frontier_dynamics.parquet | gvkey, datadate, fyear, horizon_years | OOS level for every outer-test event (including censored/competing exits), log-odds margin, percentile, hazard proxy, velocity, acceleration, crossings, crowding, historical-success density, encroachment, controls, and residualized migration surprise |
| migration_model_diagnostics.csv | horizon, prediction_year | Expanding ridge training cutoff/count, prediction count, coefficient norm, and exact model artifact |
| models/*.joblib | horizon, prediction_year | Pinned prior-year migration-expectation pipeline and provenance |
| feature_metadata.json | one run | Frontier, dynamic-scale, control, density, truth-exclusion, and downstream-boundary contracts |
| p5_diagnostics.json | one run | Coverage, crossing, model-boundary, and row-count diagnostics |
| p5_manifest.json | one run | P2-P4 input hashes and P5 output hashes, sizes, rows, seed, and timestamp |

`annualized_constant_hazard_proxy` is a constant-rate transformation of a
fixed-horizon survival probability; it is not an estimated instantaneous
hazard. `migration_surprise` is fit only from earlier calendar years. The
synthetic recovery validator reads only `true_viability` and the latent
`migration_surprise`; construction never reads the truth sidecar.

## P6 processed files

Each scenario writes the following below
`data/processed/synthetic/<scenario>/p6/`:

| File | Key | Purpose |
|---|---|---|
| survival_intervals.parquet | interval_id | Delayed-entry, time-varying issuer intervals with mutually exclusive terminal causes and frozen P5/P2 covariates |
| fold_predictions.parquet | interval_id, cause, fold | Outer-test event indicators, entry/stop provenance, risk scores, and calibrated interval probabilities |
| cause_coefficients.csv | cause, feature | Full-sample standardized coefficients, issuer-clustered uncertainty, and hazard ratios |
| fold_metrics.csv | cause, fold | Test-window concordance, AUC, Brier score, calibration, and sample counts |
| ph_diagnostics.csv | cause, feature | Schoenfeld-residual time-correlation diagnostics |
| subgroup_metrics.csv | cause, fold, subgroup | Time-split discrimination by decade and industry group where supported |
| exit_reconciliation.csv | exit_category | Eligible dated exits, assigned terminal intervals, and differences |
| models/*.joblib | cause, fold/full | Frozen preprocessing, fitted proportional-hazards parameters, calibration, and provenance |
| p6_diagnostics.json | one run | Interval, issuer, cause, terminal-reason, model, and no-return/no-causal-claim receipts |
| p6_manifest.json | one run | P2/P4/P5 input hashes and P6 output hashes, sizes, rows, seed, and timestamp |

P6 treats performance failure and merger as separate cause-specific outcomes.
Voluntary/administrative and other/unknown exits are never relabeled as
performance failures. P6 reads no synthetic truth and performs no return test.

## P7 processed files

Each scenario writes the following below
`data/processed/synthetic/<scenario>/p7/`:

| File | Key | Purpose |
|---|---|---|
| return_events.parquet | event_id | Frozen five-year P5 events, signals, controls, feature dates, industries, and fold provenance |
| event_month_paths.parquet | event_id, holding_offset | Twelve strictly post-feature monthly returns, delisting/cash flags, validity, RF, and factors |
| forward_return_targets.parquet | event_id, horizon_months | 1/3/6/12-month raw, risk-free, and excess returns plus target status and attrition |
| fmb_monthly_coefficients.csv | horizon, signal, formation_date, term | Formation-month cross-sectional coefficients, rank IC, sample size, fit, and fold |
| fmb_summary.csv | horizon, signal | Overlap-HAC coefficient/IC inference, Holm adjustment, fold consistency, and primary-test flag |
| portfolio_quantile_returns.csv | strategy, formation_date, quantile | Gross equal/value quintile returns with raw and SIC1-neutral variants |
| portfolio_assignments.parquet | strategy, event_id, leg | Auditable top/bottom gross weights used in monthly overlapping portfolios |
| portfolio_monthly_returns.csv | strategy, holding_date | Gross long, short, spread, cohort count, turnover, and factor returns |
| portfolio_factor_results.csv | strategy | Gross return, Sharpe, drawdown, hit rate, turnover, factor alpha, betas, and explicit no-net-return flag |
| portfolio_exposures.csv | strategy, characteristic | Weighted long-short characteristic exposures |
| portfolio_cohort_spreads.csv | strategy, formation_date | Long-short forward spread and quantile monotonicity |
| subsample_results.csv | strategy, fold | Fold-level gross spread inference and monotonicity |
| target_attrition.csv | horizon, target_status | Complete, delisted, missing-path, and missing-delisting-return counts |

P7 construction reads no synthetic truth. Independent validation may open the
truth sidecar only after outputs are frozen to audit the predeclared null,
migration-sign, and injected-magnitude recovery checks. P7 has no net-return,
cost, borrow, or capacity field.
# P8 cost-aware outputs

- `capacity_assignments.parquet`: frozen P7 assignments with dated spread,
  ADV, borrow eligibility, modeled capacity, fill ratio, and exclusion reason.
- `executed_positions.parquet`: lagged security-month target and actual
  weights, returns, liquidity dates, capacity limits, and forced-exit flags.
- `cost_aware_monthly_returns.csv`: gross capacity return, turnover,
  transaction cost, borrow cost, and net return for every frozen cost case.
- `cost_aware_summary.csv`: gross/net annualized return and Sharpe, drawdown,
  hit rate, turnover, costs, fill ratio, and factor-alpha diagnostics.
- `capacity_diagnostics.csv`, `cost_waterfall.csv`, and
  `delayed_execution_sensitivity.csv`: predeclared implementation audits.

# P9 descriptive archetype outputs

- `archetype_assignments.parquet`: neutral training/OOS archetype label,
  assignment source, confidence, distance, and noise status for each P3 event.
- `training_cluster_centroids.csv`: robust-scaled training centroid, radius,
  and member count for each density cluster.
- `cluster_sizes.csv` and `archetype_profiles.csv`: membership, axis profiles,
  and clearly descriptive survival/return characteristics.
- `transition_matrix.csv` and `cluster_persistence.csv`: consecutive-firm
  transitions and row-normalized probabilities.
- `cluster_stability.csv`: bounded refit cluster counts, noise, and
  adjusted-Rand agreement. Failed refits remain explicit.
