# Research Report

## Final status

The synthetic P0-P10 research pipeline is complete. Its engineering and
point-in-time gates pass, but the predeclared scientific alpha-recovery test
fails. No real-data run occurred, and this report contains no claim about
actual markets or investable performance.

## Main result

The continuous state representation did not broadly beat simple profitability
and distress baselines, niche velocity/acceleration signs were not stable, and
the observable migration-surprise signal did not recover the injected return
effect. In the migration-alpha scenario, the six-month rank IC was
-0.0187; the controlled coefficient was
0.0047 with p =
0.289 and Holm p =
1.000. The hidden oracle component was
recoverable in P7, so the failure lies in the observable feature rather than
the return-path timing.

The null-alpha control behaved correctly: IC -0.0084,
coefficient 0.0011, p =
0.761. This prevents a false positive but does not
rescue the failed injected-alpha recovery.

## Implementability

Under the frozen conservative execution proxy, the primary migration-alpha
portfolio earned 1.22%
annualized after capacity but before costs and
-6.61% after spread, slippage, and
borrow assumptions. Net Sharpe was -1.23,
maximum drawdown -56.47%, average
monthly turnover 0.46, and modeled capacity
fill 0.67. The null and
regime-shift scenarios were also negative after costs
(-6.27% and
-7.00%).

This supports only the modest H6 statement that gross results materially
overstate implementation. It does not support a trade.

## Archetypes

Training-only density clustering was not stable. Noise/unassigned rates were
94.68% in null alpha,
98.05% in migration alpha, and
95.44% in regime shift. Mean
stability ARI was approximately zero where calculable; every bounded
regime-shift stability refit failed to recover two clusters. Neutral labels
are retained for audit, but they should not be interpreted as robust business
archetypes.

## Hypothesis scorecard

- H1: not broadly recovered; combined axes did not consistently beat simple
  profitability and distress baselines.
- H2: not broadly recovered; velocity and acceleration signs varied.
- H3: consistent with weak/ambiguous static return predictability, but the
  synthetic exercise is not market evidence.
- H4: not supported; migration surprise failed injected-alpha recovery.
- H5: not promoted because the primary return signal failed.
- H6: gross materially exceeded net, but net performance was negative.

## Evidence boundary

All accounting transformations use SEC filing time, RDQ, or a conservative
180-day fallback; all formation dates are strictly post-availability. CCM,
duplicates, staleness, delistings, overlapping horizons, costs, lagged
liquidity, hashes, and row counts are independently audited.

The 729-cell cube is used only as a secondary visualization. P9 clustering is
descriptive. All figures trace to saved tables listed in
`artifacts/tables/p10_figure_registry.csv`.

## Conclusion

The repository is a reproducible negative research result: it validates the
infrastructure, rejects the current observable migration-alpha construction,
and documents what would need redesign before any real-data confirmatory
study. Replacing synthetic parquets with licensed WRDS-shaped inputs is
structurally supported, but no real-data study should be run under the failed
alpha-recovery specification without a versioned exploratory redesign.
