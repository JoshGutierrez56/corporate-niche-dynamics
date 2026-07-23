# Point-in-Time Policy

## Governing rule

No observation, transformation, link, label, model input, or portfolio weight may use information before the market could have known it. The implementation must prefer conservative delay to optimistic availability.

## Accounting availability hierarchy

Each accounting observation must store:

- availability_date
- availability_source
- availability_confidence

The source hierarchy is:

1. verified SEC filing timestamp;
2. verified earnings announcement date such as rdq;
3. datadate plus 180 calendar days.

If the source is ambiguous or invalid, the observation is unavailable rather than backfilled from future information.

## Formation mapping

An accounting update becomes eligible only at the first configured monthly formation date strictly after its public availability timestamp. Same-day execution is not assumed. Delayed-execution sensitivity is required in the implementability phase.

## Rolling and cross-sectional transformations

- No centered windows.
- No full-sample scaling, winsorization, imputation, PCA, clustering, calibration, or hyperparameter selection.
- No future-filled values.
- Historical anchors use expanding or trailing distributions ending on or before the observation's availability date.
- Relative scores use only contemporaneously available peers.
- Group fallbacks must be deterministic and logged.

## CCM links

A link is valid only when the formation date falls within linkdt and linkenddt, treating a missing end date as open only under an explicit rule. Link types and primary-link codes must be allowlisted. Duplicate valid links must be resolved by a deterministic, documented hierarchy or excluded and reported.

## Revisions, duplicates, and staleness

- Preserve source period and version fields.
- Resolve duplicate fiscal observations without choosing a record using future knowledge.
- Do not allow a later revision to rewrite an earlier historical feature unless the earlier release is unavailable and the limitation is disclosed.
- Apply a configurable maximum staleness rule.
- Enforce minimum reporting history before rolling features become valid.

## Entry, exit, and delistings

- Universe membership is evaluated using only contemporaneous CRSP-style fields.
- Delisted firms remain in the sample.
- Delisting returns are combined with ordinary returns under a predeclared rule.
- Performance failures, mergers, administrative or voluntary exits, and other exits are mutually exclusive where the source permits.
- Right censoring and delayed entry are explicit.

## Required tests before return analysis

1. No formation row precedes availability_date.
2. Link dates cover every mapped observation.
3. Duplicate links and fiscal rows reconcile.
4. Rolling features match hand-checked expanding-window examples.
5. No future values enter missing-data treatment.
6. Delisting observations and returns reconcile to exit categories.
7. Row-count waterfalls explain every exclusion.

## P2 implementation

P2 implements and independently validates the following frozen rules:

- SEC timestamps are accepted only when non-null and no earlier than datadate.
- RDQ is accepted only when non-null and no earlier than datadate.
- Invalid candidate dates are flagged; the next valid source in the hierarchy is used.
- Monthly formation is the first calendar month-end strictly after the chosen timestamp. A timestamp on month-end therefore waits until the next month-end.
- Fiscal key duplicates are rejected. P2 does not guess which revision was historically visible.
- CCM types LC, LU, and LS and primary codes P and C are allowlisted. Priority is P before C, then LC before LU before LS. An equal-priority mapping to different GVKEYs is rejected.
- Missing CCM end dates are open-ended; non-missing bounds are inclusive.
- Fundamentals may remain active for at most 18 calendar months after formation.
- P2 requires one available report. Components needing longer history must impose their own stricter P3 rule.
- Ordinary and delisting returns are combined as `(1 + ret) * (1 + dlret) - 1`. Missing delisting returns are never imputed; the ordinary return is preserved and the missing delisting return is flagged.
- A delisting month that fails a contemporaneous filter is retained only when the same security passed every baseline universe filter in the immediately preceding month.

Every processed scenario saves the availability table, filtered security-month universe, final firm-month panel, cumulative row-count waterfall, diagnostics, resolved config, hashes, and an independent validation report. No P3 feature or model is constructed in P2.

## P3 feature transformations

- Accounting components are constructed once per fiscal observation using only that release and earlier releases from the same firm.
- A component enters the feature population on the first eligible P2 security month on or after its formation date.
- Relative scores use the contemporaneous eligible firm snapshot at that feature month. The peer hierarchy is SIC2, then SIC1, then the full market, with at least 20 nonmissing peers.
- Relative references are winsorized at the contemporaneous 1st and 99th percentiles and standardized by median/MAD; standard deviation is used only when MAD is degenerate. Scores are clipped to plus or minus five.
- Competitive components are market-standardized because they are industry-level quantities and would be constant inside SIC2.
- Historically anchored scores use an expanding distribution containing only feature events from strictly prior calendar years. The minimum reference count is 100. Current-year and future events never enter the anchor.
- Synthetic truth, future returns, exits, labels, and injected effects are not read during P3 construction.

## P5 frontier and dynamics

- P5 consumes only frozen P4 fold artifacts and applies each one to every event
  in its outer-test years. Eventually censored or competing-exit rows are not
  removed from the dynamics population. Saved P4 observed-label probabilities
  must reproduce exactly for the overlapping rows.
- Viability changes are computed only between consecutive observations from
  the same outer model fold. A model refit is never labeled as firm movement.
- Cross-sectional percentile and crowding reference only the exact feature-date
  cohort. Historical-success density uses prior events whose outcome was
  already observable before the current calendar year.
- The migration expectation for a prediction year is fitted only on velocity
  observations from earlier calendar years. Every fitted row stores its
  training count and maximum training feature date.
- Synthetic truth is excluded from feature construction. It may be opened only
  by the independent synthetic recovery gate, which reads latent viability and
  latent migration surprise but not return injections or realized returns.
- Every saved component score includes its peer/reference count; relative scores also store the fallback level used.

## P6 survival intervals

- Each issuer enters only at its first eligible P5 observation and therefore
  uses explicit delayed entry rather than pretending it was observed earlier.
- A covariate interval begins at an observed feature date and ends at the next
  update, the dated exit, the five-year administrative horizon, or the sample
  end—whichever is first.
- Every interval uses only the P2/P5 state already observable at its start.
  Synthetic truth is not opened by construction or validation.
- Performance failure and merger are modeled as separate causes. A competing
  cause ends the risk interval and is coded as a non-event for the cause being
  estimated; voluntary/administrative and other exits remain distinct censored
  terminal reasons.
- Outer training data end before each declared test window. Preprocessing,
  proportional-hazards fitting, and risk-score calibration are fitted on
  training intervals only.
- P6 saves interval entry and stop times beside every fold prediction so the
  dated risk sets and concordance statistics can be independently reconstructed.

## P7 return timing

- The signal date is the P5 `feature_date`, when the accounting update is
  public, attached to a valid dated security link, and first present in the
  eligible universe. It can be later than `formation_date`, never earlier.
- No return from `feature_date` enters a target. The first eligible holding
  return is the following calendar month-end.
- A non-delisted target must have every dated monthly return through the
  declared horizon. Missing months remain missing and are never converted to
  zero.
- A valid delisting return is compounded with the ordinary return once. The
  proceeds then earn the observed risk-free return through the remaining
  horizon. A missing delisting return makes the target missing.
- Dated CCM validity is reapplied to raw CRSP-style return history. After
  formation, minimum price, size, and volume filters are not reapplied to erase
  a held security's subsequent poor performance.
- Fama-MacBeth transformations use only the contemporaneous formation
  cross-section. Longer-horizon coefficient inference uses HAC lag
  `horizon - 1`; it does not treat overlapping observations as independent.
- P7 portfolios are gross statistical simulations. P8 alone may introduce
  costs, borrow, capacity, or delayed-execution assumptions.
# P8 execution inputs

P8 freezes capacity at formation using only the contemporaneous monthly
price, volume, market capitalization, and quote. Holding-month capacity,
spread, and borrow inputs are lagged one month: `liquidity_date` must be
strictly earlier than `holding_date`. Missing lagged capacity creates no
fill; it is never future-filled. The optional delay test moves entry one
additional complete month and does not reuse the original entry month.
