# Corporate Niche Dynamics: Final Product Positioning

## Repository identity

- Repository: `corporate-niche-dynamics`
- Product label: **Corporate Niche Monitor**
- Suggested description: **An auditable research framework for mapping
  competitive crowding and strategic change in corporate business-model
  space.**

The repository name remains appropriate. The product should not be described
as an alpha model, survival model, or firm taxonomy.

## What the system is

Corporate Niche Monitor turns point-in-time firm information into:

1. six anchored coordinates describing a firm's business-model position;
2. six relative coordinates describing its position against contemporaneous
   industry peers;
3. a continuous local-crowding score;
4. an exploratory strategic-change ranking;
5. an exploratory structural-peer list; and
6. signed provenance and validation receipts.

## Recommended uses

### Supported in the synthetic benchmark

**Competitive-crowding monitoring**

- identify firms in dense versus sparse regions of business-model space;
- compare crowding through time;
- inspect whether multiple firms are converging on the same niche; and
- support competitive-intelligence and market-structure research.

**Auditable methods benchmark**

- test point-in-time feature engineering;
- evaluate leakage controls and temporal validation;
- compare alternative representations under known synthetic truth; and
- reproduce negative as well as positive results from signed artifacts.

### Exploratory; requires a separately frozen real-data test

**Strategic-drift triage**

- rank firms whose axis position changed unexpectedly;
- surface candidates for analyst review of filings, product strategy, or
  business-model transition; and
- track the direction of change across the six axes.

The synthetic score had strong direction accuracy but narrowly missed the
extreme-event precision gate.

**Comparable-company discovery**

- propose firms with similar multidimensional profiles;
- supplement, not replace, SIC/NAICS screens; and
- find cross-industry operating analogues for diligence.

The synthetic peer map was 4.23 times better than random but did not clear the
frozen validation thresholds.

**Regime-stress sensing**

- monitor whether established firm relationships weaken during structural
  shifts; and
- flag periods when historical models or peer sets may require review.

The survival extension improved in 7 of 8 regime-shift folds but failed as a
general predictor. This remains a research hypothesis.

## Rejected uses

- return prediction or investable alpha;
- general firm-survival prediction;
- stable economic archetype classification;
- causal competitive-event prediction; and
- automated real-company decisions without analyst review.

## Practical interface

A future real-data interface should expose:

- firm axis profile and year-over-year movement;
- crowding percentile and change in crowding;
- nearest candidate peers with distances and data coverage;
- drift score with the components that moved;
- regime-health and missing-data indicators; and
- source dates, model version, and audit receipt for every observation.

It should deliberately omit buy/sell labels, survival probabilities, and
archetype names unless a future independently frozen validation supports
them.

## Real-data validation sequence

1. Freeze the filing universe, timestamp rules, axis construction, and
   missing-data policy.
2. Evaluate coordinate stability, coverage, and sensitivity without reading
   outcomes.
3. Test crowding stability across adjacent filings and independent data
   perturbations.
4. Conduct blinded analyst review of high- and low-crowding cases.
5. Only after the measurement gate passes, preregister a separate event or
   outcome study.

This sequence preserves the project's strongest asset: a clear separation
between measurement validation and outcome claims.
