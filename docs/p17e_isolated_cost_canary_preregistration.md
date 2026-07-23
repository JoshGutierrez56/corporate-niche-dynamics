# P17E — Isolated Locked-Proxy Cost Canary

## Phase boundary

P17E is the smallest phase after the validated P16E canary. It runs the
cost-aware P8 contract for the already locked `anchored_axis_innovation` and
stops before P9. It may not change the candidate, injection strength, return
horizon, portfolio weighting, neutrality rule, cost cases, or execution delay
after opening P17E results.

The run uses only:

- the isolated P16E 6× synthetic `migration_alpha` and `null_alpha` P7 paths;
- the P16E candidate files built without synthetic truth;
- the existing point-in-time synthetic liquidity and factor inputs; and
- the unchanged P8 cost, borrow, capacity, and delayed-execution equations.

No truth sidecar may be read during portfolio construction.

## Frozen design

- candidate: `anchored_axis_innovation`
- scenarios: `migration_alpha` and `null_alpha`
- horizon: six months
- portfolio ladder: equal/value weighting × pooled/dated-SIC1-neutral
- primary portfolio: value-weighted, dated-SIC1-neutral
- cost cases: existing low, conservative, and severe cases
- primary cost case: conservative
- execution delays: zero and one additional month
- capacity notional: existing $1 million per leg
- output root: the isolated canary only
- no P9 clustering, P10 reporting, real data, or signal/specification search

P17E reconstructs the locked-proxy P7 assignments from the already frozen P7
targets and return paths, then applies the existing P8 equations. It does not
rewrite the canonical P7 bundle or relabel the obsolete `migration_surprise`
as the repaired signal.

## Acceptance gates

`PASS` is an engineering and traceability result, not a positive-return gate.
All of the following must hold:

1. both scenarios produce exactly one locked signal and one six-month horizon;
2. all four weighting/neutrality specifications, three cost cases, and two
   execution delays are present;
3. costs, borrow, capacity, and lagged-liquidity rules independently
   recompute;
4. no position exceeds its capacity limit and unavailable shorts receive zero
   capacity;
5. every saved input and output hash matches;
6. construction reads no truth sidecar;
7. the migration and null primary net results are reported without selecting
   between them; and
8. the main P0-P10 artifacts remain byte-identical.

Positive, null, or negative net performance is admissible and must be reported
honestly. P17E remains synthetic evidence and cannot support an investable or
real-data claim.
