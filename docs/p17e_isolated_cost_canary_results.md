# P17E — Isolated Locked-Proxy Cost Canary Results

## Result

`PASS` on all engineering, isolation, and traceability gates.

Both preregistered scenarios produced the complete one-signal, six-month P8
ladder: four weighting/neutrality specifications, three frozen cost cases, and
zero/one-month execution delays. The independent validator reported zero
errors, the full suite passed 83 tests, no truth sidecar was read, and the main
P0-P10 manifest hashes were unchanged.

## Primary portfolio

The primary portfolio is the preregistered six-month, value-weighted,
dated-SIC1-neutral spread under conservative costs and no additional delay.

| Scenario | Gross capacity return | Net return | Net Sharpe | Turnover | Capacity fill |
|---|---:|---:|---:|---:|---:|
| migration alpha | 2.49% | -5.16% | -0.84 | 44.52% | 65.46% |
| null alpha | -2.17% | -8.86% | -1.58 | 39.96% | 58.65% |

For the migration scenario, annualized transaction costs were `5.62%` and
borrow costs were `2.02%`; they overwhelmed the positive gross capacity
return. Net factor alpha was `-0.00406` per month (`p=0.00243`). The null
control was negative both gross and net.

The migration scenario exceeded the null control by about `4.65` percentage
points gross and `3.70` percentage points net, but its primary net result
remained negative. This relative gap is descriptive synthetic evidence, not a
tradeable-return claim.

## Frozen sensitivities

The migration scenario remained negative under the no-delay low-cost case
(`-2.35%` annualized net) and the severe case (`-9.04%`). The preregistered
one-month-delay/low-cost sensitivity was slightly positive (`0.46%`), but it
is not the primary specification and is not promoted or selected.

## Interpretation

P16E repaired signal recovery, but P17E does not support implementability
under the frozen primary execution assumptions. The locked signal creates a
positive synthetic gross spread relative to the null control, while turnover,
spread/slippage, borrow, and capacity constraints erase it.

P17E stops before P9. No clustering, P10 reporting, real-data run, or
investable-performance claim is authorized by this result.
