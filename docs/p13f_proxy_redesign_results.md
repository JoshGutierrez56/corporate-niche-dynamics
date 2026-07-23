# P13F — Proxy-Redesign Results

## Result

`NO_GO` on the full P5-population improvement gate.

The locked anchored-axis innovation was selected in calibration and remained
stable:

- calibration truth Spearman: `0.4695`
- evaluation truth Spearman: `0.4504`
- 2013-2015 Spearman: `0.4608`
- 2016-2018 Spearman: `0.4403`
- evaluation coverage versus the benchmark: `115.36%`

The full-population P5 benchmark reached `0.4578` in the evaluation years, so
the anchored proxy did not clear the frozen `+0.10` improvement requirement.
The independent P13F validator passed.

## Interpretation

The candidate is a much more stable state measurement than the benchmark in
the earlier calibration years, but P13F alone cannot promote it. P14F therefore
tests the already-locked candidate on the exact downstream P7 eligibility
contract without reading return values.

P13E's incorrect three-year-population artifacts and P14E's pre-calibration
coverage artifacts remain archived and excluded from decisions.
