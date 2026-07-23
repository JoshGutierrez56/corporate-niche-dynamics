# P14F — P7 Eligibility Audit Results

## Gate result

`GO`

On the exact six-month P7 eligibility contract over 2002-2018:

- rows: `22,933`
- firms: `2,317`
- anchored-proxy truth Spearman: `0.4617`
- existing-benchmark truth Spearman: `0.2058`
- improvement: `+0.2559`
- candidate coverage: `96.32%`
- calibration Spearman: `0.4655`
- evaluation Spearman: `0.4560`
- 2013-2015 Spearman: `0.4804`
- 2016-2018 Spearman: `0.4403`

Every frozen gate passed. Candidate construction and this audit read no return
values or injected alpha. The independent validator passed.

## Interpretation

The P7 failure is primarily a measurement problem. The transparent
anchored-axis innovation recovers more than twice the synthetic-truth rank
alignment of the failure-model-based migration benchmark on the relevant
sample, without sacrificing the required coverage.
