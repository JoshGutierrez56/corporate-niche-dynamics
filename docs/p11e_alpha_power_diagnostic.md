# P11E — Exploratory Alpha-Power Diagnostic

## Boundary

P11E is a versioned, post-closeout exploratory diagnostic. It does not alter
P0-P10 outputs, change a frozen hypothesis, run real data, select a trading
specification, or promote performance.

## Question

Was the frozen migration-alpha data-generating process strong enough for the
declared six-month rank-IC recovery test to detect even the hidden oracle
signal?

## Reproduce

```powershell
.\.venv\Scripts\python.exe scripts\diagnose_alpha_recovery_power.py
.\.venv\Scripts\python.exe scripts\validate_alpha_recovery_power.py
```

The build reads only the frozen migration-alpha P7 forward targets and the
synthetic truth sidecar. The independent validator recomputes every saved
summary and fold statistic.

## Interpretation rule

P11E reports whether the injected-alpha slope is detectable with
issuer-clustered uncertainty. This is descriptive and was defined after P10;
it is not a retroactive confirmatory gate.

If the oracle is not detectable, P7 cannot cleanly distinguish a weak
observable feature from an underpowered synthetic return injection. Before a
new synthetic run, a separate redesign must freeze an oracle detectability and
Monte Carlo power requirement. Real outcomes must remain unopened.

P12E subsequently froze and ran that linearized calibration through a 10x
injection. The oracle became detectable, but the observable feature did not.
See `docs/p12e_power_calibration_results.md`.
