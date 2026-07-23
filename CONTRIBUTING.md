# Contributing

Corporate Niche Dynamics is an evidence-first research repository. Changes are
welcome when they improve correctness, portability, testing, or documentation
without rewriting the meaning of a frozen result.

## Before opening a pull request

1. Read `docs/research_spec.md` and `docs/point_in_time_policy.md`.
2. Install the locked development environment with `uv sync --extra dev`.
3. Run `uv run pytest -q`.
4. Explain whether the change affects code, data, protocol, evidence, or prose.
5. Add a test for any change to point-in-time logic, validation, or numerical
   behavior.

## Frozen evidence

Do not silently replace a preregistration, threshold, negative result, signed
manifest, or claim-ledger decision. A change that could alter a research claim
must:

- use a new versioned protocol or phase;
- state whether it is confirmatory or exploratory;
- freeze its gates before opening the relevant result;
- preserve the earlier result; and
- write a new independent validation receipt.

Publication-only changes such as README, CI, portability, or packaging edits
belong in the public repository manifest. They must not be represented as part
of an earlier analytical run.

## Data and privacy

- Do not commit licensed Compustat, CRSP, WRDS, SEC-derived restricted, or
  other non-redistributable source data.
- Do not commit secrets, credentials, personal paths, or private identifiers.
- Keep generated Parquet data and fitted model objects outside Git.
- Add only compact, reviewable evidence artifacts needed to substantiate a
  documented result.

## Style

- Prefer transparent implementations over opaque optimization.
- Keep builders and independent validators separate.
- Use deterministic seeds and atomic writes.
- Preserve row-count waterfalls, missingness, and coverage diagnostics.
- Keep real-data, causal, survival, and investment claims out of synthetic
  results.

## Pull-request checklist

- [ ] Tests pass locally.
- [ ] No future information enters a feature, split, label, or target.
- [ ] Config and dependency changes are documented.
- [ ] New outputs have independent validation.
- [ ] Claim language matches the evidence tier.
- [ ] No generated bulk data, models, secrets, or licensed data are included.
