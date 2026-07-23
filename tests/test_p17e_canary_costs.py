"""Focused tests for the P17E locked-proxy cost canary."""

from __future__ import annotations

import pandas as pd

from hypercube.canary_costs import (
    LOCKED_HORIZON,
    LOCKED_SIGNAL,
    locked_proxy_config,
)
from hypercube.config import load_config
from hypercube.returns import SIGNAL_LABELS, construct_portfolio_sorts


def test_locked_proxy_config_preserves_canonical_config() -> None:
    canonical = load_config("configs/synthetic.yaml")
    p17 = locked_proxy_config(canonical)
    assert canonical.returns.primary_signal == "migration_surprise"
    assert p17.returns.signals == (LOCKED_SIGNAL,)
    assert p17.returns.horizons_months == (LOCKED_HORIZON,)
    assert p17.returns.primary_signal == LOCKED_SIGNAL
    assert p17.project.phase == "P8"


def test_locked_proxy_has_explicit_signal_label() -> None:
    assert SIGNAL_LABELS[LOCKED_SIGNAL] == LOCKED_SIGNAL


def test_locked_proxy_sort_uses_only_preregistered_signal() -> None:
    config = locked_proxy_config(load_config("configs/synthetic.yaml"))
    rows = []
    for gvkey in range(40):
        rows.append(
            {
                "event_id": f"e{gvkey}",
                "gvkey": str(gvkey),
                "feature_date": pd.Timestamp("2010-12-31"),
                "fold": 1,
                "sic1": gvkey % 4,
                "market_cap_millions": 100.0 + gvkey,
                "target_valid": True,
                "forward_total_return": gvkey / 1000.0,
                "forward_excess_return": gvkey / 1000.0,
                "horizon_months": LOCKED_HORIZON,
                LOCKED_SIGNAL: float(gvkey),
                **{control: 0.0 for control in config.returns.controls},
            }
        )
    _, assignments, _ = construct_portfolio_sorts(
        pd.DataFrame(rows), config
    )
    assert set(assignments["signal"]) == {LOCKED_SIGNAL}
    assert set(assignments["horizon_months"]) == {LOCKED_HORIZON}
