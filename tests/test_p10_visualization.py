"""Focused tests for P10 traceability helpers."""

from __future__ import annotations

import pandas as pd

from hypercube.visualization import _primary_mask


def test_primary_mask_freezes_signal_horizon_weight_and_neutrality() -> None:
    frame = pd.DataFrame(
        {
            "horizon_months": [6, 6, 3],
            "signal": [
                "migration_surprise",
                "viability_log_odds",
                "migration_surprise",
            ],
            "weighting": ["value", "value", "value"],
            "industry_neutral": [True, True, True],
        }
    )
    assert _primary_mask(frame).tolist() == [True, False, False]
