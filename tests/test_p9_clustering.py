"""Focused P9 tests for neutral labels and training-only assignment."""

from __future__ import annotations

import numpy as np

from hypercube.clustering import _assign_nearest, _excel_label


def test_archetype_labels_are_neutral_and_extend_past_z() -> None:
    assert _excel_label(0) == "Archetype A"
    assert _excel_label(25) == "Archetype Z"
    assert _excel_label(26) == "Archetype AA"


def test_radius_assignment_preserves_noise_uncertainty() -> None:
    values = np.array([[0.1, 0.0], [10.0, 10.0]])
    labels, confidence, distance = _assign_nearest(
        values,
        {0: np.array([0.0, 0.0]), 1: np.array([2.0, 2.0])},
        {0: 1.0, 1: 1.0},
    )
    assert labels.tolist() == [0, -1]
    assert 0.0 < confidence[0] < 1.0
    assert confidence[1] == 0.0
    assert distance[1] > 1.0
