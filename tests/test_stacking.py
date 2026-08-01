from __future__ import annotations

import numpy as np

from gas_forecast.stacking import apply_simplex, fit_simplex_state


def test_regularized_simplex_is_nonnegative_and_sums_to_one() -> None:
    actual = np.array([[100.0, 110.0], [101.0, 111.0], [102.0, 112.0]])
    branches = np.stack(
        [actual, actual + 5.0, actual - 8.0],
        axis=1,
    )

    state = fit_simplex_state(branches, actual, ("best", "high", "low"))
    predicted = apply_simplex(branches, state.regularized_weights)

    assert np.all(state.regularized_weights >= 0.0)
    np.testing.assert_allclose(state.regularized_weights.sum(axis=1), 1.0)
    np.testing.assert_allclose(predicted, actual, atol=1e-3)
