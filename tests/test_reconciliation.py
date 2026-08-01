from __future__ import annotations

import numpy as np

from gas_forecast.reconciliation import fit_reconciliation, reconcile_predictions


def test_reconciliation_enforces_generator_sum() -> None:
    rng = np.random.default_rng(42)
    samples = 100
    horizons = 2
    one = rng.normal(100, 2, size=(samples, horizons))
    rest = rng.normal(120, 3, size=(samples, horizons))
    actual = np.stack([one, rest, one + rest], axis=1)
    base = actual + rng.normal(0, 2, size=actual.shape)

    state = fit_reconciliation(actual, base, estimate_full_covariance=True)
    candidates = reconcile_predictions(base, state)

    diagonal = candidates["diagonal_reconciled"]
    full = candidates["full_covariance_reconciled"]
    np.testing.assert_allclose(diagonal[:, 2], diagonal[:, 0] + diagonal[:, 1])
    np.testing.assert_allclose(full[:, 2], full[:, 0] + full[:, 1])
    assert np.all((state.direct_bottom_weights >= 0) & (state.direct_bottom_weights <= 1))
