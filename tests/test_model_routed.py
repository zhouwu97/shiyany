from __future__ import annotations

import numpy as np
import pandas as pd

from gas_forecast.config import FeatureConfig, ForecastConfig
from gas_forecast.model_routed import RoutedLegacyForecaster


def test_persistence_only_route_needs_no_fitted_base_model() -> None:
    config = ForecastConfig(feature=FeatureConfig(horizons=(1, 2)))
    route = {
        "global": {"selected": "persistence_pred"},
        "targets": {},
        "cells": {},
    }
    model = RoutedLegacyForecaster(route, config).fit(
        pd.DataFrame(index=pd.date_range("2025-01-01", periods=2, freq="15min")),
        pd.DataFrame(),
        pd.DataFrame({"generator_1": [100.0, 101.0], "generator_all": [220.0, 221.0]}),
    )
    current = pd.DataFrame(
        {"generator_1": [100.0, 101.0], "generator_all": [220.0, 221.0]},
        index=pd.date_range("2025-01-01", periods=2, freq="15min"),
    )

    prediction = model.predict(pd.DataFrame(index=current.index), current)

    assert model.models_ == {}
    np.testing.assert_allclose(prediction["generator_1_t+15_pred"], [100.0, 101.0])
    np.testing.assert_allclose(prediction["generator_all_t+30_pred"], [220.0, 221.0])


def test_routed_predictions_enforce_remaining_unit_capacity() -> None:
    config = ForecastConfig(feature=FeatureConfig(horizons=(1,)))
    route = {
        "global": {"selected": "persistence_pred"},
        "targets": {},
        "cells": {},
    }
    current = pd.DataFrame(
        {"generator_1": [100.0], "generator_all": [430.0]},
        index=pd.date_range("2025-01-01", periods=1, freq="15min"),
    )
    prediction = RoutedLegacyForecaster(route, config).fit(
        pd.DataFrame(index=current.index), pd.DataFrame(), current
    ).predict(pd.DataFrame(index=current.index), current)

    assert prediction.loc[current.index[0], "generator_all_t+15_pred"] == 340.0
