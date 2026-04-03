from typing import Sequence

import pandas as pd

from ..data.models import MacroSeries
from ..features.macro import ShippingVolatilityFeatures


def _make_series(series_id: str, values: Sequence[float]) -> list[MacroSeries]:
    index = pd.date_range("2025-01-01", periods=len(values), freq="B")
    return [
        MacroSeries(
            series_id=series_id,
            timestamp=timestamp.to_pydatetime(),
            value=float(value),
            unit="index",
            frequency="daily",
            source="test",
        )
        for timestamp, value in zip(index, values)
    ]


def test_shipping_volatility_features_include_bdi_and_ovx_blocks():
    engine = ShippingVolatilityFeatures({"z_window": 20, "z_min_periods": 10, "shock_threshold": 1.5})
    bdi_values = [1000 + i * 12 for i in range(35)]
    ovx_values = [30 + (i % 5) for i in range(34)] + [70]
    bdti_values = [600 + i * 8 for i in range(35)]
    lng_values = [80 + i * 0.7 for i in range(35)]

    features = engine.compute(
        {
            "BALTIC_DRY_INDEX": _make_series("BALTIC_DRY_INDEX", bdi_values),
            "BALTIC_DIRTY_TANKER_INDEX": _make_series("BALTIC_DIRTY_TANKER_INDEX", bdti_values),
            "LNG_CARRIER_RATE_PROXY": _make_series("LNG_CARRIER_RATE_PROXY", lng_values),
            "CRUDE_OIL_VOLATILITY_INDEX": _make_series("CRUDE_OIL_VOLATILITY_INDEX", ovx_values),
        }
    )

    feature_names = {feature.feature_name for feature in features}
    assert "bdi_level" in feature_names
    assert "bdi_zscore" in feature_names
    assert "bdi_momentum_20d" in feature_names
    assert "bdi_shock_flag" in feature_names
    assert "ovx_level" in feature_names
    assert "ovx_zscore" in feature_names
    assert "ovx_momentum_10d" in feature_names
    assert "ovx_shock_flag" in feature_names
    assert "bdti_level" in feature_names
    assert "bdti_zscore" in feature_names
    assert "bdti_momentum_10d" in feature_names
    assert "lng_rate_level" in feature_names
    assert "lng_rate_zscore" in feature_names
    assert "lng_rate_momentum_10d" in feature_names

    latest_timestamp = max(feature.timestamp for feature in features)
    latest_ovx_shock = [
        feature
        for feature in features
        if feature.feature_name == "ovx_shock_flag" and feature.timestamp == latest_timestamp
    ]
    assert latest_ovx_shock
    assert latest_ovx_shock[-1].value in (0.0, 1.0)
