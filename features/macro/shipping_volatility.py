"""
Shipping Market + Volatility Cross-Asset Macro Features

Derives dry-bulk, tanker, LNG, vessel-value proxy, and OVX features used by
shipping and macro overlays.
"""

from typing import Dict, List

import numpy as np
import pandas as pd

from .base import MacroFeatureEngine
from ...data.models import MacroFeature, MacroSeries


class ShippingVolatilityFeatures(MacroFeatureEngine):
    """Compute shipping-market and OVX benchmark features."""

    SERIES_SPECS = {
        "BALTIC_DRY_INDEX": {"prefix": "bdi", "momentum_window": 20},
        "BALTIC_CAPESIZE_INDEX": {"prefix": "bci", "momentum_window": 20},
        "BALTIC_PANAMAX_INDEX": {"prefix": "bpi", "momentum_window": 20},
        "BALTIC_SUPRAMAX_INDEX": {"prefix": "bsi", "momentum_window": 20},
        "BALTIC_DIRTY_TANKER_INDEX": {"prefix": "bdti", "momentum_window": 10},
        "BALTIC_CLEAN_TANKER_INDEX": {"prefix": "bcti", "momentum_window": 10},
        "LNG_CARRIER_RATE_PROXY": {"prefix": "lng_rate", "momentum_window": 10},
        "BULKER_VESSEL_VALUE_PROXY": {"prefix": "bulker_value", "momentum_window": 20},
        "TANKER_VESSEL_VALUE_PROXY": {"prefix": "tanker_value", "momentum_window": 20},
        "LNG_CARRIER_VESSEL_VALUE_PROXY": {"prefix": "lng_carrier_value", "momentum_window": 20},
        "CRUDE_OIL_VOLATILITY_INDEX": {"prefix": "ovx", "momentum_window": 10},
    }

    def compute(self, macro_data: Dict[str, List[MacroSeries]], **kwargs) -> List[MacroFeature]:
        features: List[MacroFeature] = []

        for series_id, spec in self.SERIES_SPECS.items():
            if series_id not in macro_data:
                continue
            prefix = str(spec["prefix"])
            default_window = int(spec["momentum_window"])
            momentum_window = int(kwargs.get(f"{prefix}_momentum_window", default_window))
            features.extend(
                self._compute_index_features(
                    series_id=series_id,
                    series_data=macro_data[series_id],
                    prefix=prefix,
                    momentum_window=momentum_window,
                )
            )

        return features

    def _compute_index_features(
        self,
        series_id: str,
        series_data: List[MacroSeries],
        prefix: str,
        momentum_window: int,
    ) -> List[MacroFeature]:
        df = self.transform_series_to_dataframe(series_data)
        if df.empty:
            return []

        values = pd.to_numeric(df["value"], errors="coerce")
        values = values.replace([np.inf, -np.inf], np.nan)

        z_window = int(self.config.get("z_window", 252))
        z_min_periods = int(self.config.get("z_min_periods", 30))
        shock_threshold = float(self.config.get("shock_threshold", 2.0))

        rolling_mean = values.rolling(window=z_window, min_periods=z_min_periods).mean()
        rolling_std = values.rolling(window=z_window, min_periods=z_min_periods).std()
        zscore = (values - rolling_mean) / rolling_std.replace(0.0, np.nan)
        momentum = values.pct_change(momentum_window)
        shock_flag = (zscore.abs() >= shock_threshold).astype(float)

        computed: List[MacroFeature] = []

        for timestamp, value in values.dropna().items():
            computed.append(
                self.create_macro_feature(
                    name=f"{prefix}_level",
                    timestamp=timestamp,
                    value=float(value),
                    source_series=[series_id],
                    transform="level",
                    frequency="daily",
                )
            )

        for timestamp, value in zscore.dropna().items():
            computed.append(
                self.create_macro_feature(
                    name=f"{prefix}_zscore",
                    timestamp=timestamp,
                    value=float(value),
                    source_series=[series_id],
                    transform="z_score",
                    frequency="daily",
                    metadata={"window": z_window},
                )
            )

        for timestamp, value in momentum.dropna().items():
            computed.append(
                self.create_macro_feature(
                    name=f"{prefix}_momentum_{momentum_window}d",
                    timestamp=timestamp,
                    value=float(value),
                    source_series=[series_id],
                    transform="pct_change",
                    frequency="daily",
                    metadata={"window": momentum_window},
                )
            )

        for timestamp, value in shock_flag.dropna().items():
            computed.append(
                self.create_macro_feature(
                    name=f"{prefix}_shock_flag",
                    timestamp=timestamp,
                    value=float(value),
                    source_series=[series_id],
                    transform="shock_flag",
                    frequency="daily",
                    metadata={"threshold_abs_z": shock_threshold},
                )
            )

        return computed
