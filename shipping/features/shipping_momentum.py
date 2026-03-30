from __future__ import annotations

import pandas as pd

from ...features.base import FeatureEngine


class ShippingMomentumFeatures(FeatureEngine):
    def compute(self, feature_frame: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if feature_frame.empty:
            return pd.DataFrame(columns=["shipping_momentum_score"])
        available = [column for column in feature_frame.columns if column.endswith("_score")]
        if not available:
            return pd.DataFrame(columns=["shipping_momentum_score"])
        momentum = feature_frame[available].mean(axis=1).rolling(5, min_periods=1).mean()
        return pd.DataFrame({"shipping_momentum_score": momentum.fillna(0.0)}, index=feature_frame.index)
