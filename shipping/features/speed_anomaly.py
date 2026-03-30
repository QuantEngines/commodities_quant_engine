from __future__ import annotations

import pandas as pd

from ...features.base import FeatureEngine


class SpeedAnomalyFeatures(FeatureEngine):
    def compute(self, metrics: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if metrics.empty:
            return pd.DataFrame(columns=["speed_anomaly_count", "speed_anomaly_score"])
        daily = metrics.groupby("window_start", dropna=False).agg(speed_anomaly_count=("speed_anomaly_count", "sum")).sort_index()
        daily["speed_anomaly_score"] = self.rolling_zscore(daily["speed_anomaly_count"].astype(float), window=20, min_periods=5).clip(lower=0.0)
        return daily.fillna(0.0)
