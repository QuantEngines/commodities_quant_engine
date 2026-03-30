from __future__ import annotations

import pandas as pd

from ...features.base import FeatureEngine


class DwellTimeFeatures(FeatureEngine):
    def compute(self, metrics: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if metrics.empty:
            return pd.DataFrame(columns=["avg_dwell_hours", "dwell_time_shock"])
        port_metrics = metrics.loc[metrics["zone_type"] == "port"].copy()
        if port_metrics.empty:
            return pd.DataFrame(columns=["avg_dwell_hours", "dwell_time_shock"])
        daily = port_metrics.groupby("window_start", dropna=False).agg(avg_dwell_hours=("avg_dwell_hours", "mean")).sort_index()
        daily["dwell_time_shock"] = self.rolling_zscore(daily["avg_dwell_hours"].fillna(0.0), window=20, min_periods=5).clip(lower=0.0)
        return daily.fillna(0.0)
