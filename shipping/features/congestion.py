from __future__ import annotations

import pandas as pd

from ...features.base import FeatureEngine


class PortCongestionFeatures(FeatureEngine):
    def compute(self, metrics: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if metrics.empty:
            return pd.DataFrame(columns=["port_vessel_count", "anchored_vessel_count", "avg_dwell_hours", "port_congestion_score"])
        ports = metrics.loc[metrics["zone_type"].isin(["port", "anchorage"])].copy()
        if ports.empty:
            return pd.DataFrame(columns=["port_vessel_count", "anchored_vessel_count", "avg_dwell_hours", "port_congestion_score"])
        daily = (
            ports.groupby("window_start", dropna=False)
            .agg(
                port_vessel_count=("vessel_count", "sum"),
                anchored_vessel_count=("anchorage_event_count", "sum"),
                avg_dwell_hours=("avg_dwell_hours", "mean"),
            )
            .sort_index()
        )
        queue_z = self.rolling_zscore(daily["anchored_vessel_count"].astype(float), window=20, min_periods=5)
        dwell_z = self.rolling_zscore(daily["avg_dwell_hours"].fillna(0.0), window=20, min_periods=5)
        congestion_ratio = daily["anchored_vessel_count"] / daily["port_vessel_count"].replace(0.0, pd.NA)
        congestion_z = self.rolling_zscore(congestion_ratio.fillna(0.0), window=20, min_periods=5)
        daily["port_congestion_score"] = (0.40 * queue_z + 0.35 * congestion_z + 0.25 * dwell_z).clip(lower=0.0)
        return daily.fillna(0.0)
