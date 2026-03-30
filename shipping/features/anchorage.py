from __future__ import annotations

import pandas as pd

from ...features.base import FeatureEngine


class AnchorageFeatures(FeatureEngine):
    def compute(self, metrics: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if metrics.empty:
            return pd.DataFrame(columns=["anchorage_queue_estimate", "anchorage_buildup_score"])
        anchorage = metrics.loc[metrics["zone_type"] == "anchorage"].copy()
        if anchorage.empty:
            return pd.DataFrame(columns=["anchorage_queue_estimate", "anchorage_buildup_score"])
        daily = (
            anchorage.groupby("window_start", dropna=False)
            .agg(
                anchorage_queue_estimate=("vessel_count", "sum"),
                anchorage_event_count=("anchorage_event_count", "sum"),
            )
            .sort_index()
        )
        queue_z = self.rolling_zscore(daily["anchorage_queue_estimate"].astype(float), window=20, min_periods=5)
        buildup_z = self.rolling_zscore(daily["anchorage_event_count"].astype(float), window=20, min_periods=5)
        daily["anchorage_buildup_score"] = (0.65 * queue_z + 0.35 * buildup_z).clip(lower=0.0)
        return daily.fillna(0.0)
