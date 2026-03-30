from __future__ import annotations

import pandas as pd

from ...features.base import FeatureEngine


class TankerFlowFeatures(FeatureEngine):
    def compute(self, metrics: pd.DataFrame, **kwargs) -> pd.DataFrame:
        if metrics.empty:
            return pd.DataFrame(columns=["tanker_vessel_count", "tanker_flow_momentum"])
        tanker_metrics = metrics.loc[metrics["zone_type"] == "corridor"].copy()
        if tanker_metrics.empty:
            tanker_metrics = metrics.loc[metrics["zone_type"].isin(["corridor", "chokepoint"])].copy()
        if tanker_metrics.empty:
            return pd.DataFrame(columns=["tanker_vessel_count", "tanker_flow_momentum"])
        daily = tanker_metrics.groupby("window_start", dropna=False).agg(tanker_vessel_count=("tanker_vessel_count", "sum")).sort_index()
        baseline = daily["tanker_vessel_count"].rolling(10, min_periods=3).mean().shift(1)
        daily["tanker_flow_momentum"] = ((daily["tanker_vessel_count"] - baseline) / baseline.replace(0.0, pd.NA)).fillna(0.0).clip(-3.0, 3.0)
        return daily.fillna(0.0)
