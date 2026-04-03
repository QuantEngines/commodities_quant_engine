from __future__ import annotations

import pandas as pd
import numpy as np

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
        daily["tanker_vessel_count"] = pd.to_numeric(daily["tanker_vessel_count"], errors="coerce")
        baseline = daily["tanker_vessel_count"].rolling(10, min_periods=3).mean().shift(1)
        denominator = baseline.replace(0.0, np.nan)
        momentum = (daily["tanker_vessel_count"] - baseline) / denominator
        daily["tanker_flow_momentum"] = pd.to_numeric(momentum, errors="coerce").fillna(0.0).clip(-3.0, 3.0)
        return daily.apply(pd.to_numeric, errors="coerce").fillna(0.0)
