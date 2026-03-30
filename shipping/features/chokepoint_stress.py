from __future__ import annotations

import pandas as pd

from ...features.base import FeatureEngine


class ChokepointStressFeatures(FeatureEngine):
    def compute(self, metrics: pd.DataFrame, chokepoint_events: pd.DataFrame | None = None, **kwargs) -> pd.DataFrame:
        if metrics.empty and (chokepoint_events is None or chokepoint_events.empty):
            return pd.DataFrame(columns=["chokepoint_transit_count", "chokepoint_stress_score", "corridor_stress_score"])
        chokepoint = metrics.loc[metrics["zone_type"] == "chokepoint"].copy() if not metrics.empty else pd.DataFrame()
        daily = (
            chokepoint.groupby("window_start", dropna=False)
            .agg(
                chokepoint_transit_count=("transit_count", "sum"),
                chokepoint_speed_anomaly_count=("speed_anomaly_count", "sum"),
                chokepoint_event_severity=("max_event_severity", "mean"),
            )
            .sort_index()
            if not chokepoint.empty
            else pd.DataFrame()
        )
        if chokepoint_events is not None and not chokepoint_events.empty:
            event_frame = chokepoint_events.copy()
            event_frame["window_start"] = pd.to_datetime(event_frame["timestamp"]).dt.floor("1D")
            event_daily = event_frame.groupby("window_start", dropna=False).agg(manual_cp_severity=("severity", "mean")).sort_index()
            daily = event_daily if daily.empty else daily.join(event_daily, how="outer")
        if daily.empty:
            return pd.DataFrame(columns=["chokepoint_transit_count", "chokepoint_stress_score", "corridor_stress_score"])
        transit_z = self.rolling_zscore((-daily["chokepoint_transit_count"].astype(float)).fillna(0.0), window=20, min_periods=5).clip(lower=0.0)
        speed_anomaly_z = self.rolling_zscore(daily["chokepoint_speed_anomaly_count"].astype(float).fillna(0.0), window=20, min_periods=5).clip(lower=0.0)
        severity = daily.get("manual_cp_severity", daily.get("chokepoint_event_severity", pd.Series(0.0, index=daily.index))).fillna(0.0)
        daily["chokepoint_stress_score"] = (0.40 * transit_z + 0.30 * speed_anomaly_z + 0.30 * severity).clip(lower=0.0)
        daily["corridor_stress_score"] = daily["chokepoint_stress_score"]
        return daily.fillna(0.0)
