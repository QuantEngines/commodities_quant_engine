from __future__ import annotations

import pandas as pd

from ...features.base import FeatureEngine


class RouteDisruptionFeatures(FeatureEngine):
    def compute(self, metrics: pd.DataFrame, route_events: pd.DataFrame | None = None, **kwargs) -> pd.DataFrame:
        if metrics.empty and (route_events is None or route_events.empty):
            return pd.DataFrame(columns=["route_transit_count", "rerouting_indicator", "route_disruption_score"])
        route_metrics = metrics.loc[metrics["zone_type"].isin(["corridor", "chokepoint"])].copy() if not metrics.empty else pd.DataFrame()
        daily = (
            route_metrics.groupby("window_start", dropna=False)
            .agg(
                route_transit_count=("transit_count", "sum"),
                route_event_severity=("max_event_severity", "mean"),
            )
            .sort_index()
            if not route_metrics.empty
            else pd.DataFrame()
        )
        if route_events is not None and not route_events.empty:
            event_frame = route_events.copy()
            event_frame["window_start"] = pd.to_datetime(event_frame["timestamp"]).dt.floor("1D")
            event_daily = (
                event_frame.groupby("window_start", dropna=False)
                .agg(
                    manual_route_severity=("severity", "mean"),
                    rerouting_indicator=("detour_distance_ratio", "mean"),
                )
                .sort_index()
            )
            daily = event_daily if daily.empty else daily.join(event_daily, how="outer")
        if daily.empty:
            return pd.DataFrame(columns=["route_transit_count", "rerouting_indicator", "route_disruption_score"])
        transit_drop = self.rolling_zscore((-daily["route_transit_count"].astype(float)).fillna(0.0), window=20, min_periods=5).clip(lower=0.0)
        reroute = daily.get("rerouting_indicator", pd.Series(0.0, index=daily.index)).fillna(0.0)
        severity = daily.get("manual_route_severity", daily.get("route_event_severity", pd.Series(0.0, index=daily.index))).fillna(0.0)
        daily["rerouting_indicator"] = reroute.clip(lower=0.0)
        daily["route_disruption_score"] = (0.45 * transit_drop + 0.30 * severity + 0.25 * daily["rerouting_indicator"]).clip(lower=0.0)
        return daily.fillna(0.0)
