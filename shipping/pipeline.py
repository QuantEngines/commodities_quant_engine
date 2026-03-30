from __future__ import annotations

from datetime import datetime
from typing import Iterable, List, Mapping, Optional

import pandas as pd

from ..config.settings import settings
from .features import (
    AnchorageFeatures,
    ChokepointStressFeatures,
    DwellTimeFeatures,
    PortCongestionFeatures,
    RouteDisruptionFeatures,
    ShippingMomentumFeatures,
    SpeedAnomalyFeatures,
    TankerFlowFeatures,
)
from .geospatial.zones import ZoneCatalog
from .models import ShippingFeatureVector, ShippingObservationWindow
from .processing import aggregate_shipping_metrics, compute_daily_quality, detect_vessel_events, normalize_port_calls, normalize_route_events, normalize_vessel_positions, reconstruct_tracks


class ShippingFeaturePipeline:
    def __init__(self, zone_catalog: Optional[ZoneCatalog] = None):
        self.zone_catalog = zone_catalog or ZoneCatalog.from_config()
        self.congestion = PortCongestionFeatures()
        self.anchorage = AnchorageFeatures()
        self.route_disruption = RouteDisruptionFeatures()
        self.chokepoint = ChokepointStressFeatures()
        self.tanker_flow = TankerFlowFeatures()
        self.dwell_time = DwellTimeFeatures()
        self.speed_anomaly = SpeedAnomalyFeatures()
        self.shipping_momentum = ShippingMomentumFeatures()

    def run(
        self,
        commodity: str,
        vessel_positions: Iterable[Mapping[str, object]] | pd.DataFrame,
        port_calls: Iterable[Mapping[str, object]] | pd.DataFrame | None = None,
        route_events: Iterable[Mapping[str, object]] | pd.DataFrame | None = None,
        as_of_timestamp: Optional[datetime] = None,
    ) -> List[ShippingFeatureVector]:
        positions = normalize_vessel_positions(vessel_positions, source="shipping_pipeline", commodity_tags=[commodity])
        if positions.empty:
            return []
        port_call_frame = normalize_port_calls(port_calls if port_calls is not None else pd.DataFrame(), source="shipping_pipeline")
        route_event_frame = normalize_route_events(route_events if route_events is not None else pd.DataFrame(), source="shipping_pipeline")
        tracked = reconstruct_tracks(positions)
        events = detect_vessel_events(tracked, self.zone_catalog)
        metrics = aggregate_shipping_metrics(tracked, events, self.zone_catalog, port_calls=port_call_frame)
        quality = compute_daily_quality(positions, events)

        feature_frame = pd.concat(
            [
                self.congestion.compute(metrics),
                self.anchorage.compute(metrics),
                self.route_disruption.compute(metrics, route_events=route_event_frame),
                self.chokepoint.compute(metrics, chokepoint_events=route_event_frame.loc[route_event_frame["route_id"].astype(str).str.contains("HORMUZ|BAB_|MALACCA", na=False)] if not route_event_frame.empty else pd.DataFrame()),
                self.tanker_flow.compute(metrics),
                self.dwell_time.compute(metrics),
                self.speed_anomaly.compute(metrics),
            ],
            axis=1,
        ).sort_index()
        feature_frame = feature_frame.loc[:, ~feature_frame.columns.duplicated()]
        momentum = self.shipping_momentum.compute(feature_frame)
        feature_frame = feature_frame.join(momentum, how="outer").join(quality.set_index("window_start"), how="outer")
        feature_frame = feature_frame.fillna(0.0).sort_index()

        observation_start = pd.to_datetime(positions["timestamp"]).min().to_pydatetime()
        as_of = as_of_timestamp or pd.to_datetime(positions["timestamp"]).max().to_pydatetime()
        vectors: List[ShippingFeatureVector] = []
        for timestamp, row in feature_frame.iterrows():
            observation_window = ShippingObservationWindow(
                start_time=observation_start,
                end_time=pd.Timestamp(timestamp).to_pydatetime(),
                frequency=settings.shipping.aggregation_frequency,
                source="shipping_pipeline",
                commodity=commodity,
                geography_scope=["config_geographies"],
                source_ids=["shipping_pipeline"],
            )
            quality_score = float(row.get("shipping_data_quality_score", 0.0))
            vectors.append(
                ShippingFeatureVector(
                    timestamp=pd.Timestamp(timestamp).to_pydatetime(),
                    commodity=commodity,
                    features={column: float(row.get(column, 0.0)) for column in feature_frame.columns},
                    source="shipping_pipeline",
                    observation_window=observation_window,
                    quality_score=quality_score,
                    confidence_score=max(0.0, min(1.0, quality_score)),
                    geography_tags=["shipping_overlay"],
                    commodity_tags=[commodity],
                    key_drivers=self._key_drivers_from_row(row),
                    notes=["Derived from public/local shipping proxies."],
                )
            )
        return [vector for vector in vectors if vector.timestamp <= as_of]

    def _key_drivers_from_row(self, row: pd.Series) -> List[str]:
        drivers: List[str] = []
        for feature_name, label in (
            ("port_congestion_score", "Port congestion"),
            ("anchorage_buildup_score", "Anchorage buildup"),
            ("route_disruption_score", "Route disruption"),
            ("chokepoint_stress_score", "Chokepoint stress"),
        ):
            if float(row.get(feature_name, 0.0)) > 0.25:
                drivers.append(f"{label} is elevated")
        return drivers[:4]
