import pandas as pd

from ..shipping.features import AnchorageFeatures, ChokepointStressFeatures, PortCongestionFeatures, RouteDisruptionFeatures, TankerFlowFeatures


def build_metrics_frame() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    rows = []
    for idx, timestamp in enumerate(dates):
        stress_boost = 8 if idx >= 24 else 0
        drop_penalty = 10 if idx >= 24 else 0
        rows.extend(
            [
                {
                    "window_start": timestamp,
                    "zone_id": "FUJAIRAH",
                    "zone_type": "port",
                    "vessel_count": 15 + stress_boost,
                    "avg_speed_knots": 3.0,
                    "tanker_vessel_count": 8 + stress_boost,
                    "raw_observation_count": 40 + stress_boost,
                    "event_count": 2,
                    "transit_count": 5,
                    "anchorage_event_count": 1 + stress_boost,
                    "speed_anomaly_count": 0,
                    "max_event_severity": 0.15,
                    "avg_dwell_hours": 12 + stress_boost,
                    "port_call_count": 6,
                },
                {
                    "window_start": timestamp,
                    "zone_id": "ANCHORAGE_FUJAIRAH",
                    "zone_type": "anchorage",
                    "vessel_count": 5 + stress_boost,
                    "avg_speed_knots": 0.8,
                    "tanker_vessel_count": 5 + stress_boost,
                    "raw_observation_count": 20 + stress_boost,
                    "event_count": 1,
                    "transit_count": 0,
                    "anchorage_event_count": 2 + stress_boost,
                    "speed_anomaly_count": 0,
                    "max_event_severity": 0.10,
                    "avg_dwell_hours": 0.0,
                    "port_call_count": 0,
                },
                {
                    "window_start": timestamp,
                    "zone_id": "HORMUZ",
                    "zone_type": "chokepoint",
                    "vessel_count": 18,
                    "avg_speed_knots": 10.0,
                    "tanker_vessel_count": 10,
                    "raw_observation_count": 32,
                    "event_count": 3,
                    "transit_count": max(3, 20 - drop_penalty),
                    "anchorage_event_count": 0,
                    "speed_anomaly_count": 1 + max(0, stress_boost // 4),
                    "max_event_severity": 0.20 + 0.03 * stress_boost,
                    "avg_dwell_hours": 0.0,
                    "port_call_count": 0,
                },
                {
                    "window_start": timestamp,
                    "zone_id": "GULF_TO_SINGAPORE",
                    "zone_type": "corridor",
                    "vessel_count": 20,
                    "avg_speed_knots": 11.0,
                    "tanker_vessel_count": max(2, 24 - drop_penalty),
                    "raw_observation_count": 45,
                    "event_count": 2,
                    "transit_count": max(3, 22 - drop_penalty),
                    "anchorage_event_count": 0,
                    "speed_anomaly_count": 0,
                    "max_event_severity": 0.15 + 0.02 * stress_boost,
                    "avg_dwell_hours": 0.0,
                    "port_call_count": 0,
                },
            ]
        )
    return pd.DataFrame(rows)


def build_route_events() -> pd.DataFrame:
    dates = pd.date_range("2026-01-01", periods=30, freq="D")
    severity = [0.1] * 24 + [0.6] * 6
    detour = [0.0] * 24 + [0.35] * 6
    return pd.DataFrame(
        {
            "route_id": ["GULF_TO_SINGAPORE"] * len(dates),
            "timestamp": dates,
            "event_type": ["rerouting"] * len(dates),
            "severity": severity,
            "detour_distance_ratio": detour,
        }
    )


def test_first_pass_shipping_feature_modules_emit_positive_stress_signals():
    metrics = build_metrics_frame()
    route_events = build_route_events()

    congestion = PortCongestionFeatures().compute(metrics)
    anchorage = AnchorageFeatures().compute(metrics)
    disruption = RouteDisruptionFeatures().compute(metrics, route_events=route_events)
    chokepoint = ChokepointStressFeatures().compute(metrics, chokepoint_events=route_events.rename(columns={"route_id": "chokepoint_id"}))
    tanker_flow = TankerFlowFeatures().compute(metrics)

    latest = pd.concat([congestion, anchorage, disruption, chokepoint, tanker_flow], axis=1).fillna(0.0).iloc[-1]

    assert latest["port_congestion_score"] > 0.0
    assert latest["anchorage_buildup_score"] > 0.0
    assert latest["route_disruption_score"] > 0.0
    assert latest["chokepoint_stress_score"] > 0.0
    assert latest["tanker_flow_momentum"] < 0.0
