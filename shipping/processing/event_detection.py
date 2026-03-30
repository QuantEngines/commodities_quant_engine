from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from ..config import load_shipping_features
from ..geospatial.zones import ZoneCatalog


def _ensure_list(value: object) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    return [str(value)]


def _transition_events(frame: pd.DataFrame, id_column: str, event_type: str) -> List[Dict[str, object]]:
    events: List[Dict[str, object]] = []
    for vessel_id, vessel_frame in frame.groupby("vessel_id", sort=False):
        previous: set[str] = set()
        for row in vessel_frame.itertuples():
            current = set(_ensure_list(getattr(row, id_column)))
            enters = current.difference(previous)
            for zone_id in enters:
                events.append(
                    {
                        "event_id": f"{event_type}_{vessel_id}_{zone_id}_{pd.Timestamp(row.timestamp).strftime('%Y%m%d%H%M%S')}",
                        "vessel_id": vessel_id,
                        "event_type": event_type,
                        "timestamp": row.timestamp,
                        "zone_id": zone_id,
                        "zone_type": id_column.replace("_ids", ""),
                        "severity": 0.25,
                        "source": "derived",
                    }
                )
            previous = current
    return events


def detect_vessel_events(positions: pd.DataFrame, zone_catalog: ZoneCatalog) -> pd.DataFrame:
    if positions.empty:
        return pd.DataFrame(columns=["event_id", "vessel_id", "event_type", "timestamp", "zone_id", "zone_type", "severity", "source"])
    feature_config = load_shipping_features()
    processing = feature_config.get("processing", {})
    anchorage_speed_threshold = float(processing.get("anchorage_speed_threshold_knots", 1.5))
    anchorage_min_duration_hours = float(processing.get("anchorage_min_duration_hours", 8.0))
    speed_ratio_threshold = float(processing.get("speed_anomaly_ratio_threshold", 0.55))

    frame = zone_catalog.annotate_positions(positions).sort_values(["vessel_id", "timestamp"]).copy()
    events: List[Dict[str, object]] = []
    for column, event_type in (
        ("port_ids", "zone_enter_port"),
        ("anchorage_ids", "zone_enter_anchorage"),
        ("chokepoint_ids", "zone_enter_chokepoint"),
        ("corridor_ids", "zone_enter_corridor"),
    ):
        events.extend(_transition_events(frame, column, event_type))

    for vessel_id, vessel_frame in frame.groupby("vessel_id", sort=False):
        vessel_frame = vessel_frame.sort_values("timestamp").copy()
        rolling_speed = vessel_frame["speed_knots"].ffill().fillna(0.0).rolling(5, min_periods=2).median().shift(1)
        anomaly_mask = (
            vessel_frame["speed_knots"].fillna(0.0) < rolling_speed.fillna(0.0) * speed_ratio_threshold
        ) & (vessel_frame["speed_knots"].fillna(0.0) > 0.0)
        for row in vessel_frame.loc[anomaly_mask].itertuples():
            events.append(
                {
                    "event_id": f"speed_anomaly_{vessel_id}_{pd.Timestamp(row.timestamp).strftime('%Y%m%d%H%M%S')}",
                    "vessel_id": vessel_id,
                    "event_type": "speed_anomaly",
                    "timestamp": row.timestamp,
                    "zone_id": (_ensure_list(row.corridor_ids) or _ensure_list(row.chokepoint_ids) or [None])[0],
                    "zone_type": "corridor",
                    "severity": 0.35,
                    "source": "derived",
                }
            )

        in_anchorage = vessel_frame["anchorage_ids"].apply(lambda value: len(_ensure_list(value)) > 0)
        low_speed = vessel_frame["speed_knots"].fillna(99.0) <= anchorage_speed_threshold
        anchorage_mask = in_anchorage & low_speed
        anchor_groups = anchorage_mask.ne(anchorage_mask.shift(fill_value=False)).cumsum()
        for _, segment in vessel_frame.loc[anchorage_mask].groupby(anchor_groups[anchorage_mask]):
            if segment.empty:
                continue
            duration = (segment["timestamp"].iloc[-1] - segment["timestamp"].iloc[0]).total_seconds() / 3600.0
            if duration < anchorage_min_duration_hours:
                continue
            anchor_zone_ids = _ensure_list(segment["anchorage_ids"].iloc[0])
            events.append(
                {
                    "event_id": f"anchorage_buildup_{vessel_id}_{pd.Timestamp(segment['timestamp'].iloc[0]).strftime('%Y%m%d%H%M%S')}",
                    "vessel_id": vessel_id,
                    "event_type": "anchorage_buildup",
                    "timestamp": segment["timestamp"].iloc[-1],
                    "start_time": segment["timestamp"].iloc[0],
                    "end_time": segment["timestamp"].iloc[-1],
                    "zone_id": anchor_zone_ids[0] if anchor_zone_ids else None,
                    "zone_type": "anchorage",
                    "severity": min(1.0, duration / 48.0),
                    "source": "derived",
                }
            )

    event_frame = pd.DataFrame(events)
    if event_frame.empty:
        return pd.DataFrame(columns=["event_id", "vessel_id", "event_type", "timestamp", "zone_id", "zone_type", "severity", "source"])
    event_frame["severity"] = pd.to_numeric(event_frame["severity"], errors="coerce").fillna(0.0)
    event_frame["timestamp"] = pd.to_datetime(event_frame["timestamp"])
    return event_frame.sort_values(["timestamp", "event_id"]).reset_index(drop=True)
