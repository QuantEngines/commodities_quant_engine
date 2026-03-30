from __future__ import annotations

from typing import List

import pandas as pd

from ...config.settings import settings
from ..geospatial.zones import ZoneCatalog


def _explode_zone_column(frame: pd.DataFrame, column: str, zone_type: str, freq: str) -> pd.DataFrame:
    if column not in frame.columns or frame.empty:
        return pd.DataFrame()
    exploded = frame.copy()
    exploded[column] = exploded[column].apply(lambda value: value if isinstance(value, list) else [])
    exploded = exploded.explode(column)
    exploded = exploded.dropna(subset=[column]).copy()
    if exploded.empty:
        return exploded
    exploded["window_start"] = pd.to_datetime(exploded["timestamp"]).dt.floor(freq)
    exploded["zone_id"] = exploded[column].astype(str)
    exploded["zone_type"] = zone_type
    exploded["is_tanker"] = exploded["cargo_class"].fillna("").astype(str).str.contains("tanker|crude|oil", case=False, regex=True)
    aggregated = (
        exploded.groupby(["window_start", "zone_id", "zone_type"], dropna=False)
        .agg(
            vessel_count=("vessel_id", "nunique"),
            avg_speed_knots=("speed_knots", "mean"),
            tanker_vessel_count=("is_tanker", "sum"),
            raw_observation_count=("timestamp", "size"),
        )
        .reset_index()
    )
    return aggregated


def aggregate_shipping_metrics(
    positions: pd.DataFrame,
    events: pd.DataFrame,
    zone_catalog: ZoneCatalog,
    port_calls: pd.DataFrame | None = None,
    freq: str | None = None,
) -> pd.DataFrame:
    if positions.empty:
        return pd.DataFrame()
    freq = freq or settings.shipping.aggregation_frequency
    annotated = zone_catalog.annotate_positions(positions)
    metric_frames: List[pd.DataFrame] = []
    for column, zone_type in (
        ("port_ids", "port"),
        ("anchorage_ids", "anchorage"),
        ("chokepoint_ids", "chokepoint"),
        ("corridor_ids", "corridor"),
    ):
        metric_frames.append(_explode_zone_column(annotated, column, zone_type, freq))
    metrics = pd.concat([frame for frame in metric_frames if not frame.empty], ignore_index=True) if metric_frames else pd.DataFrame()
    if metrics.empty:
        return pd.DataFrame()

    if events is not None and not events.empty:
        event_frame = events.copy()
        event_frame["window_start"] = pd.to_datetime(event_frame["timestamp"]).dt.floor(freq)
        event_summary = (
            event_frame.groupby(["window_start", "zone_id", "zone_type"], dropna=False)
            .agg(
                event_count=("event_id", "size"),
                transit_count=("event_type", lambda values: int(sum(str(value).startswith("zone_enter") for value in values))),
                anchorage_event_count=("event_type", lambda values: int(sum(str(value) == "anchorage_buildup" for value in values))),
                speed_anomaly_count=("event_type", lambda values: int(sum(str(value) == "speed_anomaly" for value in values))),
                max_event_severity=("severity", "max"),
            )
            .reset_index()
        )
        metrics = metrics.merge(
            event_summary,
            how="left",
            on=["window_start", "zone_id", "zone_type"],
        )
    else:
        metrics["event_count"] = 0
        metrics["transit_count"] = 0
        metrics["anchorage_event_count"] = 0
        metrics["speed_anomaly_count"] = 0
        metrics["max_event_severity"] = 0.0

    if port_calls is not None and not port_calls.empty:
        port_call_frame = port_calls.copy()
        port_call_frame["window_start"] = pd.to_datetime(port_call_frame["arrival_time"]).dt.floor(freq)
        departure = pd.to_datetime(port_call_frame["departure_time"], errors="coerce")
        arrival = pd.to_datetime(port_call_frame["arrival_time"], errors="coerce")
        port_call_frame["dwell_hours"] = (departure - arrival).dt.total_seconds().div(3600.0).fillna(0.0)
        dwell_summary = (
            port_call_frame.groupby(["window_start", "port_id"], dropna=False)
            .agg(
                avg_dwell_hours=("dwell_hours", "mean"),
                port_call_count=("call_id", "nunique"),
            )
            .reset_index()
            .rename(columns={"port_id": "zone_id"})
        )
        dwell_summary["zone_type"] = "port"
        metrics = metrics.merge(dwell_summary, how="left", on=["window_start", "zone_id", "zone_type"])
    else:
        metrics["avg_dwell_hours"] = 0.0
        metrics["port_call_count"] = 0

    fill_defaults = {
        "event_count": 0,
        "transit_count": 0,
        "anchorage_event_count": 0,
        "speed_anomaly_count": 0,
        "max_event_severity": 0.0,
        "avg_dwell_hours": 0.0,
        "port_call_count": 0,
    }
    metrics = metrics.fillna(fill_defaults).sort_values(["window_start", "zone_type", "zone_id"]).reset_index(drop=True)
    return metrics
