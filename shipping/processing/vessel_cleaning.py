from __future__ import annotations

from typing import Dict, Iterable, List, Mapping, Optional

import pandas as pd


POSITION_ALIASES = {
    "mmsi": "vessel_id",
    "imo": "vessel_id",
    "ship_id": "vessel_id",
    "ts": "timestamp",
    "time": "timestamp",
    "lat": "latitude",
    "lon": "longitude",
    "lng": "longitude",
    "sog": "speed_knots",
    "speed": "speed_knots",
    "heading": "heading_degrees",
    "course": "course_degrees",
    "nav_status": "navigation_status",
}

PORT_CALL_ALIASES = {
    "arrival": "arrival_time",
    "departure": "departure_time",
    "port": "port_id",
    "mmsi": "vessel_id",
}

ROUTE_EVENT_ALIASES = {
    "route": "route_id",
    "time": "timestamp",
    "event": "event_type",
}


def _as_frame(records: Iterable[Mapping[str, object]] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(records, pd.DataFrame):
        return records.copy()
    return pd.DataFrame(list(records))


def _normalize_timestamp_series(series: pd.Series) -> pd.Series:
    timestamps = pd.to_datetime(series, utc=True, errors="coerce")
    return timestamps.dt.tz_convert("UTC").dt.tz_localize(None)


def _quality_score_frame(frame: pd.DataFrame, important_columns: List[str]) -> pd.Series:
    available = frame[important_columns].notna().mean(axis=1)
    return available.clip(lower=0.0, upper=1.0)


def normalize_vessel_positions(
    records: Iterable[Mapping[str, object]] | pd.DataFrame,
    source: str = "unknown",
    commodity_tags: Optional[List[str]] = None,
) -> pd.DataFrame:
    frame = _as_frame(records)
    frame = frame.rename(columns={key: value for key, value in POSITION_ALIASES.items() if key in frame.columns})
    if frame.empty:
        columns = [
            "vessel_id",
            "timestamp",
            "latitude",
            "longitude",
            "speed_knots",
            "course_degrees",
            "heading_degrees",
            "vessel_type",
            "cargo_class",
            "source",
            "data_quality_score",
            "commodity_tags",
        ]
        return pd.DataFrame(columns=columns)
    required = {"vessel_id", "timestamp", "latitude", "longitude"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required vessel position columns: {sorted(missing)}")
    frame["timestamp"] = _normalize_timestamp_series(frame["timestamp"])
    frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
    frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
    for optional in ("speed_knots", "course_degrees", "heading_degrees", "draught_meters"):
        if optional in frame.columns:
            frame[optional] = pd.to_numeric(frame[optional], errors="coerce")
        else:
            frame[optional] = pd.NA
    for optional in ("vessel_type", "cargo_class", "navigation_status", "source_record_id"):
        if optional not in frame.columns:
            frame[optional] = pd.NA
    frame = frame.dropna(subset=["vessel_id", "timestamp", "latitude", "longitude"])
    frame = frame[
        frame["latitude"].between(-90.0, 90.0)
        & frame["longitude"].between(-180.0, 180.0)
    ].copy()
    frame["source"] = source
    frame["commodity_tags"] = [list(commodity_tags or []) for _ in range(len(frame))]
    frame["data_quality_score"] = _quality_score_frame(
        frame,
        important_columns=["vessel_id", "timestamp", "latitude", "longitude", "speed_knots", "vessel_type"],
    )
    frame = frame.sort_values(["vessel_id", "timestamp", "latitude", "longitude"]).drop_duplicates(
        subset=["vessel_id", "timestamp", "latitude", "longitude"],
        keep="last",
    )
    return frame.reset_index(drop=True)


def normalize_port_calls(records: Iterable[Mapping[str, object]] | pd.DataFrame, source: str = "manual") -> pd.DataFrame:
    frame = _as_frame(records)
    frame = frame.rename(columns={key: value for key, value in PORT_CALL_ALIASES.items() if key in frame.columns})
    if frame.empty:
        return pd.DataFrame(columns=["call_id", "vessel_id", "port_id", "arrival_time", "departure_time", "source"])
    required = {"vessel_id", "port_id", "arrival_time"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required port call columns: {sorted(missing)}")
    frame["arrival_time"] = _normalize_timestamp_series(frame["arrival_time"])
    if "departure_time" in frame.columns:
        frame["departure_time"] = _normalize_timestamp_series(frame["departure_time"])
    else:
        frame["departure_time"] = pd.NaT
    if "call_id" not in frame.columns:
        frame["call_id"] = [f"port_call_{idx}" for idx in range(len(frame))]
    frame["source"] = source
    frame["data_quality_score"] = _quality_score_frame(frame, important_columns=["vessel_id", "port_id", "arrival_time", "departure_time"])
    return frame.sort_values(["vessel_id", "arrival_time"]).reset_index(drop=True)


def normalize_route_events(records: Iterable[Mapping[str, object]] | pd.DataFrame, source: str = "manual") -> pd.DataFrame:
    frame = _as_frame(records)
    frame = frame.rename(columns={key: value for key, value in ROUTE_EVENT_ALIASES.items() if key in frame.columns})
    if frame.empty:
        return pd.DataFrame(columns=["event_id", "route_id", "timestamp", "event_type", "severity", "source"])
    required = {"route_id", "timestamp", "event_type"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required route event columns: {sorted(missing)}")
    frame["timestamp"] = _normalize_timestamp_series(frame["timestamp"])
    frame["severity"] = pd.to_numeric(frame.get("severity", 0.0), errors="coerce").fillna(0.0)
    if "event_id" not in frame.columns:
        frame["event_id"] = [f"route_event_{idx}" for idx in range(len(frame))]
    frame["source"] = source
    frame["data_quality_score"] = _quality_score_frame(frame, important_columns=["route_id", "timestamp", "event_type", "severity"])
    return frame.sort_values(["timestamp", "route_id"]).reset_index(drop=True)
