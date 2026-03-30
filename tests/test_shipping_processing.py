from datetime import datetime

import pandas as pd

from ..shipping.geospatial import ZoneCatalog
from ..shipping.processing import (
    aggregate_shipping_metrics,
    compute_daily_quality,
    detect_vessel_events,
    normalize_vessel_positions,
    reconstruct_tracks,
)


def test_normalize_vessel_positions_handles_aliases_and_timezone_normalization():
    records = [
        {
            "mmsi": "123456789",
            "ts": "2026-03-01T00:00:00+05:30",
            "lat": "25.16",
            "lon": "56.31",
            "sog": "0.8",
            "vessel_type": "Crude Tanker",
        }
    ]

    frame = normalize_vessel_positions(records, source="csv_vessel")

    assert list(frame["vessel_id"]) == ["123456789"]
    assert frame.loc[0, "timestamp"] == datetime(2026, 2, 28, 18, 30)
    assert float(frame.loc[0, "latitude"]) == 25.16
    assert frame.loc[0, "source"] == "csv_vessel"
    assert 0.0 <= float(frame.loc[0, "data_quality_score"]) <= 1.0


def test_event_detection_and_aggregation_capture_anchorage_and_transit_signals():
    catalog = ZoneCatalog.from_config()
    records = [
        {"vessel_id": "tanker-1", "timestamp": "2026-03-01T00:00:00Z", "latitude": 25.16, "longitude": 56.31, "speed_knots": 0.4, "cargo_class": "crude_tanker"},
        {"vessel_id": "tanker-1", "timestamp": "2026-03-01T10:00:00Z", "latitude": 25.17, "longitude": 56.32, "speed_knots": 0.5, "cargo_class": "crude_tanker"},
        {"vessel_id": "tanker-1", "timestamp": "2026-03-01T20:00:00Z", "latitude": 25.18, "longitude": 56.33, "speed_knots": 0.4, "cargo_class": "crude_tanker"},
        {"vessel_id": "tanker-2", "timestamp": "2026-03-01T08:00:00Z", "latitude": 26.10, "longitude": 56.90, "speed_knots": 11.0, "cargo_class": "crude_tanker"},
    ]

    positions = normalize_vessel_positions(records, source="test")
    tracks = reconstruct_tracks(positions)
    events = detect_vessel_events(tracks, catalog)
    metrics = aggregate_shipping_metrics(tracks, events, catalog)

    assert not events.empty
    assert "anchorage_buildup" in set(events["event_type"])
    assert not metrics.empty
    assert metrics["vessel_count"].sum() >= 2
    assert metrics["transit_count"].max() >= 0


def test_quality_scoring_penalizes_sparse_observation_days():
    rich = normalize_vessel_positions(
        [
            {"vessel_id": "v1", "timestamp": "2026-03-01T00:00:00Z", "latitude": 25.16, "longitude": 56.31},
            {"vessel_id": "v1", "timestamp": "2026-03-01T06:00:00Z", "latitude": 25.17, "longitude": 56.32},
            {"vessel_id": "v1", "timestamp": "2026-03-01T12:00:00Z", "latitude": 25.18, "longitude": 56.33},
            {"vessel_id": "v1", "timestamp": "2026-03-01T18:00:00Z", "latitude": 25.19, "longitude": 56.34},
        ]
    )
    sparse = normalize_vessel_positions(
        [{"vessel_id": "v2", "timestamp": "2026-03-02T00:00:00Z", "latitude": 25.16, "longitude": 56.31}]
    )
    quality = compute_daily_quality(pd.concat([rich, sparse], ignore_index=True))

    rich_score = float(quality.loc[quality["window_start"] == pd.Timestamp("2026-03-01"), "shipping_data_quality_score"].iloc[0])
    sparse_score = float(quality.loc[quality["window_start"] == pd.Timestamp("2026-03-02"), "shipping_data_quality_score"].iloc[0])

    assert rich_score > sparse_score
