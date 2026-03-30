from __future__ import annotations

from typing import List

import pandas as pd

from ...config.settings import settings


def reconstruct_tracks(positions: pd.DataFrame, max_gap_hours: int | None = None) -> pd.DataFrame:
    if positions.empty:
        return positions.copy()
    max_gap_hours = max_gap_hours or settings.shipping.max_track_gap_hours
    frame = positions.sort_values(["vessel_id", "timestamp"]).copy()
    gap = frame.groupby("vessel_id")["timestamp"].diff().dt.total_seconds().div(3600.0).fillna(0.0)
    frame["track_break"] = (gap > float(max_gap_hours)).astype(int)
    frame["track_sequence"] = frame.groupby("vessel_id")["track_break"].cumsum()
    frame["track_id"] = frame["vessel_id"].astype(str) + "_track_" + frame["track_sequence"].astype(str)
    frame["time_delta_hours"] = gap
    return frame


def summarize_tracks(tracks: pd.DataFrame) -> pd.DataFrame:
    if tracks.empty:
        return pd.DataFrame(columns=["track_id", "vessel_id", "start_time", "end_time", "position_count"])
    summary = (
        tracks.groupby(["track_id", "vessel_id"], dropna=False)
        .agg(
            start_time=("timestamp", "min"),
            end_time=("timestamp", "max"),
            position_count=("timestamp", "size"),
            avg_speed_knots=("speed_knots", "mean"),
        )
        .reset_index()
    )
    return summary
