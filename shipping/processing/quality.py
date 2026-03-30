from __future__ import annotations

import pandas as pd


def compute_daily_quality(
    positions: pd.DataFrame,
    events: pd.DataFrame | None = None,
    expected_daily_position_updates: int = 6,
) -> pd.DataFrame:
    if positions.empty:
        return pd.DataFrame(columns=["window_start", "shipping_data_quality_score", "shipping_data_quality_penalty"])
    frame = positions.copy()
    frame["window_start"] = pd.to_datetime(frame["timestamp"]).dt.floor("1D")
    quality = (
        frame.groupby("window_start", dropna=False)
        .agg(
            vessel_count=("vessel_id", "nunique"),
            observation_count=("timestamp", "size"),
            completeness=("data_quality_score", "mean"),
        )
        .reset_index()
    )
    expected = quality["vessel_count"].clip(lower=1) * float(expected_daily_position_updates)
    coverage_ratio = (quality["observation_count"] / expected).clip(lower=0.0, upper=1.0)
    quality["shipping_data_quality_score"] = (0.55 * quality["completeness"].fillna(0.0) + 0.45 * coverage_ratio).clip(0.0, 1.0)
    if events is not None and not events.empty:
        event_frame = events.copy()
        event_frame["window_start"] = pd.to_datetime(event_frame["timestamp"]).dt.floor("1D")
        anomaly = event_frame.groupby("window_start").agg(speed_anomaly_count=("event_id", "size")).reset_index()
        quality = quality.merge(anomaly, how="left", on="window_start")
        quality["speed_anomaly_count"] = quality["speed_anomaly_count"].fillna(0.0)
        quality["shipping_data_quality_score"] = (
            quality["shipping_data_quality_score"] - quality["speed_anomaly_count"].clip(upper=5.0) * 0.02
        ).clip(0.0, 1.0)
    quality["shipping_data_quality_penalty"] = (1.0 - quality["shipping_data_quality_score"]).clip(0.0, 1.0)
    return quality[["window_start", "shipping_data_quality_score", "shipping_data_quality_penalty"]]
