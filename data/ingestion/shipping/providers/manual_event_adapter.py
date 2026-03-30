from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from ..base import ShippingSourceAdapter


class ManualEventAdapter(ShippingSourceAdapter):
    def _load_frame(self, path: str) -> pd.DataFrame:
        file_path = Path(path)
        if not file_path.exists():
            return pd.DataFrame()
        if file_path.suffix == ".parquet":
            return pd.read_parquet(file_path)
        return pd.read_csv(file_path)

    def fetch_route_events(
        self,
        start_time: datetime,
        end_time: datetime,
        corridor_ids: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        frames = []
        for path in (self.config.get("event_paths", {}) or {}).values():
            frame = self._load_frame(str(path))
            if not frame.empty:
                frames.append(frame)
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True)
        if "timestamp" in combined.columns:
            combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")
            combined = combined[(combined["timestamp"] >= pd.Timestamp(start_time)) & (combined["timestamp"] <= pd.Timestamp(end_time))]
        if corridor_ids and "route_id" in combined.columns:
            combined = combined[combined["route_id"].isin(corridor_ids)]
        return combined.reset_index(drop=True)

    def fetch_chokepoint_status(
        self,
        start_time: datetime,
        end_time: datetime,
        chokepoint_ids: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        frame = self.fetch_route_events(start_time=start_time, end_time=end_time, corridor_ids=chokepoint_ids)
        if frame.empty:
            return frame
        if "chokepoint_id" in frame.columns and chokepoint_ids:
            frame = frame[frame["chokepoint_id"].isin(chokepoint_ids)]
        return frame.reset_index(drop=True)
