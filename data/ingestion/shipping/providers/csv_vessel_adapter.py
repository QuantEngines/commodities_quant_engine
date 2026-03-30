from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from ..base import ShippingSourceAdapter


class CSVVesselAdapter(ShippingSourceAdapter):
    def _load_frame(self, path: str) -> pd.DataFrame:
        file_path = Path(path)
        if not file_path.exists():
            return pd.DataFrame()
        if file_path.suffix == ".parquet":
            return pd.read_parquet(file_path)
        return pd.read_csv(file_path)

    def fetch_vessel_positions(
        self,
        start_time: datetime,
        end_time: datetime,
        vessel_type: Optional[str] = None,
        geography_scope: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        frames = []
        for path in (self.config.get("position_paths", {}) or {}).values():
            frame = self._load_frame(str(path))
            if not frame.empty:
                frames.append(frame)
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True)
        if "timestamp" in combined.columns:
            combined["timestamp"] = pd.to_datetime(combined["timestamp"], errors="coerce")
            combined = combined[(combined["timestamp"] >= pd.Timestamp(start_time)) & (combined["timestamp"] <= pd.Timestamp(end_time))]
        if vessel_type and "vessel_type" in combined.columns:
            combined = combined[combined["vessel_type"].astype(str).str.contains(vessel_type, case=False, na=False)]
        return combined.reset_index(drop=True)

    def fetch_port_calls(
        self,
        start_time: datetime,
        end_time: datetime,
        port_ids: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        frames = []
        for path in (self.config.get("port_call_paths", {}) or {}).values():
            frame = self._load_frame(str(path))
            if not frame.empty:
                frames.append(frame)
        if not frames:
            return pd.DataFrame()
        combined = pd.concat(frames, ignore_index=True)
        if "arrival_time" in combined.columns:
            combined["arrival_time"] = pd.to_datetime(combined["arrival_time"], errors="coerce")
            combined = combined[(combined["arrival_time"] >= pd.Timestamp(start_time)) & (combined["arrival_time"] <= pd.Timestamp(end_time))]
        if port_ids and "port_id" in combined.columns:
            combined = combined[combined["port_id"].isin(port_ids)]
        return combined.reset_index(drop=True)
