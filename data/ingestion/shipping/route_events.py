from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from ....shipping.processing.vessel_cleaning import normalize_route_events
from .base import ShippingSourceAdapter


class RouteEventIngestionSource:
    def __init__(self, adapter: ShippingSourceAdapter):
        self.adapter = adapter

    def fetch_route_events(
        self,
        start_time: datetime,
        end_time: datetime,
        corridor_ids: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        raw = self.adapter.fetch_route_events(start_time=start_time, end_time=end_time, corridor_ids=corridor_ids)
        return normalize_route_events(raw, source=self.adapter.name)

    def fetch_chokepoint_status(
        self,
        start_time: datetime,
        end_time: datetime,
        chokepoint_ids: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        return self.adapter.fetch_chokepoint_status(start_time=start_time, end_time=end_time, chokepoint_ids=chokepoint_ids)
