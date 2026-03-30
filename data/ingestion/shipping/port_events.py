from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from ....shipping.processing.vessel_cleaning import normalize_port_calls
from .base import ShippingSourceAdapter


class PortEventIngestionSource:
    def __init__(self, adapter: ShippingSourceAdapter):
        self.adapter = adapter

    def fetch_port_calls(
        self,
        start_time: datetime,
        end_time: datetime,
        port_ids: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        raw = self.adapter.fetch_port_calls(start_time=start_time, end_time=end_time, port_ids=port_ids)
        return normalize_port_calls(raw, source=self.adapter.name)

    def fetch_port_congestion(
        self,
        start_time: datetime,
        end_time: datetime,
        port_ids: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        return self.adapter.fetch_port_congestion(start_time=start_time, end_time=end_time, port_ids=port_ids)
