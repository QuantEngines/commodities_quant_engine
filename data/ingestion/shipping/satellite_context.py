from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from .base import ShippingSourceAdapter


class SatelliteContextIngestionSource:
    def __init__(self, adapter: ShippingSourceAdapter):
        self.adapter = adapter

    def fetch_satellite_context(
        self,
        start_time: datetime,
        end_time: datetime,
        geography_scope: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        return self.adapter.fetch_satellite_context(start_time=start_time, end_time=end_time, geography_scope=geography_scope)
