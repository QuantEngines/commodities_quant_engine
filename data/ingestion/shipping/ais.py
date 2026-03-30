from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from ....shipping.processing.vessel_cleaning import normalize_vessel_positions
from .base import ShippingSourceAdapter


class AISIngestionSource:
    def __init__(self, adapter: ShippingSourceAdapter):
        self.adapter = adapter

    def fetch_vessel_positions(
        self,
        start_time: datetime,
        end_time: datetime,
        vessel_type: Optional[str] = None,
        geography_scope: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        raw = self.adapter.fetch_vessel_positions(
            start_time=start_time,
            end_time=end_time,
            vessel_type=vessel_type,
            geography_scope=geography_scope,
        )
        return normalize_vessel_positions(raw, source=self.adapter.name)
