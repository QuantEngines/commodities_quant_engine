from __future__ import annotations

from datetime import datetime

import pandas as pd

from ..base import ShippingSourceAdapter


class SatelliteStubAdapter(ShippingSourceAdapter):
    def fetch_satellite_context(
        self,
        start_time: datetime,
        end_time: datetime,
        geography_scope: list[str] | None = None,
    ) -> pd.DataFrame:
        return pd.DataFrame()
