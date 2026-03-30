from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd

from ..base import ShippingSourceAdapter


class GenericAISAdapter(ShippingSourceAdapter):
    """Public/free AIS adapter stub.

    Concrete endpoint details are intentionally left to local configuration so the
    repository does not assume a specific vendor or undocumented API contract.
    """

    def fetch_vessel_positions(
        self,
        start_time: datetime,
        end_time: datetime,
        vessel_type: Optional[str] = None,
        geography_scope: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        return pd.DataFrame()
