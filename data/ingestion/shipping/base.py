from __future__ import annotations

from abc import ABC
from datetime import datetime
from typing import Any, Dict, Optional

import pandas as pd


class ShippingSourceAdapter(ABC):
    """Interface for free/public/manual shipping-data adapters."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.name = str(self.config.get("name", self.__class__.__name__))

    def fetch_vessel_positions(
        self,
        start_time: datetime,
        end_time: datetime,
        vessel_type: Optional[str] = None,
        geography_scope: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        return pd.DataFrame()

    def fetch_vessel_events(
        self,
        start_time: datetime,
        end_time: datetime,
        geography_scope: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        return pd.DataFrame()

    def fetch_port_calls(
        self,
        start_time: datetime,
        end_time: datetime,
        port_ids: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        return pd.DataFrame()

    def fetch_port_congestion(
        self,
        start_time: datetime,
        end_time: datetime,
        port_ids: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        return pd.DataFrame()

    def fetch_route_events(
        self,
        start_time: datetime,
        end_time: datetime,
        corridor_ids: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        return pd.DataFrame()

    def fetch_weather_context(
        self,
        start_time: datetime,
        end_time: datetime,
        geography_scope: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        return pd.DataFrame()

    def fetch_satellite_context(
        self,
        start_time: datetime,
        end_time: datetime,
        geography_scope: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        return pd.DataFrame()

    def fetch_chokepoint_status(
        self,
        start_time: datetime,
        end_time: datetime,
        chokepoint_ids: Optional[list[str]] = None,
    ) -> pd.DataFrame:
        return pd.DataFrame()
