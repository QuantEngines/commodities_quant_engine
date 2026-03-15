from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd

from ...config.settings import settings
from ..contract_master.manager import contract_master
from ..models import OHLCV
from .provider_registry import ProviderRegistry, provider_registry


class MarketDataService:
    """Convenience service for loading standardized market data from configured providers."""

    def __init__(self, registry: Optional[ProviderRegistry] = None):
        self.registry = registry or provider_registry

    def load_price_frame(
        self,
        commodity: str,
        start_date: date,
        end_date: date,
        contract: Optional[str] = None,
        exchange: Optional[str] = None,
    ) -> pd.DataFrame:
        commodity_config = settings.commodities[commodity]
        provider = self.registry.get(exchange or commodity_config.exchange)
        if provider is None:
            raise ValueError(f"No provider configured for exchange {exchange or commodity_config.exchange}.")
        active_contract = contract or (contract_master.get_active_contract(commodity, end_date).symbol if contract_master.get_active_contract(commodity, end_date) else commodity)
        rows = provider.fetch_ohlcv(active_contract, start_date, end_date)
        if not rows:
            return pd.DataFrame()
        return self._rows_to_frame(rows)

    def _rows_to_frame(self, rows: list[OHLCV]) -> pd.DataFrame:
        frame = pd.DataFrame([row.to_dict() for row in rows])
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        return frame.set_index("timestamp").sort_index()


market_data_service = MarketDataService()
