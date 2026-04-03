from __future__ import annotations

from datetime import date
from typing import Optional

import pandas as pd

from ...config.settings import settings
from ..contract_master.manager import contract_master
from ..models import OHLCV
from ..storage.local import LocalStorage
from .provider_registry import ProviderRegistry, provider_registry


class MarketDataService:
    """Convenience service for loading standardized market data from configured providers."""

    def __init__(self, registry: Optional[ProviderRegistry] = None, storage: Optional[LocalStorage] = None):
        self.registry = registry or provider_registry
        self.storage = storage or LocalStorage()

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

    def cache_key(
        self,
        commodity: str,
        start_date: date,
        end_date: date,
        contract: Optional[str] = None,
        exchange: Optional[str] = None,
    ) -> str:
        venue = (exchange or settings.commodities[commodity].exchange).upper()
        symbol = (contract or commodity).upper()
        return f"{commodity.upper()}_{venue}_{symbol}_{start_date.isoformat()}_{end_date.isoformat()}"

    def cache_price_frame(
        self,
        commodity: str,
        price_frame: pd.DataFrame,
        start_date: date,
        end_date: date,
        contract: Optional[str] = None,
        exchange: Optional[str] = None,
    ) -> str:
        if price_frame.empty:
            raise ValueError("Cannot cache an empty price frame.")
        cache_key = self.cache_key(
            commodity=commodity,
            start_date=start_date,
            end_date=end_date,
            contract=contract,
            exchange=exchange,
        )
        payload = price_frame.reset_index().copy()
        path = self.storage.resolve(settings.storage.market_data_store, f"{cache_key}.parquet")
        payload.to_parquet(path, index=False)
        return str(path)

    def load_cached_price_frame(
        self,
        commodity: str,
        start_date: date,
        end_date: date,
        contract: Optional[str] = None,
        exchange: Optional[str] = None,
    ) -> pd.DataFrame:
        cache_key = self.cache_key(
            commodity=commodity,
            start_date=start_date,
            end_date=end_date,
            contract=contract,
            exchange=exchange,
        )
        path = self.storage.resolve(settings.storage.market_data_store, f"{cache_key}.parquet")
        if not path.exists():
            return pd.DataFrame()
        frame = pd.read_parquet(path)
        if "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"])
            frame = frame.set_index("timestamp")
        return frame.sort_index()

    def load_or_fetch_price_frame(
        self,
        commodity: str,
        start_date: date,
        end_date: date,
        contract: Optional[str] = None,
        exchange: Optional[str] = None,
        refresh: bool = False,
        persist: bool = True,
    ) -> pd.DataFrame:
        if not refresh:
            cached = self.load_cached_price_frame(
                commodity=commodity,
                start_date=start_date,
                end_date=end_date,
                contract=contract,
                exchange=exchange,
            )
            if not cached.empty:
                return cached

        frame = self.load_price_frame(
            commodity=commodity,
            start_date=start_date,
            end_date=end_date,
            contract=contract,
            exchange=exchange,
        )
        if persist and not frame.empty:
            self.cache_price_frame(
                commodity=commodity,
                price_frame=frame,
                start_date=start_date,
                end_date=end_date,
                contract=contract,
                exchange=exchange,
            )
        return frame


market_data_service = MarketDataService()
