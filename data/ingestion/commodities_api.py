from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from .local_first import LocalFirstDataSource
from ..models import OHLCV


class CommoditiesAPIDataSource(LocalFirstDataSource):
    """Optional reference-data adapter for commodities-api.com.

    This adapter is intentionally opt-in and mapping-driven:
    - local catalog data remains the first choice
    - users must provide an API key
    - users should provide explicit symbol mappings for the commodities they want
    - returned bars are reference/spot-style daily bars, not native MCX futures bars
    """

    exchange_code = "COMMODITIES_API"
    default_segment = "reference_data"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.base_url = (config.get("base_url") or "https://api.commodities-api.com/api").rstrip("/")
        self.api_key = config.get("api_key")
        self.symbol_map = {str(key).upper(): str(value).upper() for key, value in (config.get("symbol_map") or {}).items()}
        self.base_currency = str(config.get("base_currency", "USD")).upper()
        self.quote_currency = str(config.get("quote_currency", "USD")).upper()
        self.timeout = int(config.get("timeout", 20))
        self.invert_quotes = bool(config.get("invert_quotes", True))
        self.session = requests.Session()

    def fetch_ohlcv(self, contract: str, start_date: date, end_date: date) -> List[OHLCV]:
        local_rows = super().fetch_ohlcv(contract, start_date, end_date)
        if local_rows:
            return local_rows
        commodity = self._commodity_from_contract(contract)
        frame = self._fetch_reference_price_frame(commodity, start_date, end_date)
        if frame.empty:
            return []
        return [
            OHLCV(
                timestamp=index.to_pydatetime(),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=int(row.get("volume", 0)),
                open_interest=None,
                contract=contract,
            )
            for index, row in frame.iterrows()
        ]

    def fetch_spot(self, commodity: str, start_date: date, end_date: date) -> pd.DataFrame:
        local_frame = super().fetch_spot(commodity, start_date, end_date)
        if not local_frame.empty:
            return local_frame
        return self._fetch_reference_price_frame(commodity, start_date, end_date)

    def fetch_reference_rates(self, currency_pair: str, start_date: date, end_date: date) -> pd.DataFrame:
        local_frame = super().fetch_reference_rates(currency_pair, start_date, end_date)
        if not local_frame.empty:
            return local_frame
        base, _, quote = currency_pair.upper().partition("/")
        if not (base and quote and self.api_key):
            return pd.DataFrame()
        payload = self._request_json(
            "timeseries",
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            base=base,
            symbols=quote,
        )
        if payload is None:
            return pd.DataFrame()
        rates = self._extract_rates(payload)
        records = []
        for timestamp_key, values in sorted(rates.items()):
            price = self._extract_price(values, quote, invert_quotes=False)
            if price is None:
                continue
            records.append({"timestamp": pd.Timestamp(timestamp_key), "value": price})
        if not records:
            return pd.DataFrame()
        frame = pd.DataFrame(records).set_index("timestamp").sort_index()
        return frame.loc[pd.Timestamp(start_date) : pd.Timestamp(end_date)]

    def _fetch_reference_price_frame(self, commodity: str, start_date: date, end_date: date) -> pd.DataFrame:
        symbol = self.symbol_map.get(commodity.upper())
        if not symbol or not self.api_key:
            return pd.DataFrame()
        payload = self._request_json(
            "timeseries",
            start_date=start_date.isoformat(),
            end_date=end_date.isoformat(),
            base=self.base_currency,
            symbols=symbol,
        )
        if payload is None:
            return pd.DataFrame()
        rates = self._extract_rates(payload)
        rows: List[Dict[str, object]] = []
        for timestamp_key, values in sorted(rates.items()):
            close = self._extract_price(values, symbol, invert_quotes=self.invert_quotes)
            if close is None:
                continue
            rows.append(
                {
                    "timestamp": pd.Timestamp(timestamp_key),
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "volume": 0,
                    "source": "COMMODITIES_API",
                }
            )
        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows).set_index("timestamp").sort_index()
        return frame.loc[pd.Timestamp(start_date) : pd.Timestamp(end_date)]

    def _request_json(self, endpoint: str, **params: Any) -> Optional[Dict[str, Any]]:
        if not self.api_key:
            return None
        query = {"access_key": self.api_key, **params}
        try:
            response = self._retry_request(self.session.get, f"{self.base_url}/{endpoint}", params=query, timeout=self.timeout)
            response.raise_for_status()
            payload = response.json()
        except Exception:
            return None
        if isinstance(payload, dict) and payload.get("success") is False:
            return None
        return payload

    def _extract_rates(self, payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        container = payload.get("data") if isinstance(payload.get("data"), dict) else payload
        rates = container.get("rates", {}) if isinstance(container, dict) else {}
        return rates if isinstance(rates, dict) else {}

    def _extract_price(self, values: Any, symbol: str, invert_quotes: bool) -> Optional[float]:
        if not isinstance(values, dict):
            return None
        raw = values.get(symbol)
        if raw in (None, 0):
            return None
        try:
            numeric = float(raw)
        except (TypeError, ValueError):
            return None
        if invert_quotes:
            return 1.0 / numeric if numeric else None
        return numeric
