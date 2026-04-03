from __future__ import annotations

import logging
import os
from datetime import date, datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from ..models import Contract, OHLCV
from .base import DataSource

logger = logging.getLogger(__name__)


def _resolve_secret(value: Optional[str]) -> str:
    if not value:
        return ""
    token = str(value).strip()
    if token.startswith("${") and token.endswith("}"):
        return os.getenv(token[2:-1], "").strip()
    return token


class ICICIBreezeDataSource(DataSource):
    """Optional ICICI Breeze adapter for commodity historical bars."""

    exchange_code = "MCX"
    default_segment = "commodities"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = _resolve_secret(config.get("api_key", ""))
        self.secret_key = _resolve_secret(config.get("secret_key", ""))
        self.session_token = _resolve_secret(config.get("session_token", ""))
        self.product_type = str(config.get("product_type", "futures"))
        self.exchange_code = str(config.get("exchange_code", "MCX")).upper()
        self.fallback_enabled = bool(config.get("fallback_enabled", True))
        self.client = None
        self.enabled = bool(self.api_key and self.secret_key and self.session_token)

        if self.enabled:
            self._init_client()
        else:
            logger.warning(
                "ICICI Breeze disabled: missing api_key, secret_key, or session_token. "
                "Set these in .env / environment to enable live data."
            )

    def _init_client(self) -> None:
        try:
            from breeze_connect import BreezeConnect

            self.client = BreezeConnect(api_key=self.api_key)
            self.client.generate_session(api_secret=self.secret_key, session_token=self.session_token)
            logger.info("ICICI Breeze initialized successfully")
        except ImportError:
            logger.error("breeze-connect not installed. Install with: pip install breeze-connect")
            self.enabled = False
        except Exception as exc:
            logger.error(f"Failed to initialize ICICI Breeze: {exc}")
            self.enabled = False

    def fetch_contract_master(
        self, commodity: str, as_of_date: Optional[date] = None
    ) -> List[Contract]:
        as_of_date = as_of_date or date.today()
        # Breeze contract lookup differs by instrument family; use a conservative
        # synthetic contract fallback while keeping source lineage explicit.
        return [
            Contract(
                commodity=commodity,
                symbol=commodity,
                expiry_date=as_of_date,
                lot_size=1,
                tick_size=0.01,
                multiplier=1,
                exchange=self.exchange_code,
                segment=self.default_segment,
                quote_currency="INR",
                source="icici_breeze",
                metadata={"synthetic_contract": True},
            )
        ]

    def fetch_ohlcv(
        self, contract: str, start_date: date, end_date: date
    ) -> List[OHLCV]:
        if not self.enabled or not self.client:
            return []

        try:
            # Breeze payload shape can vary by SDK version; handle common variants.
            response = self.client.get_historical_data_v2(
                interval="1day",
                from_date=f"{start_date.isoformat()}T00:00:00.000Z",
                to_date=f"{end_date.isoformat()}T23:59:59.000Z",
                stock_code=contract,
                exchange_code=self.exchange_code,
                product_type=self.product_type,
            )
        except Exception:
            try:
                response = self.client.get_historical_data(
                    interval="1day",
                    from_date=f"{start_date.isoformat()}T00:00:00.000Z",
                    to_date=f"{end_date.isoformat()}T23:59:59.000Z",
                    stock_code=contract,
                    exchange_code=self.exchange_code,
                    product_type=self.product_type,
                )
            except Exception as exc:
                logger.error(f"Error fetching Breeze OHLCV for {contract}: {exc}")
                return []

        rows = []
        if isinstance(response, dict):
            rows = response.get("Success") or response.get("success") or response.get("data") or []
        elif isinstance(response, list):
            rows = response

        parsed: List[OHLCV] = []
        for row in rows:
            try:
                ts_raw = row.get("datetime") or row.get("timestamp") or row.get("date")
                ts = pd.Timestamp(ts_raw).to_pydatetime()
                parsed.append(
                    OHLCV(
                        timestamp=ts,
                        open=float(row.get("open", row.get("Open", 0.0))),
                        high=float(row.get("high", row.get("High", 0.0))),
                        low=float(row.get("low", row.get("Low", 0.0))),
                        close=float(row.get("close", row.get("Close", 0.0))),
                        volume=int(float(row.get("volume", row.get("Volume", 0.0)) or 0.0)),
                        open_interest=(
                            int(float(row.get("open_interest", row.get("oi", 0.0)) or 0.0))
                            if row.get("open_interest") is not None or row.get("oi") is not None
                            else None
                        ),
                        contract=contract,
                    )
                )
            except Exception:
                continue
        return parsed

    def fetch_open_interest(
        self, contract: str, start_date: date, end_date: date
    ) -> pd.DataFrame:
        points = self.fetch_ohlcv(contract, start_date, end_date)
        if not points:
            return pd.DataFrame()
        frame = pd.DataFrame(
            [
                {"timestamp": p.timestamp, "open_interest": p.open_interest}
                for p in points
                if p.open_interest is not None
            ]
        )
        if frame.empty:
            return frame
        return frame.set_index("timestamp")

    def fetch_spot(
        self, commodity: str, start_date: date, end_date: date
    ) -> pd.DataFrame:
        points = self.fetch_ohlcv(commodity, start_date, end_date)
        if not points:
            return pd.DataFrame()
        frame = pd.DataFrame(
            [{"timestamp": p.timestamp, "price": p.close, "volume": p.volume} for p in points]
        )
        return frame.set_index("timestamp") if not frame.empty else frame

    def fetch_reference_rates(
        self, currency_pair: str, start_date: date, end_date: date
    ) -> pd.DataFrame:
        logger.debug(f"Breeze does not expose FX reference rates for {currency_pair}")
        return pd.DataFrame()

    def fetch_macro_series(
        self, series_name: str, start_date: date, end_date: date
    ) -> pd.DataFrame:
        logger.debug(f"Breeze does not expose macro series for {series_name}")
        return pd.DataFrame()

    def fetch_weather(
        self, location: str, start_date: date, end_date: date
    ) -> pd.DataFrame:
        logger.debug(f"Breeze does not expose weather data for {location}")
        return pd.DataFrame()

    def fetch_calendar(self, exchange: str, year: int) -> List[date]:
        business_days = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="B")
        return [ts.date() for ts in business_days]
