from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .base import DataSource
from ..models import Contract, OHLCV


class LocalFirstDataSource(DataSource):
    """Free/local-first data source with optional public-provider fallback."""

    exchange_code: str = ""
    default_segment: str = "generic"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.local_catalog = config.get("local_catalog", {})
        self.url_catalog = config.get("url_catalog", {})
        self.free_tickers = config.get("free_tickers", {})
        self.allow_yfinance = bool(config.get("allow_yfinance", False))
        self.calendar_holidays = set(pd.to_datetime(config.get("holidays", [])).date)

    def fetch_contract_master(self, commodity: str, as_of_date: Optional[date] = None) -> List[Contract]:
        as_of_date = as_of_date or date.today()
        contracts = []
        for offset in range(3):
            anchor = self._month_anchor(as_of_date, offset)
            symbol = f"{commodity}{anchor.strftime('%b').upper()}{anchor.year % 100:02d}"
            contracts.append(
                Contract(
                    commodity=commodity,
                    symbol=symbol,
                    expiry_date=anchor + timedelta(days=27),
                    lot_size=int(self.config.get("lot_sizes", {}).get(commodity, 1)),
                    tick_size=float(self.config.get("tick_sizes", {}).get(commodity, 0.01)),
                    multiplier=int(self.config.get("multipliers", {}).get(commodity, 1)),
                    exchange=self.exchange_code or self.config.get("exchange", "LOCAL"),
                    segment=self.default_segment,
                )
            )
        return contracts

    def fetch_ohlcv(self, contract: str, start_date: date, end_date: date) -> List[OHLCV]:
        commodity = self._commodity_from_contract(contract)
        frame = self._load_market_frame([contract, commodity, f"ohlcv:{contract}", f"ohlcv:{commodity}"], start_date, end_date)
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
                open_interest=int(row["open_interest"]) if pd.notna(row.get("open_interest")) else None,
                contract=contract,
            )
            for index, row in frame.iterrows()
        ]

    def fetch_open_interest(self, contract: str, start_date: date, end_date: date) -> pd.DataFrame:
        commodity = self._commodity_from_contract(contract)
        frame = self._load_market_frame([f"open_interest:{contract}", f"open_interest:{commodity}", contract, commodity], start_date, end_date)
        if "open_interest" in frame.columns:
            return frame[["open_interest"]].copy()
        return pd.DataFrame(index=frame.index)

    def fetch_spot(self, commodity: str, start_date: date, end_date: date) -> pd.DataFrame:
        return self._load_market_frame([f"spot:{commodity}", commodity], start_date, end_date)

    def fetch_reference_rates(self, currency_pair: str, start_date: date, end_date: date) -> pd.DataFrame:
        return self._load_series_frame([currency_pair, f"fx:{currency_pair}", f"FX_{currency_pair.replace('/', '_')}"], start_date, end_date)

    def fetch_macro_series(self, series_name: str, start_date: date, end_date: date) -> pd.DataFrame:
        return self._load_series_frame([series_name], start_date, end_date)

    def fetch_weather(self, location: str, start_date: date, end_date: date) -> pd.DataFrame:
        return self._load_series_frame([f"weather:{location}", location], start_date, end_date)

    def fetch_calendar(self, exchange: str, year: int) -> List[date]:
        business_days = pd.date_range(start=f"{year}-01-01", end=f"{year}-12-31", freq="B")
        return [ts.date() for ts in business_days if ts.date() not in self.calendar_holidays]

    def _load_market_frame(self, keys: List[str], start_date: date, end_date: date) -> pd.DataFrame:
        frame = self._load_any_frame(keys)
        if frame.empty:
            ticker = self._first_ticker(keys)
            if ticker and self.allow_yfinance:
                frame = self._download_yfinance_frame(ticker, start_date, end_date)
        return self._filter_and_standardize_market_frame(frame, start_date, end_date)

    def _load_series_frame(self, keys: List[str], start_date: date, end_date: date) -> pd.DataFrame:
        frame = self._load_any_frame(keys)
        if frame.empty:
            ticker = self._first_ticker(keys)
            if ticker and self.allow_yfinance:
                frame = self._download_yfinance_frame(ticker, start_date, end_date)
        if frame.empty:
            return pd.DataFrame()
        frame = frame.copy()
        if "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"])
            frame = frame.set_index("timestamp")
        elif not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError("Series data must include a timestamp column or DatetimeIndex.")
        return frame.sort_index().loc[pd.Timestamp(start_date) : pd.Timestamp(end_date)]

    def _load_any_frame(self, keys: List[str]) -> pd.DataFrame:
        for key in keys:
            if key in self.local_catalog:
                return self._read_tabular(self.local_catalog[key])
            if key in self.url_catalog:
                return self._read_tabular(self.url_catalog[key])
        return pd.DataFrame()

    def _read_tabular(self, location: str) -> pd.DataFrame:
        path = str(location)
        if path.startswith("http://") or path.startswith("https://"):
            if path.endswith(".parquet"):
                return pd.read_parquet(path)
            return pd.read_csv(path)
        file_path = Path(path)
        if not file_path.exists():
            return pd.DataFrame()
        if file_path.suffix == ".parquet":
            return pd.read_parquet(file_path)
        if file_path.suffix in {".csv", ".txt"}:
            return pd.read_csv(file_path)
        if file_path.suffix == ".json":
            return pd.read_json(file_path)
        raise ValueError(f"Unsupported tabular source: {location}")

    def _filter_and_standardize_market_frame(self, frame: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
        if frame.empty:
            return frame
        frame = frame.copy()
        rename_map = {column: column.lower() for column in frame.columns}
        frame = frame.rename(columns=rename_map)
        if "date" in frame.columns and "timestamp" not in frame.columns:
            frame = frame.rename(columns={"date": "timestamp"})
        if "datetime" in frame.columns and "timestamp" not in frame.columns:
            frame = frame.rename(columns={"datetime": "timestamp"})
        if "adj close" in frame.columns and "close" not in frame.columns:
            frame = frame.rename(columns={"adj close": "close"})
        if "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"])
            frame = frame.set_index("timestamp")
        elif not isinstance(frame.index, pd.DatetimeIndex):
            raise ValueError("OHLCV data must include a timestamp column or DatetimeIndex.")
        required = ["open", "high", "low", "close"]
        for column in required:
            if column not in frame.columns:
                raise ValueError(f"Missing required OHLCV column: {column}")
        if "volume" not in frame.columns:
            frame["volume"] = 0
        frame = frame.sort_index()
        return frame.loc[pd.Timestamp(start_date) : pd.Timestamp(end_date)]

    def _download_yfinance_frame(self, ticker: str, start_date: date, end_date: date) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError:
            return pd.DataFrame()
        downloaded = yf.download(ticker, start=start_date, end=end_date + timedelta(days=1), progress=False, auto_adjust=False)
        if downloaded.empty:
            return pd.DataFrame()
        downloaded = downloaded.reset_index().rename(columns={"Date": "timestamp"})
        return downloaded

    def _first_ticker(self, keys: List[str]) -> Optional[str]:
        for key in keys:
            ticker = self.free_tickers.get(key)
            if ticker:
                return ticker
        return None

    def _commodity_from_contract(self, contract: str) -> str:
        alpha = "".join(character for character in contract if character.isalpha()).upper()
        for month_code in ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"):
            if alpha.endswith(month_code):
                return alpha[: -len(month_code)]
        return alpha

    def _month_anchor(self, as_of_date: date, offset: int) -> date:
        month = as_of_date.month - 1 + offset
        year = as_of_date.year + month // 12
        month = month % 12 + 1
        return date(year, month, 1)

    def as_dict_records(self, rows: List[OHLCV]) -> List[Dict[str, Any]]:
        return [asdict(row) for row in rows]
