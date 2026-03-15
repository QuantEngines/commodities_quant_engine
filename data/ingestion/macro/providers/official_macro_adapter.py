from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ..base import MacroDataSource
from ....models import MacroEvent, MacroSeries, NewsItem


class OfficialMacroAdapter(MacroDataSource):
    """Official/public macro adapter driven by free local files or public URLs."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.series_catalog = config.get("series_catalog", {})
        self.event_calendar_catalog = config.get("event_calendar_catalog", {})
        self.fred_series = config.get("fred_series", {})
        self.allow_fred = bool(config.get("allow_fred", False))

    def fetch_macro_series(self, series_name: str, start_date: date, end_date: date) -> List[MacroSeries]:
        frame = self._load_series_frame(series_name)
        if frame.empty and self.allow_fred and series_name in self.fred_series:
            frame = self._fetch_fred_series(series_name, start_date, end_date)
        if frame.empty:
            return []
        frame = frame.loc[pd.Timestamp(start_date) : pd.Timestamp(end_date)]
        unit = self.series_catalog.get(series_name, {}).get("unit", "level")
        frequency = self.series_catalog.get(series_name, {}).get("frequency", "daily")
        source = self.series_catalog.get(series_name, {}).get("source", "official")
        return self.standardize_macro_series(frame.reset_index().rename(columns={"index": "timestamp", "value": "value"}), series_name, unit, frequency, source)

    def fetch_macro_release_calendar(self, country: str, start_date: date, end_date: date) -> List[MacroEvent]:
        frame = self._load_event_calendar(country)
        if frame.empty:
            frame = self._fallback_calendar(country, start_date, end_date)
        if frame.empty:
            return []
        frame = frame.loc[(frame["timestamp"] >= pd.Timestamp(start_date)) & (frame["timestamp"] <= pd.Timestamp(end_date))].copy()
        events = []
        for row in frame.to_dict(orient="records"):
            events.append(
                MacroEvent(
                    event_id=str(row.get("event_id", f"{country}_{row['event_type']}_{pd.Timestamp(row['timestamp']).strftime('%Y%m%d')}")),
                    event_type=str(row["event_type"]),
                    country=country,
                    timestamp=pd.Timestamp(row["timestamp"]).to_pydatetime(),
                    title=str(row.get("title", row["event_type"])),
                    description=row.get("description"),
                    expected_impact=str(row.get("expected_impact", "medium")),
                    source=str(row.get("source", "official")),
                    metadata=row.get("metadata", {}),
                )
            )
        return events

    def fetch_policy_rates(self, country: str, start_date: date, end_date: date) -> List[MacroSeries]:
        mapping = {"IN": "IN_RBI_RATE", "US": "US_FED_RATE"}
        if country not in mapping:
            return []
        return self.fetch_macro_series(mapping[country], start_date, end_date)

    def fetch_yields(self, country: str, tenor: str, start_date: date, end_date: date) -> List[MacroSeries]:
        return self.fetch_macro_series(f"{country}_YIELD_{tenor.upper()}", start_date, end_date)

    def fetch_fx_reference(self, currency_pair: str, start_date: date, end_date: date) -> List[MacroSeries]:
        return self.fetch_macro_series(f"FX_{currency_pair.replace('/', '_')}", start_date, end_date)

    def fetch_news_headlines(self, topics: List[str], start_date: date, end_date: date) -> List[NewsItem]:
        return []

    def fetch_news_articles(self, news_ids: List[str]) -> List[NewsItem]:
        return []

    def fetch_macro_events(self, country: str, event_types: List[str], start_date: date, end_date: date) -> List[MacroEvent]:
        return [event for event in self.fetch_macro_release_calendar(country, start_date, end_date) if event.event_type in event_types]

    def fetch_contract_master(self, commodity: str, as_of_date: Optional[date] = None):
        raise NotImplementedError("OfficialMacroAdapter does not serve contract metadata.")

    def fetch_ohlcv(self, contract: str, start_date: date, end_date: date):
        raise NotImplementedError("OfficialMacroAdapter does not serve OHLCV.")

    def fetch_open_interest(self, contract: str, start_date: date, end_date: date):
        raise NotImplementedError("OfficialMacroAdapter does not serve open interest.")

    def fetch_spot(self, commodity: str, start_date: date, end_date: date):
        raise NotImplementedError("OfficialMacroAdapter does not serve spot data.")

    def fetch_reference_rates(self, currency_pair: str, start_date: date, end_date: date):
        raise NotImplementedError("OfficialMacroAdapter does not serve generic reference rates.")

    def fetch_weather(self, location: str, start_date: date, end_date: date):
        raise NotImplementedError("OfficialMacroAdapter does not serve weather.")

    def fetch_calendar(self, exchange: str, year: int):
        raise NotImplementedError("OfficialMacroAdapter does not serve exchange calendars.")

    def _load_series_frame(self, series_name: str) -> pd.DataFrame:
        spec = self.series_catalog.get(series_name, {})
        location = spec.get("path") or spec.get("url")
        if not location:
            return pd.DataFrame()
        frame = self._read_frame(location)
        if frame.empty:
            return frame
        time_column = spec.get("timestamp_column", "timestamp")
        value_column = spec.get("value_column", "value")
        frame = frame.rename(columns={time_column: "timestamp", value_column: "value"})
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        frame = frame.set_index("timestamp").sort_index()
        return frame[["value"]]

    def _load_event_calendar(self, country: str) -> pd.DataFrame:
        spec = self.event_calendar_catalog.get(country, {})
        location = spec.get("path") or spec.get("url")
        if not location:
            return pd.DataFrame()
        frame = self._read_frame(location)
        if frame.empty:
            return frame
        time_column = spec.get("timestamp_column", "timestamp")
        frame = frame.rename(columns={time_column: "timestamp"})
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        return frame

    def _read_frame(self, location: str) -> pd.DataFrame:
        if location.startswith("http://") or location.startswith("https://"):
            if location.endswith(".parquet"):
                return pd.read_parquet(location)
            return pd.read_csv(location)
        file_path = Path(location)
        if not file_path.exists():
            return pd.DataFrame()
        if file_path.suffix == ".parquet":
            return pd.read_parquet(file_path)
        if file_path.suffix in {".csv", ".txt"}:
            return pd.read_csv(file_path)
        if file_path.suffix == ".json":
            return pd.read_json(file_path)
        return pd.DataFrame()

    def _fetch_fred_series(self, series_name: str, start_date: date, end_date: date) -> pd.DataFrame:
        try:
            from fredapi import Fred
        except ImportError:
            return pd.DataFrame()
        api_key = self.config.get("fred_api_key")
        if not api_key:
            return pd.DataFrame()
        fred = Fred(api_key=api_key)
        series = fred.get_series(self.fred_series[series_name], observation_start=start_date, observation_end=end_date)
        if series.empty:
            return pd.DataFrame()
        return series.rename("value").to_frame()

    def _fallback_calendar(self, country: str, start_date: date, end_date: date) -> pd.DataFrame:
        if country != "IN":
            return pd.DataFrame()
        dates = pd.date_range(start=start_date, end=end_date, freq="MS")
        records = []
        for month_start in dates:
            records.append(
                {
                    "timestamp": month_start + pd.Timedelta(days=11),
                    "event_type": "inflation_release",
                    "title": "India CPI Release",
                    "expected_impact": "high",
                    "source": "MOSPI",
                }
            )
        return pd.DataFrame(records)
