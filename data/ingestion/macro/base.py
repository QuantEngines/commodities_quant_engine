from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from datetime import datetime, date
import pandas as pd

from ..base import DataSource
from ...models import MacroSeries, MacroEvent, NewsItem

class MacroDataSource(DataSource):
    """Abstract base class for macroeconomic data sources."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.source_name = config.get('source_name', self.__class__.__name__)

    @abstractmethod
    def fetch_macro_series(self, series_name: str, start_date: date, end_date: date) -> List[MacroSeries]:
        """Fetch standardized macro series data."""
        pass

    @abstractmethod
    def fetch_macro_release_calendar(self, country: str, start_date: date, end_date: date) -> List[MacroEvent]:
        """Fetch scheduled macro event calendar."""
        pass

    @abstractmethod
    def fetch_policy_rates(self, country: str, start_date: date, end_date: date) -> List[MacroSeries]:
        """Fetch central bank policy rates."""
        pass

    @abstractmethod
    def fetch_yields(self, country: str, tenor: str, start_date: date, end_date: date) -> List[MacroSeries]:
        """Fetch government bond yields."""
        pass

    @abstractmethod
    def fetch_fx_reference(self, currency_pair: str, start_date: date, end_date: date) -> List[MacroSeries]:
        """Fetch FX reference rates."""
        pass

    @abstractmethod
    def fetch_news_headlines(self, topics: List[str], start_date: date, end_date: date) -> List[NewsItem]:
        """Fetch news headlines by topic."""
        pass

    @abstractmethod
    def fetch_news_articles(self, news_ids: List[str]) -> List[NewsItem]:
        """Fetch full news articles by ID."""
        pass

    @abstractmethod
    def fetch_macro_events(self, country: str, event_types: List[str], start_date: date, end_date: date) -> List[MacroEvent]:
        """Fetch macro events by type and country."""
        pass

    def standardize_macro_series(self, raw_data: pd.DataFrame, series_id: str, unit: str,
                                frequency: str, source: str) -> List[MacroSeries]:
        """Standardize raw macro data into MacroSeries objects."""
        series_list = []
        for _, row in raw_data.iterrows():
            series = MacroSeries(
                series_id=series_id,
                timestamp=row['timestamp'] if isinstance(row['timestamp'], datetime) else pd.to_datetime(row['timestamp']),
                value=float(row['value']),
                unit=unit,
                frequency=frequency,
                source=source,
                is_revised=row.get('is_revised', False),
                original_timestamp=row.get('original_timestamp'),
                metadata=row.get('metadata', {})
            )
            series_list.append(series)
        return series_list

    def standardize_macro_events(self, raw_data: pd.DataFrame, event_type: str, country: str,
                                source: str) -> List[MacroEvent]:
        """Standardize raw event data into MacroEvent objects."""
        events = []
        for _, row in raw_data.iterrows():
            event = MacroEvent(
                event_id=row.get('event_id', f"{event_type}_{country}_{row['timestamp'].strftime('%Y%m%d')}"),
                event_type=event_type,
                country=country,
                timestamp=row['timestamp'] if isinstance(row['timestamp'], datetime) else pd.to_datetime(row['timestamp']),
                title=row['title'],
                description=row.get('description'),
                expected_impact=row.get('expected_impact', 'medium'),
                actual_value=row.get('actual_value'),
                consensus_value=row.get('consensus_value'),
                previous_value=row.get('previous_value'),
                source=source,
                metadata=row.get('metadata', {})
            )
            events.append(event)
        return events

    def standardize_news_items(self, raw_data: pd.DataFrame, source: str) -> List[NewsItem]:
        """Standardize raw news data into NewsItem objects."""
        news_items = []
        for _, row in raw_data.iterrows():
            news = NewsItem(
                news_id=row.get('news_id', f"{source}_{row['timestamp'].strftime('%Y%m%d_%H%M%S')}"),
                timestamp=row['timestamp'] if isinstance(row['timestamp'], datetime) else pd.to_datetime(row['timestamp']),
                headline=row['headline'],
                source=source,
                url=row.get('url'),
                content=row.get('content'),
                relevance_score=row.get('relevance_score', 0.0),
                sentiment_score=row.get('sentiment_score'),
                topics=row.get('topics', []),
                commodity_tags=row.get('commodity_tags', []),
                metadata=row.get('metadata', {})
            )
            news_items.append(news)
        return news_items