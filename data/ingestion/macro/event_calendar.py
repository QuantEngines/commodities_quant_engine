"""
Event Calendar Ingestion Module

Handles ingestion of macroeconomic event calendars and scheduled releases.
"""

import pandas as pd
from typing import List, Dict, Optional, Any
from datetime import datetime, date, timedelta
from pathlib import Path

from ...data.models import MacroEvent
from ...data.storage.local import storage
from .macro.base import MacroDataSource
from ...config.settings import settings

class EventCalendarIngestion:
    """Manages ingestion of macroeconomic event calendars."""

    def __init__(self):
        self.sources = self._initialize_sources()

    def _initialize_sources(self) -> Dict[str, MacroDataSource]:
        """Initialize event data sources."""
        sources = {}
        for source_name, source_config in settings.macro.sources.items():
            if source_config.enabled:
                try:
                    module_path, class_name = source_config.adapter_class.rsplit('.', 1)
                    module = __import__(module_path, fromlist=[class_name])
                    adapter_class = getattr(module, class_name)
                    sources[source_name] = adapter_class(source_config.config)
                except Exception as e:
                    print(f"Failed to initialize event source {source_name}: {e}")
        return sources

    def ingest_event_calendar(self, country: str, start_date: date, end_date: date,
                             force_refresh: bool = False) -> List[MacroEvent]:
        """
        Ingest macroeconomic event calendar for a country.

        Args:
            country: Country code (e.g., 'IN', 'US')
            start_date: Start date for calendar
            end_date: End date for calendar
            force_refresh: If True, ignore cache

        Returns:
            List of MacroEvent objects
        """
        cache_key = f"event_calendar_{country}_{start_date}_{end_date}"

        if not force_refresh:
            cached_data = self._load_from_cache(cache_key)
            if cached_data:
                return cached_data

        all_events = []

        # Fetch from each source
        for source_name, source in self.sources.items():
            try:
                events = source.fetch_macro_release_calendar(country, start_date, end_date)
                all_events.extend(events)
                print(f"Fetched {len(events)} events from {source_name}")
            except Exception as e:
                print(f"Error fetching events from {source_name}: {e}")

        # Remove duplicates and sort
        unique_events = self._deduplicate_events(all_events)

        # Cache results
        self._save_to_cache(cache_key, unique_events)

        return unique_events

    def ingest_events_by_type(self, country: str, event_types: List[str],
                             start_date: date, end_date: date) -> List[MacroEvent]:
        """Fetch events filtered by type."""
        all_events = self.ingest_event_calendar(country, start_date, end_date)

        # Filter by event types
        filtered_events = []
        for event in all_events:
            if any(event_type in event.event_type for event_type in event_types):
                filtered_events.append(event)

        return filtered_events

    def get_upcoming_events(self, days_ahead: int = 30) -> List[MacroEvent]:
        """Get upcoming macro events across all countries."""
        start_date = date.today()
        end_date = start_date + timedelta(days=days_ahead)

        countries = ['IN', 'US', 'CN']  # Configurable list

        all_events = []
        for country in countries:
            events = self.ingest_event_calendar(country, start_date, end_date)
            all_events.extend(events)

        # Sort by date
        return sorted(all_events, key=lambda x: x.timestamp)

    def get_high_impact_events(self, start_date: date, end_date: date) -> List[MacroEvent]:
        """Get high-impact macro events."""
        all_events = []
        countries = ['IN', 'US']

        for country in countries:
            events = self.ingest_event_calendar(country, start_date, end_date)
            high_impact = [e for e in events if e.expected_impact == 'high']
            all_events.extend(high_impact)

        return sorted(all_events, key=lambda x: x.timestamp)

    def _deduplicate_events(self, events: List[MacroEvent]) -> List[MacroEvent]:
        """Remove duplicate events."""
        seen = set()
        unique = []

        for event in events:
            # Use event ID, type, country, and date as uniqueness key
            key = (event.event_type, event.country, event.timestamp.date(), event.title.strip().lower())
            if key not in seen:
                seen.add(key)
                unique.append(event)

        return sorted(unique, key=lambda x: x.timestamp)

    def _load_from_cache(self, cache_key: str) -> Optional[List[MacroEvent]]:
        """Load event data from cache."""
        try:
            df = storage.load_domain_dataframe("event_cache", cache_key)
            if not df.empty:
                return [MacroEvent(**row) for _, row in df.iterrows()]
        except Exception:
            pass
        return None

    def _save_to_cache(self, cache_key: str, events: List[MacroEvent]):
        """Save event data to cache."""
        if not events:
            return

        df = pd.DataFrame([e.__dict__ for e in events])
        storage.append_dataframe(df, "event_cache", cache_key)

    def get_event_risk_windows(self, events: List[MacroEvent], risk_window_days: int = 3) -> List[Dict[str, Any]]:
        """Calculate risk windows around high-impact events."""
        risk_windows = []

        for event in events:
            if event.expected_impact in ['high', 'medium']:
                window_start = event.timestamp.date() - timedelta(days=risk_window_days)
                window_end = event.timestamp.date() + timedelta(days=1)  # Include event day

                risk_windows.append({
                    'event': event,
                    'risk_window_start': window_start,
                    'risk_window_end': window_end,
                    'days_until_event': (event.timestamp.date() - date.today()).days
                })

        return risk_windows

    def is_event_risk_day(self, target_date: date, events: Optional[List[MacroEvent]] = None,
                         risk_window_days: int = 3) -> bool:
        """Check if a date falls within an event risk window."""
        if events is None:
            events = self.get_upcoming_events(days_ahead=90)

        risk_windows = self.get_event_risk_windows(events, risk_window_days)

        for window in risk_windows:
            if window['risk_window_start'] <= target_date <= window['risk_window_end']:
                return True

        return False

    def get_event_summary(self, events: List[MacroEvent]) -> Dict[str, Any]:
        """Generate summary statistics for events."""
        if not events:
            return {"total_events": 0}

        countries = list(set(e.country for e in events))
        event_types = list(set(e.event_type for e in events))
        impacts = {}
        for e in events:
            impacts[e.expected_impact] = impacts.get(e.expected_impact, 0) + 1

        summary = {
            "total_events": len(events),
            "countries": countries,
            "event_types": event_types,
            "impact_distribution": impacts,
            "date_range": (min(e.timestamp.date() for e in events), max(e.timestamp.date() for e in events))
        }

        return summary