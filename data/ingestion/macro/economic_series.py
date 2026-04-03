"""
Economic Series Ingestion Module

Handles ingestion of macroeconomic time series data from various sources.
"""

import pandas as pd
from typing import List, Dict, Optional, Any
from datetime import datetime, date
from pathlib import Path

from ...data.models import MacroSeries
from ...data.storage.local import storage
from .macro.base import MacroDataSource
from ...config.settings import settings

class EconomicSeriesIngestion:
    """Manages ingestion of economic series data."""

    def __init__(self):
        self.sources = self._initialize_sources()

    def _initialize_sources(self) -> Dict[str, MacroDataSource]:
        """Initialize configured macro data sources."""
        sources = {}
        for source_name, source_config in settings.macro.sources.items():
            if source_config.enabled:
                try:
                    # Dynamic import and instantiation
                    module_path, class_name = source_config.adapter_class.rsplit('.', 1)
                    module = __import__(module_path, fromlist=[class_name])
                    adapter_class = getattr(module, class_name)
                    sources[source_name] = adapter_class(source_config.config)
                except Exception as e:
                    print(f"Failed to initialize {source_name}: {e}")
        return sources

    def ingest_series(self, series_name: str, start_date: date, end_date: date,
                     force_refresh: bool = False) -> List[MacroSeries]:
        """
        Ingest economic series data from all available sources.

        Args:
            series_name: Name of the series (e.g., 'IN_CPI_YOY')
            start_date: Start date for data
            end_date: End date for data
            force_refresh: If True, ignore cache and re-fetch

        Returns:
            List of MacroSeries objects
        """
        canonical_series_name = self._resolve_series_alias(series_name)
        cache_key = f"macro_series_{canonical_series_name}_{start_date}_{end_date}"

        if not force_refresh:
            cached_data = self._load_from_cache(cache_key)
            if cached_data:
                return cached_data

        all_series = []

        # Try each source
        for source_name, source in self.sources.items():
            try:
                series_data = source.fetch_macro_series(canonical_series_name, start_date, end_date)
                all_series.extend(series_data)
                print(f"Fetched {len(series_data)} points from {source_name}")
            except Exception as e:
                print(f"Error fetching from {source_name}: {e}")

        # Remove duplicates and sort
        unique_series = self._deduplicate_series(all_series)

        # Cache the results
        self._save_to_cache(cache_key, unique_series)

        return unique_series

    def ingest_multiple_series(self, series_names: List[str], start_date: date, end_date: date,
                              force_refresh: bool = False) -> Dict[str, List[MacroSeries]]:
        """Ingest multiple series in batch."""
        results = {}
        for series_name in series_names:
            results[series_name] = self.ingest_series(series_name, start_date, end_date, force_refresh)
        return results

    def _deduplicate_series(self, series_list: List[MacroSeries]) -> List[MacroSeries]:
        """Remove duplicate series data points."""
        seen = set()
        unique = []

        for series in series_list:
            key = (series.series_id, series.timestamp, series.source)
            if key not in seen:
                seen.add(key)
                unique.append(series)

        return sorted(unique, key=lambda x: (x.timestamp, x.source))

    def _load_from_cache(self, cache_key: str) -> Optional[List[MacroSeries]]:
        """Load series data from cache."""
        try:
            df = storage.load_domain_dataframe("macro_cache", cache_key)
            if not df.empty:
                return [MacroSeries(**row) for _, row in df.iterrows()]
        except Exception:
            pass
        return None

    def _save_to_cache(self, cache_key: str, series_list: List[MacroSeries]):
        """Save series data to cache."""
        if not series_list:
            return

        df = pd.DataFrame([s.__dict__ for s in series_list])
        storage.append_dataframe(df, "macro_cache", cache_key)

    def get_available_series(self) -> List[str]:
        """Get list of available series across all sources."""
        available = set()
        for source in self.sources.values():
            series_catalog = getattr(source, "series_catalog", {}) or {}
            fred_series = getattr(source, "fred_series", {}) or {}
            available.update(series_catalog.keys())
            available.update(fred_series.keys())

        available.update(settings.macro.series_mappings.keys())
        for mapping in settings.macro.series_mappings.values():
            available.update(str(value) for value in mapping.values() if value)
        return sorted(available)

    def _resolve_series_alias(self, series_name: str) -> str:
        """Resolve aliases like BDI/OVX to canonical internal series ids."""
        normalized = str(series_name).upper()
        mappings = settings.macro.series_mappings
        if normalized in mappings:
            return normalized

        for canonical, alias_map in mappings.items():
            canonical_normalized = str(canonical).upper()
            if normalized == canonical_normalized:
                return str(canonical)
            aliases = {canonical_normalized}
            aliases.update(str(key).upper() for key in alias_map.keys())
            aliases.update(str(value).upper() for value in alias_map.values())
            if normalized in aliases:
                return str(canonical)

        return str(series_name)

    def validate_series_data(self, series_list: List[MacroSeries]) -> Dict[str, Any]:
        """Validate ingested series data quality."""
        if not series_list:
            return {"valid": False, "errors": ["No data"]}

        timestamps = [s.timestamp for s in series_list]
        values = [s.value for s in series_list]

        validation = {
            "count": len(series_list),
            "date_range": (min(timestamps), max(timestamps)) if timestamps else None,
            "missing_values": sum(1 for s in series_list if s.value is None),
            "sources": list(set(s.source for s in series_list)),
            "valid": True
        }

        return validation