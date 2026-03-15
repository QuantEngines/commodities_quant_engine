from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd

from ....config.commodity_universe import default_news_topics
from ...models import NewsItem
from ...storage.local import storage
from ....config.settings import settings
from .base import MacroDataSource


class NewsIngestion:
    """Free/local-first news ingestion orchestrator."""

    def __init__(self):
        self.sources = self._initialize_sources()

    def _initialize_sources(self) -> Dict[str, MacroDataSource]:
        sources: Dict[str, MacroDataSource] = {}
        for source_name, source_config in settings.macro.sources.items():
            if source_config.enabled and "news" in source_config.adapter_class.lower():
                try:
                    module_path, class_name = source_config.adapter_class.rsplit(".", 1)
                    module = __import__(module_path, fromlist=[class_name])
                    adapter_class = getattr(module, class_name)
                    sources[source_name] = adapter_class(source_config.config)
                except Exception:
                    continue
        return sources

    def ingest_headlines(self, topics: List[str], start_date: date, end_date: date, force_refresh: bool = False) -> List[NewsItem]:
        cache_key = f"news_headlines_{'_'.join(sorted(topics))}_{start_date}_{end_date}"
        if not force_refresh:
            cached = self._load_from_cache(cache_key)
            if cached:
                return cached
        headlines: List[NewsItem] = []
        for source in self.sources.values():
            try:
                headlines.extend(source.fetch_news_headlines(topics, start_date, end_date))
            except Exception:
                continue
        deduped = self._deduplicate_headlines(headlines)
        self._save_to_cache(cache_key, deduped)
        return deduped

    def search_by_commodities(self, commodities: List[str], start_date: date, end_date: date, force_refresh: bool = False) -> List[NewsItem]:
        commodity_topics = default_news_topics()
        topics: List[str] = []
        for commodity in commodities:
            topics.extend(commodity_topics.get(commodity.upper(), [commodity.lower()]))
        return self.ingest_headlines(sorted(set(topics)), start_date, end_date, force_refresh)

    def get_news_sentiment_summary(self, headlines: List[NewsItem]) -> Dict[str, Any]:
        if not headlines:
            return {"total_headlines": 0, "avg_sentiment": None, "sentiment_distribution": {}}
        sentiments = [item.sentiment_score for item in headlines if item.sentiment_score is not None]
        return {
            "total_headlines": len(headlines),
            "avg_sentiment": sum(sentiments) / len(sentiments) if sentiments else None,
            "sentiment_distribution": {
                "positive": len([value for value in sentiments if value > 0.1]),
                "negative": len([value for value in sentiments if value < -0.1]),
                "neutral": len([value for value in sentiments if -0.1 <= value <= 0.1]),
            },
            "top_sources": sorted({item.source for item in headlines}),
        }

    def _deduplicate_headlines(self, headlines: List[NewsItem]) -> List[NewsItem]:
        seen = set()
        unique: List[NewsItem] = []
        for item in sorted(headlines, key=lambda record: record.timestamp, reverse=True):
            key = (item.headline.strip().lower(), item.timestamp.date(), item.source)
            if key not in seen:
                seen.add(key)
                unique.append(item)
        return unique

    def _load_from_cache(self, cache_key: str) -> Optional[List[NewsItem]]:
        df = storage.load_domain_dataframe("news_cache", cache_key)
        if df.empty:
            return None
        records = []
        for row in df.to_dict(orient="records"):
            row["timestamp"] = pd.Timestamp(row["timestamp"]).to_pydatetime()
            row["topics"] = row.get("topics", []) if isinstance(row.get("topics"), list) else []
            row["commodity_tags"] = row.get("commodity_tags", []) if isinstance(row.get("commodity_tags"), list) else []
            row["metadata"] = row.get("metadata", {}) if isinstance(row.get("metadata"), dict) else {}
            records.append(NewsItem(**row))
        return records

    def _save_to_cache(self, cache_key: str, headlines: List[NewsItem]):
        if not headlines:
            return
        df = pd.DataFrame([item.__dict__ for item in headlines])
        storage.append_dataframe(df, "news_cache", cache_key, dedupe_on=["news_id"])
