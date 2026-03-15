from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import date, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

from .....config.commodity_universe import commodity_keyword_map
from ..base import MacroDataSource
from ....models import MacroEvent, MacroSeries, NewsItem


class GenericNewsAdapter(MacroDataSource):
    """Free news adapter using RSS and local CSV files, with paid feeds optional."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.rss_feeds = config.get(
            "rss_feeds",
            {
                "rbi_press": "https://www.rbi.org.in/Scripts/BS_PressReleaseDisplay.aspx",
                "pib_economy": "https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3",
                "economic_times": "https://economictimes.indiatimes.com/rssfeedsdefault.cms",
            },
        )
        self.csv_paths = config.get("csv_paths", {})
        self.allow_paid_sources = bool(config.get("allow_paid_sources", False))
        self.session = requests.Session()
        self.timeout = int(config.get("timeout", 20))

    def fetch_macro_series(self, series_name: str, start_date: date, end_date: date) -> List[MacroSeries]:
        return []

    def fetch_macro_release_calendar(self, country: str, start_date: date, end_date: date) -> List[MacroEvent]:
        return []

    def fetch_policy_rates(self, country: str, start_date: date, end_date: date) -> List[MacroSeries]:
        return []

    def fetch_yields(self, country: str, tenor: str, start_date: date, end_date: date) -> List[MacroSeries]:
        return []

    def fetch_fx_reference(self, currency_pair: str, start_date: date, end_date: date) -> List[MacroSeries]:
        return []

    def fetch_news_headlines(self, topics: List[str], start_date: date, end_date: date) -> List[NewsItem]:
        headlines: List[NewsItem] = []
        for source_name, feed_url in self.rss_feeds.items():
            if not self.allow_paid_sources and any(paid_name in source_name.lower() for paid_name in ("reuters", "bloomberg")):
                continue
            try:
                headlines.extend(self._fetch_rss_headlines(source_name, feed_url, topics, start_date, end_date))
            except Exception:
                continue
        csv_path = self.csv_paths.get("news_csv")
        if csv_path:
            headlines.extend(self._fetch_csv_headlines(csv_path, topics, start_date, end_date))
        return self._deduplicate_headlines(headlines)

    def fetch_news_articles(self, news_ids: List[str]) -> List[NewsItem]:
        return []

    def fetch_macro_events(self, country: str, event_types: List[str], start_date: date, end_date: date) -> List[MacroEvent]:
        return []

    def fetch_contract_master(self, commodity: str, as_of_date: Optional[date] = None):
        raise NotImplementedError("GenericNewsAdapter does not serve contract metadata.")

    def fetch_ohlcv(self, contract: str, start_date: date, end_date: date):
        raise NotImplementedError("GenericNewsAdapter does not serve OHLCV.")

    def fetch_open_interest(self, contract: str, start_date: date, end_date: date):
        raise NotImplementedError("GenericNewsAdapter does not serve open interest.")

    def fetch_spot(self, commodity: str, start_date: date, end_date: date):
        raise NotImplementedError("GenericNewsAdapter does not serve spot data.")

    def fetch_reference_rates(self, currency_pair: str, start_date: date, end_date: date):
        raise NotImplementedError("GenericNewsAdapter does not serve reference rates.")

    def fetch_weather(self, location: str, start_date: date, end_date: date):
        raise NotImplementedError("GenericNewsAdapter does not serve weather.")

    def fetch_calendar(self, exchange: str, year: int):
        raise NotImplementedError("GenericNewsAdapter does not serve exchange calendars.")

    def _fetch_rss_headlines(self, source_name: str, feed_url: str, topics: List[str], start_date: date, end_date: date) -> List[NewsItem]:
        response = self.session.get(feed_url, timeout=self.timeout)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        headlines: List[NewsItem] = []
        for item in root.findall(".//item"):
            title = self._element_text(item, "title")
            description = self._element_text(item, "description")
            published = self._parse_date(self._element_text(item, "pubDate"))
            if not published or not (start_date <= published.date() <= end_date):
                continue
            matched_topics = self._match_topics(f"{title} {description}", topics)
            if topics and not matched_topics:
                continue
            headlines.append(
                NewsItem(
                    news_id=f"{source_name}_{published.strftime('%Y%m%d%H%M%S')}_{abs(hash(title)) % 10000}",
                    timestamp=published,
                    headline=title,
                    source=source_name.upper(),
                    url=self._element_text(item, "link"),
                    content=description,
                    relevance_score=self._relevance_score(title, description, topics, matched_topics),
                    sentiment_score=self._sentiment_score(f"{title} {description}"),
                    topics=matched_topics,
                    commodity_tags=self._extract_commodity_tags(f"{title} {description}"),
                    metadata={"feed_url": feed_url, "free_source": True},
                )
            )
        return headlines

    def _fetch_csv_headlines(self, csv_path: str, topics: List[str], start_date: date, end_date: date) -> List[NewsItem]:
        file_path = Path(csv_path)
        if not file_path.exists():
            return []
        df = pd.read_csv(file_path)
        headlines: List[NewsItem] = []
        for row in df.to_dict(orient="records"):
            published = pd.Timestamp(row.get("timestamp")).to_pydatetime()
            if not (start_date <= published.date() <= end_date):
                continue
            title = str(row.get("headline", ""))
            content = str(row.get("content", ""))
            matched_topics = self._match_topics(f"{title} {content}", topics)
            if topics and not matched_topics:
                continue
            headlines.append(
                NewsItem(
                    news_id=str(row.get("news_id", f"csv_{published.strftime('%Y%m%d%H%M%S')}")),
                    timestamp=published,
                    headline=title,
                    source=str(row.get("source", "LOCAL_CSV")),
                    url=row.get("url"),
                    content=content,
                    relevance_score=float(row.get("relevance_score", self._relevance_score(title, content, topics, matched_topics))),
                    sentiment_score=float(row["sentiment_score"]) if pd.notna(row.get("sentiment_score")) else self._sentiment_score(f"{title} {content}"),
                    topics=matched_topics or row.get("topics", []),
                    commodity_tags=self._extract_commodity_tags(f"{title} {content}"),
                    metadata={"csv_source": csv_path, "free_source": True},
                )
            )
        return headlines

    def _element_text(self, item: ET.Element, tag_name: str) -> str:
        element = item.find(tag_name)
        return element.text.strip() if element is not None and element.text else ""

    def _parse_date(self, value: str) -> Optional[datetime]:
        if not value:
            return None
        try:
            return parsedate_to_datetime(value).replace(tzinfo=None)
        except Exception:
            try:
                return pd.Timestamp(value).to_pydatetime()
            except Exception:
                return None

    def _match_topics(self, text: str, topics: List[str]) -> List[str]:
        if not topics:
            return []
        lowered = text.lower()
        return [topic for topic in topics if topic.lower() in lowered]

    def _extract_commodity_tags(self, text: str) -> List[str]:
        commodity_map = commodity_keyword_map()
        lowered = text.lower()
        tags = []
        for commodity, keywords in commodity_map.items():
            if any(keyword in lowered for keyword in keywords):
                tags.append(commodity)
        return tags

    def _relevance_score(self, title: str, content: str, topics: List[str], matched_topics: List[str]) -> float:
        topic_weight = len(matched_topics) / len(topics) if topics else 0.4
        commodity_bonus = min(0.3, len(self._extract_commodity_tags(f"{title} {content}")) * 0.1)
        return max(0.0, min(1.0, topic_weight + commodity_bonus))

    def _sentiment_score(self, text: str) -> Optional[float]:
        positive_terms = {"rise", "support", "strong", "boost", "ease", "surplus"}
        negative_terms = {"fall", "risk", "weak", "pressure", "tight", "shock"}
        tokens = set(re.findall(r"[a-zA-Z]+", text.lower()))
        score = 0.0
        score += 0.15 * len(tokens & positive_terms)
        score -= 0.15 * len(tokens & negative_terms)
        return max(-1.0, min(1.0, score))

    def _deduplicate_headlines(self, headlines: List[NewsItem]) -> List[NewsItem]:
        seen = set()
        unique: List[NewsItem] = []
        for headline in sorted(headlines, key=lambda item: item.timestamp, reverse=True):
            key = (headline.headline.strip().lower(), headline.timestamp.date(), headline.source)
            if key not in seen:
                seen.add(key)
                unique.append(headline)
        return unique
