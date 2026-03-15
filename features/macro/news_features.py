"""
News-Derived Macro Features

Computes macroeconomic features from news and text data.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, date, timedelta
from collections import defaultdict

from .base import MacroFeatureEngine
from ...data.models import NewsItem, MacroFeature

class NewsFeatures(MacroFeatureEngine):
    """Computes news-derived macroeconomic features."""

    def compute(self, macro_data: Dict[str, List[NewsItem]], **kwargs) -> List[MacroFeature]:
        """Compute news features from news data."""
        features = []

        # News data comes as NewsItem objects, not MacroSeries
        news_data = kwargs.get('news_data', [])
        if not news_data:
            return features

        # Sentiment aggregation
        sentiment_features = self._compute_sentiment_features(news_data)
        features.extend(sentiment_features)

        # Topic intensity features
        topic_features = self._compute_topic_intensity_features(news_data)
        features.extend(topic_features)

        # News volume and burst detection
        volume_features = self._compute_news_volume_features(news_data)
        features.extend(volume_features)

        # Hawkish/dovish tone analysis
        tone_features = self._compute_tone_features(news_data)
        features.extend(tone_features)

        return features

    def _compute_sentiment_features(self, news_items: List[NewsItem]) -> List[MacroFeature]:
        """Compute aggregated sentiment features."""
        features = []

        # Group news by date
        daily_news = defaultdict(list)
        for news in news_items:
            if news.sentiment_score is not None:
                daily_news[news.timestamp.date()].append(news)

        # Compute daily sentiment aggregates
        for news_date, day_news in daily_news.items():
            sentiments = [n.sentiment_score for n in day_news]

            # Average sentiment
            avg_sentiment = np.mean(sentiments)
            features.append(self.create_macro_feature(
                name='macro_sentiment_score',
                timestamp=datetime.combine(news_date, datetime.min.time()),
                value=avg_sentiment,
                source_series=['NEWS_HEADLINES'],
                transform='sentiment_aggregate',
                frequency='daily',
                metadata={
                    'news_count': len(sentiments),
                    'sentiment_std': np.std(sentiments) if len(sentiments) > 1 else 0
                }
            ))

            # Sentiment volatility
            if len(sentiments) > 1:
                sentiment_vol = np.std(sentiments)
                features.append(self.create_macro_feature(
                    name='sentiment_volatility',
                    timestamp=datetime.combine(news_date, datetime.min.time()),
                    value=sentiment_vol,
                    source_series=['NEWS_HEADLINES'],
                    transform='sentiment_std',
                    frequency='daily'
                ))

        return features

    def _compute_topic_intensity_features(self, news_items: List[NewsItem]) -> List[MacroFeature]:
        """Compute topic intensity features."""
        features = []

        # Define key macro topics
        macro_topics = {
            'inflation': ['inflation', 'prices', 'cpi', 'wpi', 'cost'],
            'growth': ['growth', 'gdp', 'economy', 'expansion', 'recession'],
            'rates': ['rate', 'rbi', 'fed', 'policy', 'interest'],
            'fx': ['dollar', 'rupee', 'currency', 'fx', 'exchange']
        }

        # Group news by date
        daily_news = defaultdict(list)
        for news in news_items:
            daily_news[news.timestamp.date()].append(news)

        # Compute topic intensity per day
        for news_date, day_news in daily_news.items():
            total_news = len(day_news)

            for topic_name, keywords in macro_topics.items():
                topic_count = 0
                for news in day_news:
                    text = f"{news.headline} {news.content or ''}".lower()
                    if any(keyword in text for keyword in keywords):
                        topic_count += 1

                intensity = topic_count / total_news if total_news > 0 else 0

                features.append(self.create_macro_feature(
                    name=f'{topic_name}_news_intensity',
                    timestamp=datetime.combine(news_date, datetime.min.time()),
                    value=intensity,
                    source_series=['NEWS_HEADLINES'],
                    transform='topic_intensity',
                    frequency='daily',
                    metadata={
                        'topic': topic_name,
                        'topic_mentions': topic_count,
                        'total_news': total_news
                    }
                ))

        return features

    def _compute_news_volume_features(self, news_items: List[NewsItem]) -> List[MacroFeature]:
        """Compute news volume and burst detection features."""
        features = []

        # Group by date
        daily_counts = defaultdict(int)
        for news in news_items:
            daily_counts[news.timestamp.date()] += 1

        # Convert to time series
        dates = sorted(daily_counts.keys())
        counts = [daily_counts[date] for date in dates]

        if len(counts) < 7:  # Need some history for burst detection
            return features

        # Compute 7-day moving average
        ma_window = 7
        moving_avg = pd.Series(counts).rolling(window=ma_window, min_periods=ma_window).mean()

        # Detect bursts (volume > 2 * moving average)
        for i, news_date in enumerate(dates[ma_window-1:], start=ma_window-1):
            current_count = counts[i]
            avg_count = moving_avg.iloc[i]

            if not pd.isna(avg_count) and avg_count > 0:
                burst_ratio = current_count / avg_count

                features.append(self.create_macro_feature(
                    name='news_volume_burst',
                    timestamp=datetime.combine(news_date, datetime.min.time()),
                    value=burst_ratio,
                    source_series=['NEWS_HEADLINES'],
                    transform='volume_burst_ratio',
                    frequency='daily',
                    metadata={
                        'current_count': current_count,
                        'moving_avg': avg_count,
                        'is_burst': burst_ratio > 2.0
                    }
                ))

        return features

    def _compute_tone_features(self, news_items: List[NewsItem]) -> List[MacroFeature]:
        """Compute hawkish/dovish tone features."""
        features = []

        # Keywords for hawkish vs dovish tone
        hawkish_keywords = ['raise', 'hike', 'tighten', 'restrictive', 'hawkish', 'inflation']
        dovish_keywords = ['cut', 'ease', 'accommodative', 'dovish', 'stimulus', 'support']

        # Group by date
        daily_news = defaultdict(list)
        for news in news_items:
            daily_news[news.timestamp.date()].append(news)

        for news_date, day_news in daily_news.items():
            hawkish_count = 0
            dovish_count = 0

            for news in day_news:
                text = f"{news.headline} {news.content or ''}".lower()

                hawkish_matches = sum(1 for keyword in hawkish_keywords if keyword in text)
                dovish_matches = sum(1 for keyword in dovish_keywords if keyword in text)

                hawkish_count += hawkish_matches
                dovish_count += dovish_matches

            total_signals = hawkish_count + dovish_count
            if total_signals > 0:
                hawkish_ratio = hawkish_count / total_signals

                features.append(self.create_macro_feature(
                    name='hawkish_dovish_score',
                    timestamp=datetime.combine(news_date, datetime.min.time()),
                    value=hawkish_ratio,  # 1.0 = fully hawkish, 0.0 = fully dovish
                    source_series=['NEWS_HEADLINES'],
                    transform='hawkish_dovish_sentiment',
                    frequency='daily',
                    metadata={
                        'hawkish_signals': hawkish_count,
                        'dovish_signals': dovish_count,
                        'total_signals': total_signals
                    }
                ))

        return features

    def compute_commodity_news_relevance(self, news_items: List[NewsItem],
                                       commodity: str) -> List[MacroFeature]:
        """Compute commodity-specific news relevance."""
        features = []

        # Commodity-specific keywords
        commodity_keywords = {
            'GOLD': ['gold', 'bullion', 'precious metals', 'jewelry'],
            'CRUDE': ['oil', 'crude', 'petroleum', 'energy', 'opec'],
            'COPPER': ['copper', 'base metals', 'industrial metals'],
            'SILVER': ['silver', 'precious metals'],
            'COTTON': ['cotton', 'textile', 'fiber']
        }

        keywords = commodity_keywords.get(commodity.upper(), [commodity.lower()])

        # Group by date
        daily_news = defaultdict(list)
        for news in news_items:
            daily_news[news.timestamp.date()].append(news)

        for news_date, day_news in daily_news.items():
            relevant_count = 0

            for news in day_news:
                text = f"{news.headline} {news.content or ''}".lower()
                if any(keyword in text for keyword in keywords):
                    relevant_count += 1

            relevance_ratio = relevant_count / len(day_news) if day_news else 0

            features.append(self.create_macro_feature(
                name=f'{commodity.lower()}_news_relevance',
                timestamp=datetime.combine(news_date, datetime.min.time()),
                value=relevance_ratio,
                source_series=['NEWS_HEADLINES'],
                transform='commodity_news_relevance',
                frequency='daily',
                metadata={
                    'commodity': commodity,
                    'relevant_news': relevant_count,
                    'total_news': len(day_news)
                }
            ))

        return features

    def classify_news_sentiment_regime(self, news_features: List[MacroFeature]) -> str:
        """Classify overall news sentiment regime."""
        sentiment_features = [f for f in news_features if f.feature_name == 'macro_sentiment_score']

        if not sentiment_features:
            return 'unknown'

        latest_sentiment = max(sentiment_features, key=lambda x: x.timestamp)
        sentiment = latest_sentiment.value

        # Check for burst conditions
        burst_features = [f for f in news_features if f.feature_name == 'news_volume_burst']
        is_burst = False
        if burst_features:
            latest_burst = max(burst_features, key=lambda x: x.timestamp)
            is_burst = latest_burst.metadata.get('is_burst', False)

        # Classify regime
        if is_burst and sentiment < -0.2:
            return 'negative_burst'
        elif is_burst and sentiment > 0.2:
            return 'positive_burst'
        elif sentiment < -0.3:
            return 'bearish'
        elif sentiment > 0.3:
            return 'bullish'
        else:
            return 'neutral'