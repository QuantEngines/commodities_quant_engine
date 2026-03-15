"""
Rates Feature Engineering

Computes interest rate and liquidity-related features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, date

from .base import MacroFeatureEngine
from ...data.models import MacroSeries, MacroFeature

class RatesFeatures(MacroFeatureEngine):
    """Computes interest rate and liquidity-related macroeconomic features."""

    def compute(self, macro_data: Dict[str, List[MacroSeries]], **kwargs) -> List[MacroFeature]:
        """Compute rates features from macro series data."""
        features = []

        # RBI policy rate features
        if 'IN_RBI_RATE' in macro_data:
            rbi_features = self._compute_rbi_rate_features(macro_data['IN_RBI_RATE'])
            features.extend(rbi_features)

        # Fed rate features
        if 'US_FED_RATE' in macro_data:
            fed_features = self._compute_fed_rate_features(macro_data['US_FED_RATE'])
            features.extend(fed_features)

        # Real rate calculations
        if 'IN_RBI_RATE' in macro_data and 'IN_CPI_YOY' in macro_data:
            real_rate_features = self._compute_real_rates(
                macro_data['IN_RBI_RATE'], macro_data['IN_CPI_YOY'], 'IN'
            )
            features.extend(real_rate_features)

        if 'US_FED_RATE' in macro_data and 'US_CPI_YOY' in macro_data:
            us_real_rate_features = self._compute_real_rates(
                macro_data['US_FED_RATE'], macro_data['US_CPI_YOY'], 'US'
            )
            features.extend(us_real_rate_features)

        # Rate change patterns
        if 'IN_RBI_RATE' in macro_data:
            change_features = self._compute_rate_change_patterns(macro_data['IN_RBI_RATE'], 'IN')
            features.extend(change_features)

        return features

    def _compute_rbi_rate_features(self, rbi_series: List[MacroSeries]) -> List[MacroFeature]:
        """Compute RBI policy rate features."""
        features = []
        df = self.transform_series_to_dataframe(rbi_series)

        if df.empty:
            return features

        series = df['value']

        # Current RBI rate level
        for timestamp, value in series.dropna().items():
            features.append(self.create_macro_feature(
                name='rbi_rate',
                timestamp=timestamp,
                value=value,
                source_series=['IN_RBI_RATE'],
                transform='level',
                frequency='daily'
            ))

        return features

    def _compute_fed_rate_features(self, fed_series: List[MacroSeries]) -> List[MacroFeature]:
        """Compute Federal Reserve rate features."""
        features = []
        df = self.transform_series_to_dataframe(fed_series)

        if df.empty:
            return features

        series = df['value']

        # Current Fed rate level
        for timestamp, value in series.dropna().items():
            features.append(self.create_macro_feature(
                name='fed_rate',
                timestamp=timestamp,
                value=value,
                source_series=['US_FED_RATE'],
                transform='level',
                frequency='daily'
            ))

        return features

    def _compute_real_rates(self, rate_series: List[MacroSeries],
                           inflation_series: List[MacroSeries], country: str) -> List[MacroFeature]:
        """Compute real interest rates (nominal rate - inflation)."""
        features = []

        rate_df = self.transform_series_to_dataframe(rate_series)
        inflation_df = self.transform_series_to_dataframe(inflation_series)

        if rate_df.empty or inflation_df.empty:
            return features

        # Align timestamps (rates are daily, inflation monthly)
        # Use forward-fill for inflation to match rate timestamps
        inflation_daily = inflation_df['value'].resample('D').ffill()

        combined = pd.concat([rate_df['value'], inflation_daily], axis=1, keys=['rate', 'inflation'])
        combined = combined.dropna()

        real_rate = combined['rate'] - combined['inflation']

        feature_name = f'real_rate_{country.lower()}'
        for timestamp, value in real_rate.items():
            features.append(self.create_macro_feature(
                name=feature_name,
                timestamp=timestamp,
                value=value,
                source_series=[f'{country}_RATE', f'{country}_CPI_YOY'],
                transform='rate_minus_inflation',
                frequency='daily'
            ))

        return features

    def _compute_rate_change_patterns(self, rate_series: List[MacroSeries], country: str) -> List[MacroFeature]:
        """Compute rate change patterns and regime indicators."""
        features = []
        df = self.transform_series_to_dataframe(rate_series)

        if df.empty or len(df) < 30:  # Need some history
            return features

        series = df['value']

        # Recent rate changes (30-day change)
        recent_change = series - series.shift(30)
        for timestamp, value in recent_change.dropna().items():
            features.append(self.create_macro_feature(
                name=f'{country.lower()}_rate_change_30d',
                timestamp=timestamp,
                value=value,
                source_series=[f'{country}_RATE'],
                transform='change_since_days',
                frequency='daily'
            ))

        # Rate volatility (30-day rolling std)
        rate_volatility = series.rolling(window=30, min_periods=10).std()
        for timestamp, value in rate_volatility.dropna().items():
            features.append(self.create_macro_feature(
                name=f'{country.lower()}_rate_volatility',
                timestamp=timestamp,
                value=value,
                source_series=[f'{country}_RATE'],
                transform='rolling_std',
                frequency='daily'
            ))

        # Rate trend (90-day change)
        trend_change = series - series.shift(90)
        for timestamp, value in trend_change.dropna().items():
            features.append(self.create_macro_feature(
                name=f'{country.lower()}_rate_trend_90d',
                timestamp=timestamp,
                value=value,
                source_series=[f'{country}_RATE'],
                transform='change_since_days',
                frequency='daily'
            ))

        return features

    def classify_rate_regime(self, rate_features: List[MacroFeature], country: str) -> str:
        """Classify current rate regime."""
        # Extract latest rate and recent changes
        rate_level_features = [f for f in rate_features if f.feature_name == f'{country.lower()}_rate']
        change_features = [f for f in rate_features if 'rate_change_30d' in f.feature_name]

        if not rate_level_features:
            return 'unknown'

        latest_rate = max(rate_level_features, key=lambda x: x.timestamp)
        rate_level = latest_rate.value

        recent_change = 0
        if change_features:
            latest_change = max(change_features, key=lambda x: x.timestamp)
            recent_change = latest_change.value

        # Simple regime classification
        if recent_change > 0.25:  # Hiked more than 25bps recently
            return 'tightening'
        elif recent_change < -0.25:  # Cut more than 25bps recently
            return 'easing'
        elif rate_level > 6.0:  # High absolute level
            return 'restrictive'
        elif rate_level < 4.0:  # Low absolute level
            return 'accommodative'
        else:
            return 'neutral'