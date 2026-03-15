"""
Growth Feature Engineering

Computes growth-related features from GDP, IIP, PMI, and other indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, date

from .base import MacroFeatureEngine
from ...data.models import MacroSeries, MacroFeature

class GrowthFeatures(MacroFeatureEngine):
    """Computes growth-related macroeconomic features."""

    def compute(self, macro_data: Dict[str, List[MacroSeries]], **kwargs) -> List[MacroFeature]:
        """Compute growth features from macro series data."""
        features = []

        # GDP-based features
        if 'IN_GDP_YOY' in macro_data:
            gdp_features = self._compute_gdp_features(macro_data['IN_GDP_YOY'])
            features.extend(gdp_features)

        # Industrial production features
        if 'IN_IIP_YOY' in macro_data:
            iip_features = self._compute_iip_features(macro_data['IN_IIP_YOY'])
            features.extend(iip_features)

        # PMI features (if available)
        if 'IN_PMI' in macro_data:
            pmi_features = self._compute_pmi_features(macro_data['IN_PMI'])
            features.extend(pmi_features)

        # Growth trends and acceleration
        if 'IN_GDP_YOY' in macro_data:
            trend_features = self._compute_growth_trends(macro_data['IN_GDP_YOY'])
            features.extend(trend_features)

        return features

    def _compute_gdp_features(self, gdp_series: List[MacroSeries]) -> List[MacroFeature]:
        """Compute GDP-based growth features."""
        features = []
        df = self.transform_series_to_dataframe(gdp_series)

        if df.empty:
            return features

        series = df['value']

        # Year-over-year GDP growth
        for timestamp, value in series.dropna().items():
            features.append(self.create_macro_feature(
                name='gdp_yoy',
                timestamp=timestamp,
                value=value,
                source_series=['IN_GDP_YOY'],
                transform='level',
                frequency='quarterly'
            ))

        # GDP growth trend (4-quarter moving average)
        trend = self.apply_transform(series, {'transform': 'rolling_mean', 'window': 4})
        for timestamp, value in trend.dropna().items():
            features.append(self.create_macro_feature(
                name='gdp_trend',
                timestamp=timestamp,
                value=value,
                source_series=['IN_GDP_YOY'],
                transform='rolling_mean',
                frequency='quarterly'
            ))

        # GDP growth acceleration (quarterly change in growth rate)
        acceleration = series.diff()
        for timestamp, value in acceleration.dropna().items():
            features.append(self.create_macro_feature(
                name='gdp_acceleration',
                timestamp=timestamp,
                value=value,
                source_series=['IN_GDP_YOY'],
                transform='first_difference',
                frequency='quarterly'
            ))

        return features

    def _compute_iip_features(self, iip_series: List[MacroSeries]) -> List[MacroFeature]:
        """Compute Industrial Production Index features."""
        features = []
        df = self.transform_series_to_dataframe(iip_series)

        if df.empty:
            return features

        series = df['value']

        # IIP year-over-year change
        yoy_change = self.apply_transform(series, {'transform': 'yoy_change'})
        for timestamp, value in yoy_change.dropna().items():
            features.append(self.create_macro_feature(
                name='iip_yoy',
                timestamp=timestamp,
                value=value,
                source_series=['IN_IIP_YOY'],
                transform='yoy_change',
                frequency='monthly'
            ))

        # IIP trend (3-month moving average)
        trend = self.apply_transform(series, {'transform': 'rolling_mean', 'window': 3})
        for timestamp, value in trend.dropna().items():
            features.append(self.create_macro_feature(
                name='iip_trend',
                timestamp=timestamp,
                value=value,
                source_series=['IN_IIP_YOY'],
                transform='rolling_mean',
                frequency='monthly'
            ))

        return features

    def _compute_pmi_features(self, pmi_series: List[MacroSeries]) -> List[MacroFeature]:
        """Compute PMI-based features."""
        features = []
        df = self.transform_series_to_dataframe(pmi_series)

        if df.empty:
            return features

        series = df['value']

        # PMI level (expansion > 50, contraction < 50)
        for timestamp, value in series.dropna().items():
            features.append(self.create_macro_feature(
                name='pmi_level',
                timestamp=timestamp,
                value=value,
                source_series=['IN_PMI'],
                transform='level',
                frequency='monthly'
            ))

        # PMI momentum (change from previous month)
        momentum = series.diff()
        for timestamp, value in momentum.dropna().items():
            features.append(self.create_macro_feature(
                name='pmi_momentum',
                timestamp=timestamp,
                value=value,
                source_series=['IN_PMI'],
                transform='first_difference',
                frequency='monthly'
            ))

        return features

    def _compute_growth_trends(self, gdp_series: List[MacroSeries]) -> List[MacroFeature]:
        """Compute growth trend and cyclical features."""
        features = []
        df = self.transform_series_to_dataframe(gdp_series)

        if df.empty or len(df) < 8:  # Need at least 2 years of quarterly data
            return features

        series = df['value']

        # Growth cycle position (relative to long-term trend)
        trend_window = 8  # 2 years of quarterly data
        long_term_trend = series.rolling(window=trend_window, min_periods=trend_window).mean()

        cycle_position = series - long_term_trend
        z_cycle = (cycle_position - cycle_position.rolling(window=trend_window, min_periods=4).mean()) / \
                  cycle_position.rolling(window=trend_window, min_periods=4).std()

        for timestamp, value in z_cycle.dropna().items():
            features.append(self.create_macro_feature(
                name='growth_cycle_position',
                timestamp=timestamp,
                value=value,
                source_series=['IN_GDP_YOY'],
                transform='cycle_z_score',
                frequency='quarterly'
            ))

        return features

    def classify_growth_regime(self, growth_features: List[MacroFeature]) -> str:
        """Classify current growth regime."""
        # Extract latest GDP growth
        gdp_features = [f for f in growth_features if f.feature_name == 'gdp_yoy']
        if not gdp_features:
            return 'unknown'

        latest_gdp = max(gdp_features, key=lambda x: x.timestamp)
        growth_rate = latest_gdp.value

        # Extract IIP trend
        iip_trend_features = [f for f in growth_features if f.feature_name == 'iip_trend']
        industrial_trend = None
        if iip_trend_features:
            latest_iip = max(iip_trend_features, key=lambda x: x.timestamp)
            industrial_trend = latest_iip.value

        # Simple regime classification
        if growth_rate < 4.0:
            if industrial_trend and industrial_trend < 0:
                return 'recession'
            else:
                return 'slowdown'
        elif 4.0 <= growth_rate < 6.0:
            return 'moderate_growth'
        elif 6.0 <= growth_rate < 8.0:
            return 'strong_growth'
        else:
            return 'boom'