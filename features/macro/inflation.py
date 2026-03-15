"""
Inflation Feature Engineering

Computes inflation-related features from CPI, WPI, and other price indices.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, date

from .base import MacroFeatureEngine
from ...data.models import MacroSeries, MacroFeature

class InflationFeatures(MacroFeatureEngine):
    """Computes inflation-related macroeconomic features."""

    def compute(self, macro_data: Dict[str, List[MacroSeries]], **kwargs) -> List[MacroFeature]:
        """Compute inflation features from macro series data."""
        features = []

        # CPI-based features
        if 'IN_CPI_YOY' in macro_data:
            cpi_features = self._compute_cpi_features(macro_data['IN_CPI_YOY'])
            features.extend(cpi_features)

        if 'US_CPI_YOY' in macro_data:
            us_cpi_features = self._compute_us_cpi_features(macro_data['US_CPI_YOY'])
            features.extend(us_cpi_features)

        # Cross-country inflation differential
        if 'IN_CPI_YOY' in macro_data and 'US_CPI_YOY' in macro_data:
            diff_features = self._compute_inflation_differential(
                macro_data['IN_CPI_YOY'], macro_data['US_CPI_YOY']
            )
            features.extend(diff_features)

        # WPI features (India-specific)
        if 'IN_WPI_YOY' in macro_data:
            wpi_features = self._compute_wpi_features(macro_data['IN_WPI_YOY'])
            features.extend(wpi_features)

        return features

    def _compute_cpi_features(self, cpi_series: List[MacroSeries]) -> List[MacroFeature]:
        """Compute CPI-based inflation features."""
        features = []
        df = self.transform_series_to_dataframe(cpi_series)

        if df.empty:
            return features

        series = df['value']

        # Year-over-year change
        yoy_change = self.apply_transform(series, {'transform': 'yoy_change'})
        for timestamp, value in yoy_change.dropna().items():
            features.append(self.create_macro_feature(
                name='cpi_yoy',
                timestamp=timestamp,
                value=value,
                source_series=['IN_CPI_YOY'],
                transform='yoy_change',
                frequency='monthly'
            ))

        # Month-over-month change
        mom_change = self.apply_transform(series, {'transform': 'mom_change'})
        for timestamp, value in mom_change.dropna().items():
            features.append(self.create_macro_feature(
                name='cpi_mom',
                timestamp=timestamp,
                value=value,
                source_series=['IN_CPI_YOY'],
                transform='mom_change',
                frequency='monthly'
            ))

        # Inflation trend (3-month moving average)
        trend = self.apply_transform(series, {'transform': 'rolling_mean', 'window': 3})
        for timestamp, value in trend.dropna().items():
            features.append(self.create_macro_feature(
                name='cpi_trend',
                timestamp=timestamp,
                value=value,
                source_series=['IN_CPI_YOY'],
                transform='rolling_mean',
                frequency='monthly'
            ))

        # Inflation acceleration (second difference)
        acceleration = self.apply_transform(series, {'transform': 'second_difference'})
        for timestamp, value in acceleration.dropna().items():
            features.append(self.create_macro_feature(
                name='cpi_acceleration',
                timestamp=timestamp,
                value=value,
                source_series=['IN_CPI_YOY'],
                transform='second_difference',
                frequency='monthly'
            ))

        return features

    def _compute_us_cpi_features(self, cpi_series: List[MacroSeries]) -> List[MacroFeature]:
        """Compute US CPI features."""
        features = []
        df = self.transform_series_to_dataframe(cpi_series)

        if df.empty:
            return features

        series = df['value']

        # US year-over-year inflation
        yoy_change = self.apply_transform(series, {'transform': 'yoy_change'})
        for timestamp, value in yoy_change.dropna().items():
            features.append(self.create_macro_feature(
                name='us_cpi_yoy',
                timestamp=timestamp,
                value=value,
                source_series=['US_CPI_YOY'],
                transform='yoy_change',
                frequency='monthly'
            ))

        return features

    def _compute_inflation_differential(self, in_cpi: List[MacroSeries],
                                      us_cpi: List[MacroSeries]) -> List[MacroFeature]:
        """Compute India-US inflation differential."""
        features = []

        in_df = self.transform_series_to_dataframe(in_cpi)
        us_df = self.transform_series_to_dataframe(us_cpi)

        if in_df.empty or us_df.empty:
            return features

        # Align timestamps and compute differential
        combined = pd.concat([in_df['value'], us_df['value']], axis=1, keys=['IN', 'US'])
        combined = combined.dropna()

        differential = combined['IN'] - combined['US']

        for timestamp, value in differential.items():
            features.append(self.create_macro_feature(
                name='in_us_inflation_diff',
                timestamp=timestamp,
                value=value,
                source_series=['IN_CPI_YOY', 'US_CPI_YOY'],
                transform='difference',
                frequency='monthly'
            ))

        return features

    def _compute_wpi_features(self, wpi_series: List[MacroSeries]) -> List[MacroFeature]:
        """Compute WPI-based features (India-specific)."""
        features = []
        df = self.transform_series_to_dataframe(wpi_series)

        if df.empty:
            return features

        series = df['value']

        # WPI year-over-year change
        yoy_change = self.apply_transform(series, {'transform': 'yoy_change'})
        for timestamp, value in yoy_change.dropna().items():
            features.append(self.create_macro_feature(
                name='wpi_yoy',
                timestamp=timestamp,
                value=value,
                source_series=['IN_WPI_YOY'],
                transform='yoy_change',
                frequency='monthly'
            ))

        return features

    def compute_inflation_regime(self, inflation_features: List[MacroFeature]) -> str:
        """Classify current inflation regime."""
        # Extract latest CPI yoy
        cpi_features = [f for f in inflation_features if f.feature_name == 'cpi_yoy']
        if not cpi_features:
            return 'unknown'

        latest_cpi = max(cpi_features, key=lambda x: x.timestamp)
        inflation_rate = latest_cpi.value

        # Simple regime classification
        if inflation_rate < 2.0:
            return 'disinflationary'
        elif 2.0 <= inflation_rate < 4.0:
            return 'stable'
        elif 4.0 <= inflation_rate < 6.0:
            return 'elevated'
        else:
            return 'high_inflation'