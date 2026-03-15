"""
FX Feature Engineering

Computes foreign exchange and external sector features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any
from datetime import datetime, date

from .base import MacroFeatureEngine
from ...data.models import MacroSeries, MacroFeature

class FXFeatures(MacroFeatureEngine):
    """Computes foreign exchange and external sector macroeconomic features."""

    def compute(self, macro_data: Dict[str, List[MacroSeries]], **kwargs) -> List[MacroFeature]:
        """Compute FX features from macro series data."""
        features = []

        # USD-INR features
        if 'FX_USD_INR' in macro_data:
            usd_inr_features = self._compute_usd_inr_features(macro_data['FX_USD_INR'])
            features.extend(usd_inr_features)

        # DXY features
        if 'FX_DXY' in macro_data:
            dxy_features = self._compute_dxy_features(macro_data['FX_DXY'])
            features.extend(dxy_features)

        # FX volatility features
        if 'FX_USD_INR' in macro_data:
            vol_features = self._compute_fx_volatility_features(macro_data['FX_USD_INR'])
            features.extend(vol_features)

        # Cross-currency relationships
        if 'FX_USD_INR' in macro_data and 'FX_DXY' in macro_data:
            relationship_features = self._compute_fx_relationships(
                macro_data['FX_USD_INR'], macro_data['FX_DXY']
            )
            features.extend(relationship_features)

        return features

    def _compute_usd_inr_features(self, usd_inr_series: List[MacroSeries]) -> List[MacroFeature]:
        """Compute USD-INR specific features."""
        features = []
        df = self.transform_series_to_dataframe(usd_inr_series)

        if df.empty:
            return features

        series = df['value']

        # Current USD-INR level
        for timestamp, value in series.dropna().items():
            features.append(self.create_macro_feature(
                name='usd_inr',
                timestamp=timestamp,
                value=value,
                source_series=['FX_USD_INR'],
                transform='level',
                frequency='daily'
            ))

        # USD-INR percentage change
        pct_change = self.apply_transform(series, {'transform': 'pct_change'})
        for timestamp, value in pct_change.dropna().items():
            features.append(self.create_macro_feature(
                name='usd_inr_change',
                timestamp=timestamp,
                value=value,
                source_series=['FX_USD_INR'],
                transform='pct_change',
                frequency='daily'
            ))

        # USD-INR trend (20-day moving average)
        trend = self.apply_transform(series, {'transform': 'rolling_mean', 'window': 20})
        for timestamp, value in trend.dropna().items():
            features.append(self.create_macro_feature(
                name='usd_inr_trend',
                timestamp=timestamp,
                value=value,
                source_series=['FX_USD_INR'],
                transform='rolling_mean',
                frequency='daily'
            ))

        return features

    def _compute_dxy_features(self, dxy_series: List[MacroSeries]) -> List[MacroFeature]:
        """Compute DXY (US Dollar Index) features."""
        features = []
        df = self.transform_series_to_dataframe(dxy_series)

        if df.empty:
            return features

        series = df['value']

        # Current DXY level
        for timestamp, value in series.dropna().items():
            features.append(self.create_macro_feature(
                name='dxy',
                timestamp=timestamp,
                value=value,
                source_series=['FX_DXY'],
                transform='level',
                frequency='daily'
            ))

        # DXY change
        pct_change = self.apply_transform(series, {'transform': 'pct_change'})
        for timestamp, value in pct_change.dropna().items():
            features.append(self.create_macro_feature(
                name='dxy_change',
                timestamp=timestamp,
                value=value,
                source_series=['FX_DXY'],
                transform='pct_change',
                frequency='daily'
            ))

        return features

    def _compute_fx_volatility_features(self, usd_inr_series: List[MacroSeries]) -> List[MacroFeature]:
        """Compute FX volatility features."""
        features = []
        df = self.transform_series_to_dataframe(usd_inr_series)

        if df.empty or len(df) < 30:
            return features

        returns = df['value'].pct_change().dropna()

        # Realized volatility (30-day rolling)
        realized_vol = returns.rolling(window=30, min_periods=10).std() * np.sqrt(252)  # Annualized
        for timestamp, value in realized_vol.dropna().items():
            features.append(self.create_macro_feature(
                name='usd_inr_volatility',
                timestamp=timestamp,
                value=value,
                source_series=['FX_USD_INR'],
                transform='realized_volatility',
                frequency='daily'
            ))

        # FX stress indicator (95th percentile of recent volatility)
        vol_95th = returns.rolling(window=60, min_periods=30).quantile(0.95)
        for timestamp, value in vol_95th.dropna().items():
            features.append(self.create_macro_feature(
                name='usd_inr_stress',
                timestamp=timestamp,
                value=value,
                source_series=['FX_USD_INR'],
                transform='rolling_percentile_95',
                frequency='daily'
            ))

        return features

    def _compute_fx_relationships(self, usd_inr_series: List[MacroSeries],
                                dxy_series: List[MacroSeries]) -> List[MacroFeature]:
        """Compute relationships between different FX variables."""
        features = []

        inr_df = self.transform_series_to_dataframe(usd_inr_series)
        dxy_df = self.transform_series_to_dataframe(dxy_series)

        if inr_df.empty or dxy_df.empty:
            return features

        # Align timestamps
        combined = pd.concat([inr_df['value'], dxy_df['value']], axis=1, keys=['INR', 'DXY'])
        combined = combined.dropna()

        # INR vs DXY correlation (60-day rolling)
        correlation = combined['INR'].rolling(window=60, min_periods=30).corr(combined['DXY'])
        for timestamp, value in correlation.dropna().items():
            features.append(self.create_macro_feature(
                name='inr_dxy_correlation',
                timestamp=timestamp,
                value=value,
                source_series=['FX_USD_INR', 'FX_DXY'],
                transform='rolling_correlation',
                frequency='daily'
            ))

        # INR deviation from DXY-driven expectation
        # Simple model: INR should move with DXY, deviations indicate local factors
        dxy_returns = combined['DXY'].pct_change()
        inr_returns = combined['INR'].pct_change()

        # Beta of INR vs DXY (rolling 60-day)
        beta_window = 60
        covariances = (inr_returns * dxy_returns).rolling(window=beta_window, min_periods=30)
        variances = (dxy_returns ** 2).rolling(window=beta_window, min_periods=30)
        beta = covariances.mean() / variances.mean()

        for timestamp, value in beta.dropna().items():
            features.append(self.create_macro_feature(
                name='inr_dxy_beta',
                timestamp=timestamp,
                value=value,
                source_series=['FX_USD_INR', 'FX_DXY'],
                transform='rolling_beta',
                frequency='daily'
            ))

        return features

    def classify_fx_regime(self, fx_features: List[MacroFeature]) -> str:
        """Classify current FX regime."""
        # Extract latest USD-INR and its trend
        inr_level_features = [f for f in fx_features if f.feature_name == 'usd_inr']
        inr_trend_features = [f for f in fx_features if f.feature_name == 'usd_inr_trend']

        if not inr_level_features:
            return 'unknown'

        latest_inr = max(inr_level_features, key=lambda x: x.timestamp)
        inr_level = latest_inr.value

        trend_value = None
        if inr_trend_features:
            latest_trend = max(inr_trend_features, key=lambda x: x.timestamp)
            trend_value = latest_trend.value

        # Simple regime classification based on INR levels and trends
        if inr_level > 85:  # Weak INR
            if trend_value and trend_value > inr_level:
                return 'inr_weakening'
            else:
                return 'inr_weak'
        elif inr_level < 75:  # Strong INR
            if trend_value and trend_value < inr_level:
                return 'inr_strengthening'
            else:
                return 'inr_strong'
        else:  # Neutral range
            if trend_value:
                if trend_value > inr_level + 0.5:
                    return 'inr_weakening'
                elif trend_value < inr_level - 0.5:
                    return 'inr_strengthening'
            return 'inr_stable'