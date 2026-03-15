"""
Macro Feature Engineering Base Classes

Extends the base FeatureEngine for macroeconomic data.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import pandas as pd
import numpy as np
from datetime import datetime, date

from ..base import FeatureEngine
from ...data.models import MacroFeature, MacroSeries
from ...config.settings import settings

class MacroFeatureEngine(FeatureEngine):
    """Base class for macroeconomic feature engineering."""

    def __init__(self, feature_config: Dict[str, Any]):
        self.config = feature_config
        self.feature_name = feature_config.get('name', self.__class__.__name__)
        self.enabled = feature_config.get('enabled', True)

    @abstractmethod
    def compute(self, macro_data: Dict[str, List[MacroSeries]], **kwargs) -> List[MacroFeature]:
        """Compute macro features from raw macro series data."""
        pass

    def transform_series_to_dataframe(self, series_list: List[MacroSeries]) -> pd.DataFrame:
        """Convert list of MacroSeries to time-indexed DataFrame."""
        if not series_list:
            return pd.DataFrame()

        df = pd.DataFrame([{
            'timestamp': s.timestamp,
            'value': s.value,
            'source': s.source,
            'is_revised': s.is_revised
        } for s in series_list])

        df = df.set_index('timestamp').sort_index()
        return df

    def apply_transform(self, series: pd.Series, transform_config: Dict[str, Any]) -> pd.Series:
        """Apply a transformation to a time series."""
        transform_type = transform_config.get('transform')

        if transform_type == 'yoy_change':
            return series.pct_change(365)  # Approximate year-over-year
        elif transform_type == 'mom_change':
            return series.pct_change(30)   # Approximate month-over-month
        elif transform_type == 'level':
            return series
        elif transform_type == 'rolling_mean':
            window = transform_config.get('window', 3)
            return series.rolling(window=window, min_periods=1).mean()
        elif transform_type == 'pct_change':
            return series.pct_change()
        elif transform_type == 'second_difference':
            return series.diff().diff()
        elif transform_type == 'z_score':
            window = transform_config.get('window_days', 252)
            return (series - series.rolling(window=window, min_periods=30).mean()) / series.rolling(window=window, min_periods=30).std()
        else:
            return series

    def handle_missing_data(self, series: pd.Series, policy: str) -> pd.Series:
        """Apply missing data policy."""
        if policy == 'forward_fill':
            return series.fillna(method='ffill')
        elif policy == 'backward_fill':
            return series.fillna(method='bfill')
        elif policy == 'interpolate':
            return series.interpolate()
        elif policy == 'zero':
            return series.fillna(0)
        elif policy == 'drop':
            return series.dropna()
        else:
            return series

    def create_macro_feature(self, name: str, timestamp: datetime, value: float,
                           source_series: List[str], transform: str, **kwargs) -> MacroFeature:
        """Create a MacroFeature object."""
        return MacroFeature(
            feature_name=name,
            timestamp=timestamp,
            value=value,
            source_series=source_series,
            transform=transform,
            lag_days=kwargs.get('lag_days', 0),
            frequency=kwargs.get('frequency', 'daily'),
            missing_data_policy=kwargs.get('missing_policy', 'forward_fill'),
            metadata=kwargs.get('metadata', {})
        )

    def validate_feature_data(self, features: List[MacroFeature]) -> Dict[str, Any]:
        """Validate computed features."""
        if not features:
            return {"valid": False, "errors": ["No features computed"]}

        values = [f.value for f in features if f.value is not None]
        timestamps = [f.timestamp for f in features]

        validation = {
            "count": len(features),
            "valid_values": len(values),
            "missing_values": len(features) - len(values),
            "date_range": (min(timestamps), max(timestamps)) if timestamps else None,
            "outliers": self._detect_outliers(values) if values else [],
            "valid": len(values) > 0
        }

        return validation

    def _detect_outliers(self, values: List[float], threshold: float = 3.0) -> List[int]:
        """Detect outlier indices using z-score."""
        if len(values) < 3:
            return []

        mean_val = np.mean(values)
        std_val = np.std(values)

        if std_val == 0:
            return []

        z_scores = [(v - mean_val) / std_val for v in values]
        return [i for i, z in enumerate(z_scores) if abs(z) > threshold]