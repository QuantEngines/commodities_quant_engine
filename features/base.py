from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np
import pandas as pd


class FeatureEngine(ABC):
    """Base class for timestamp-safe feature engineering."""

    @abstractmethod
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        raise NotImplementedError

    def rolling_zscore(self, series: pd.Series, window: int = 252, min_periods: int = 20) -> pd.Series:
        mean = series.rolling(window=window, min_periods=min_periods).mean().shift(1)
        std = series.rolling(window=window, min_periods=min_periods).std(ddof=0).shift(1)
        standardized = (series - mean) / std.replace(0.0, np.nan)
        return self.clip_extremes(standardized.replace([np.inf, -np.inf], np.nan).fillna(0.0))

    def clip_extremes(self, series: pd.Series, clip_abs: float = 5.0) -> pd.Series:
        return series.clip(lower=-clip_abs, upper=clip_abs)


class MomentumFeatures(FeatureEngine):
    def compute(self, data: pd.DataFrame, windows: Iterable[int] = (5, 10, 20, 60)) -> pd.DataFrame:
        prices = data["close"].astype(float)
        features = {}
        for window in windows:
            raw_momentum = prices.pct_change(window)
            features[f"momentum_{window}d"] = self.rolling_zscore(raw_momentum)
            features[f"roc_{window}d"] = raw_momentum.fillna(0.0)
        drawdown = prices / prices.cummax() - 1.0
        features["drawdown_20d"] = drawdown.rolling(20, min_periods=5).min().fillna(0.0)
        features["short_reversal_5d"] = self.rolling_zscore((-prices.pct_change(5)).fillna(0.0), window=126, min_periods=20)
        trend_strength = prices.pct_change(20) / prices.pct_change().rolling(20, min_periods=5).std().replace(0.0, np.nan)
        features["trend_strength_20d"] = self.clip_extremes(trend_strength.replace([np.inf, -np.inf], np.nan).fillna(0.0))
        return pd.DataFrame(features, index=data.index)


class VolatilityFeatures(FeatureEngine):
    def compute(self, data: pd.DataFrame, windows: Iterable[int] = (5, 10, 20, 60)) -> pd.DataFrame:
        returns = data["close"].pct_change().fillna(0.0)
        features = {}
        for window in windows:
            realized = returns.rolling(window, min_periods=max(3, window // 2)).std(ddof=0)
            features[f"volatility_{window}d"] = self.rolling_zscore(realized)
            features[f"realized_vol_{window}d"] = (realized * np.sqrt(252)).fillna(0.0)
        features["volume_trend_20d"] = self.rolling_zscore(
            data["volume"].astype(float).pct_change().rolling(20, min_periods=5).mean().fillna(0.0),
            window=126,
            min_periods=20,
        )
        return pd.DataFrame(features, index=data.index)


class CarryFeatures(FeatureEngine):
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        carry = pd.Series(0.0, index=data.index)
        if "spot_close" in data.columns:
            spot = data["spot_close"].replace(0.0, np.nan)
            futures = data["close"].replace(0.0, np.nan)
            carry = ((spot - futures) / spot).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.DataFrame(
            {
                "carry_yield": carry,
                "roll_yield": carry.rolling(5, min_periods=1).mean().fillna(0.0),
            },
            index=data.index,
        )


class CurveFeatures(FeatureEngine):
    def compute(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        curve_slope = pd.Series(0.0, index=data.index)
        front_next_spread = pd.Series(0.0, index=data.index)
        if {"next_close", "close"}.issubset(data.columns):
            front_next_spread = (data["next_close"] - data["close"]).fillna(0.0)
            curve_slope = front_next_spread / data["close"].replace(0.0, np.nan)
            curve_slope = curve_slope.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return pd.DataFrame(
            {
                "curve_slope": curve_slope,
                "front_next_spread": front_next_spread,
            },
            index=data.index,
        )
