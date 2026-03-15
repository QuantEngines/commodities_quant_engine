from __future__ import annotations

from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ..config.settings import settings
from ..data.models import RegimeState
from ..features.base import MomentumFeatures, VolatilityFeatures


class RegimeEngine:
    """Research-oriented regime classifier using only information available at time t."""

    def __init__(self):
        self.vol_engine = VolatilityFeatures()
        self.mom_engine = MomentumFeatures()
        self.window = settings.signal.regime_window_days

    def detect_regime(self, data: pd.DataFrame, commodity: str) -> RegimeState:
        subset = data.tail(self.window)
        feature_frame = self._build_feature_frame(subset)
        return self.detect_regime_from_features(
            feature_frame.iloc[-1].to_dict(),
            commodity=commodity,
            timestamp=subset.index[-1].to_pydatetime() if isinstance(subset.index, pd.DatetimeIndex) else datetime.utcnow(),
        )

    def detect_regime_from_features(
        self,
        features: Dict[str, float],
        commodity: Optional[str] = None,
        timestamp: Optional[datetime] = None,
    ) -> RegimeState:
        current_vol = float(features.get("volatility_20d", 0.0))
        current_mom = float(features.get("momentum_20d", 0.0))
        trend_strength = float(features.get("trend_strength_20d", 0.0))

        if current_mom > 0.75 and trend_strength > 0.5 and current_vol < 1.0:
            label = "trend_following_bullish"
            prob = 0.72
        elif current_mom < -0.75 and trend_strength < -0.5 and current_vol < 1.0:
            label = "trend_following_bearish"
            prob = 0.72
        elif current_vol > 1.5:
            label = "volatile_reversal"
            prob = 0.78
        elif abs(current_mom) < 0.35 and abs(trend_strength) < 0.35:
            label = "mean_reverting_rangebound"
            prob = 0.62
        else:
            label = "neutral"
            prob = 0.50

        confidence = min(0.95, max(0.2, prob + abs(trend_strength) * 0.05 - max(0.0, current_vol - 1.0) * 0.03))
        return RegimeState(
            label=label,
            probability=prob,
            confidence=confidence,
            features={
                "volatility_20d": current_vol,
                "momentum_20d": current_mom,
                "trend_strength_20d": trend_strength,
            },
            timestamp=timestamp or datetime.utcnow(),
        )

    def _build_feature_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        return pd.concat(
            [
                self.vol_engine.compute(data),
                self.mom_engine.compute(data),
            ],
            axis=1,
        ).replace([np.inf, -np.inf], np.nan).fillna(0.0)
