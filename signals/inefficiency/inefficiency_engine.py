from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from ...config.settings import settings
from ...data.models import InefficiencySignal


class InefficiencyEngine:
    """Pricing inefficiency detector with timestamp-safe rolling normalization."""

    def __init__(self):
        self.z_threshold = settings.signal.inefficiency_z_threshold
        self.window = settings.signal.inefficiency_window_days

    def detect_inefficiency(
        self,
        data: pd.DataFrame,
        commodity: str,
        fair_value: Optional[pd.Series] = None,
    ) -> InefficiencySignal:
        closes = data["close"].astype(float)
        fair_value = fair_value if fair_value is not None else self._hybrid_fair_value(data)
        deviation = closes - fair_value
        rolling_std = deviation.rolling(self.window, min_periods=max(5, self.window // 2)).std(ddof=0)
        rolling_median = deviation.rolling(self.window, min_periods=max(5, self.window // 2)).median()
        rolling_mad = (deviation - rolling_median).abs().rolling(self.window, min_periods=max(5, self.window // 2)).median()
        robust_scale = (1.4826 * rolling_mad).replace(0.0, np.nan)
        combined_scale = pd.concat([rolling_std, robust_scale], axis=1).mean(axis=1).replace(0.0, np.nan)
        deviation_z = (deviation / combined_scale).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5.0, 5.0)
        current_z = float(deviation_z.iloc[-1])
        persistence = int((deviation_z.abs() > self.z_threshold).iloc[-self.window :].sum())
        instability = bool(deviation_z.diff().abs().rolling(5, min_periods=2).mean().iloc[-1] > 0.75)
        fair_value_gap = float(deviation.iloc[-1] / closes.iloc[-1]) if closes.iloc[-1] else 0.0
        trailing_abs = deviation_z.abs().iloc[-self.window :]
        cross_sectional_score = float(trailing_abs.rank(pct=True).iloc[-1]) if len(trailing_abs) else None

        return InefficiencySignal(
            commodity=commodity,
            deviation_z=current_z,
            persistence=persistence,
            timestamp=data.index[-1].to_pydatetime() if isinstance(data.index, pd.DatetimeIndex) else datetime.now(timezone.utc).replace(tzinfo=None),
            instability_warning=instability,
            fair_value_gap=fair_value_gap,
            cross_sectional_score=cross_sectional_score,
        )

    def _hybrid_fair_value(self, data: pd.DataFrame) -> pd.Series:
        closes = data["close"].astype(float)
        fast_anchor = closes.ewm(span=max(3, self.window // 2), adjust=False).mean()
        slow_anchor = closes.rolling(self.window, min_periods=max(5, self.window // 2)).median()
        fair_value = 0.6 * fast_anchor + 0.4 * slow_anchor.fillna(fast_anchor)
        if "spot_close" in data.columns:
            spot = data["spot_close"].astype(float).replace(0.0, np.nan)
            fair_value = 0.5 * fair_value + 0.5 * spot.fillna(fair_value)
        return fair_value.fillna(closes)
