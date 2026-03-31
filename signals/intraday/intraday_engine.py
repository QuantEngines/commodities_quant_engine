"""
IntradayFactorRotationEngine — Intraday signal generation within daily regime frames.

Architecture:
1. Daily: Run full signal cycle, establish regime + factor weights
2. Intraday (1H, 4H): Reuse daily regime/regime weights, compute intraday tactical alphas
3. Output: Intraday directional scores within regime boundaries

Prevents whipsaw: Intraday signals respect daily regime bias, don't flip contra-regime.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config.settings import settings
from ..data.models import DirectionalSignal
from ..signals.directional.directional_alpha import DirectionalAlphaEngine


class IntradayFeatureWindows:
    """Time windows for intraday signal computation."""

    # Supported intraday intervals
    INTERVALS = {
        "1H": 60,      # 1-hour bars
        "4H": 240,     # 4-hour bars
        "1D": 1440,    # Daily bars (baseline)
    }

    def __init__(self, interval: str = "1H"):
        if interval not in self.INTERVALS:
            raise ValueError(f"Invalid interval: {interval}. Must be in {list(self.INTERVALS.keys())}")
        self.interval = interval
        self.minutes = self.INTERVALS[interval]

    def is_intraday(self) -> bool:
        """Check if interval is intraday (not daily or lower frequency)."""
        return self.interval in ["1H", "4H"]


class IntradayDirectionalAlpha:
    """Compute high-frequency directional alpha within daily regime."""

    def __init__(self):
        self.directional_engine = DirectionalAlphaEngine()

    def compute_intraday_signal(
        self,
        intraday_data: pd.DataFrame,
        daily_regime: str,
        daily_factor_weights: Dict[str, float],
        interval: str = "1H",
    ) -> Optional[DirectionalSignal]:
        """
        Compute intraday directional signal.

        Args:
            intraday_data: OHLCV DataFrame at intraday frequency (1H, 4H, etc.)
            daily_regime: Daily regime label (from daily signal cycle)
            daily_factor_weights: Factor weights for current regime
            interval: Intraday interval (1H, 4H)

        Returns:
            DirectionalSignal at intraday level, or None if insufficient data
        """
        if intraday_data.empty or len(intraday_data) < 5:
            return None

        # Build intraday feature frame
        feature_frame = self.directional_engine.build_feature_frame(intraday_data)
        latest_features = feature_frame.iloc[-1].to_dict()

        # Compute intraday momentum (mean reversion, trend strength)
        momentum = self._compute_intraday_momentum(intraday_data, interval)

        # Regime-constrained score: intraday momentum adjusted by daily regime
        regime_bias = {"bullish": 0.5, "neutral": 0.0, "bearish": -0.5}.get(
            daily_regime, 0.0
        )
        intraday_score = momentum + regime_bias * 0.2  # Regime provides 20% bias

        # Clamp to [-1, 1]
        intraday_score = max(-1.0, min(1.0, intraday_score))

        # Confidence: lower for intraday (higher noise-to-signal ratio)
        confidence = 0.6 + (abs(momentum) * 0.3)  # Range [0.6, 0.9]

        return DirectionalSignal(
            commodity="",  # Will be filled by caller
            horizon=1 if interval == "1H" else 4,  # Intraday horizon
            score=intraday_score,
            confidence=confidence,
            features=latest_features,
            timestamp=datetime.now(),
            model_version="intraday_v1",
        )

    def _compute_intraday_momentum(self, data: pd.DataFrame, interval: str) -> float:
        """
        Compute intraday momentum signal.

        Metrics:
        - Mean reversion: Close vs MA (faster reversion in thin hours)
        - Trend strength: Close > Open for N bars
        - Volatility regime: High vol = mean reversion, Low vol = trend
        """
        if len(data) < 3:
            return 0.0

        closes = data["close"].values
        recent_ma = np.mean(closes[-5:])
        current_price = closes[-1]

        # Mean reversion component
        ma_deviation = (current_price - recent_ma) / (np.std(closes[-5:]) + 1e-8)
        mean_reversion_signal = -ma_deviation * 0.3  # Reverting signal

        # Trend component (bars above/below open)
        bars_up = sum(1 for i in range(len(closes) - 1) if data["close"].iloc[i] > data["open"].iloc[i])
        trend_signal = (bars_up / len(closes)) - 0.5  # Normalize to [-0.5, 0.5]

        # Combine
        momentum = (mean_reversion_signal + trend_signal * 0.5) / 2.0
        return np.tanh(momentum)  # Smooth to [-1, 1]


class IntradayFactorRotationEngine:
    """
    Main engine: Generates intraday signals synchronized with daily regime.

    Workflow:
    1. Daily signal cycle runs → regime + factor weights established
    2. Throughout day: Intraday bars accumulate
    3. On each intraday bar: Compute tactical alpha, constrain by daily regime
    4. Output: Cross-timeframe signal with daily conviction + intraday timing
    """

    def __init__(self):
        self.intraday_alpha = IntradayDirectionalAlpha()

    def generate_intraday_signal_package(
        self,
        commodity: str,
        daily_signal: Dict,  # Result from daily run_signal_cycle
        intraday_price_data: pd.DataFrame,
        interval: str = "1H",
    ) -> Dict:
        """
        Generate complete intraday signal package.

        Args:
            commodity: Commodity symbol
            daily_signal: Daily signal package (includes regime, factor weights)
            intraday_price_data: Intraday OHLCV (1H or 4H bars)
            interval: Intraday interval

        Returns:
            Dict with:
            - daily_regime: Same as daily
            - daily_confidence: Same as daily
            - intraday_score: Tactical alpha
            - combined_signal: Weighted combination (daily conviction + intraday timing)
            - suggested_entry: Specific intraday entry signal
        """
        # Extract daily signal info
        daily_regime = daily_signal.get("regime_label", "neutral")
        daily_score = daily_signal.get("composite_score", 0.0)
        daily_confidence = daily_signal.get("confidence_score", 0.5)
        factor_weights = daily_signal.get("factor_weights", {})

        # Compute intraday alpha
        intraday_signal = self.intraday_alpha.compute_intraday_signal(
            intraday_data=intraday_price_data,
            daily_regime=daily_regime,
            daily_factor_weights=factor_weights,
            interval=interval,
        )

        if intraday_signal is None:
            intraday_score = 0.0
            intraday_confidence = 0.3
        else:
            intraday_score = intraday_signal.score
            intraday_confidence = intraday_signal.confidence

        # Combine daily conviction with intraday timing
        # Daily weight: 70%, Intraday: 30%
        combined_signal = (daily_score * 0.7) + (intraday_score * 0.3)
        combined_confidence = (daily_confidence * 0.7) + (intraday_confidence * 0.3)

        # Entry signal: Intraday entry only if aligned with daily regime
        entry_signal = self._determine_entry_signal(
            daily_regime, daily_score, intraday_score, combined_signal
        )

        return {
            "commodity": commodity,
            "timestamp": datetime.now(),
            "interval": interval,
            "daily_regime": daily_regime,
            "daily_score": daily_score,
            "daily_confidence": daily_confidence,
            "intraday_score": intraday_score,
            "intraday_confidence": intraday_confidence,
            "combined_signal": combined_signal,
            "combined_confidence": combined_confidence,
            "entry_signal": entry_signal,
            "is_intraday_entry_valid": entry_signal != "hold",
        }

    def _determine_entry_signal(
        self,
        daily_regime: str,
        daily_score: float,
        intraday_score: float,
        combined: float,
    ) -> str:
        """
        Determine specific intraday entry signal.

        Rules:
        - Daily regime must be non-neutral (bullish/bearish)
        - Intraday score must align with daily regime
        - Combined score above threshold
        """
        if daily_regime == "neutral":
            return "hold"  # No clear regime bias

        if combined < 0.05:
            return "hold"  # Combined signal too weak

        if daily_regime == "bullish":
            if intraday_score > 0.1:
                return "long"
            elif intraday_score < -0.3:
                return "hold"  # Intraday bearish contra-daily
            else:
                return "hold|long"  # Weak long

        elif daily_regime == "bearish":
            if intraday_score < -0.1:
                return "short"
            elif intraday_score > 0.3:
                return "hold"  # Intraday bullish contra-daily
            else:
                return "hold|short"  # Weak short

        return "hold"


# Singleton instance
intraday_factor_rotation_engine = IntradayFactorRotationEngine()
