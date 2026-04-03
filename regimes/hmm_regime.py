from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from ..config.settings import settings


@dataclass
class HMMRegimeClassification:
    label: str
    probability: float
    confidence: float
    features: Dict[str, float]


class HMMRegimeClassifier:
    """Hidden-state style regime classifier using GMM posteriors + Markov transitions."""

    FEATURE_COLUMNS = ["volatility_20d", "momentum_20d", "trend_strength_20d"]

    def classify(self, feature_frame: pd.DataFrame) -> Optional[HMMRegimeClassification]:
        if feature_frame.empty:
            return None
        if len(feature_frame) < int(settings.signal.hmm_min_history_rows):
            return None

        matrix = (
            feature_frame.reindex(columns=self.FEATURE_COLUMNS)
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
            .fillna(0.0)
        )
        if matrix.empty:
            return None

        n_states = max(2, min(int(settings.signal.hmm_states), len(matrix) // 25))
        if n_states < 2:
            return None

        model = GaussianMixture(
            n_components=n_states,
            covariance_type="full",
            random_state=42,
            reg_covar=1e-6,
            max_iter=300,
        )
        model.fit(matrix.values)
        posterior = model.predict_proba(matrix.values)
        states = np.argmax(posterior, axis=1)

        transition = self._transition_matrix(states=states, n_states=n_states)
        posterior_latest = posterior[-1]
        if len(states) > 1:
            transition_prior = transition[states[-2]]
            blend = float(settings.signal.hmm_transition_blend)
            combined = (1.0 - blend) * posterior_latest + blend * transition_prior
        else:
            combined = posterior_latest

        state_idx = int(np.argmax(combined))
        mask = states == state_idx
        state_profile = matrix.loc[mask].mean() if mask.any() else matrix.iloc[-1]
        features = {
            "volatility_20d": float(state_profile.get("volatility_20d", 0.0)),
            "momentum_20d": float(state_profile.get("momentum_20d", 0.0)),
            "trend_strength_20d": float(state_profile.get("trend_strength_20d", 0.0)),
        }
        label = self._label_from_profile(features)
        probability = float(np.clip(combined[state_idx], 0.0, 1.0))
        confidence = self._confidence(features, probability)
        return HMMRegimeClassification(
            label=label,
            probability=probability,
            confidence=confidence,
            features=features,
        )

    def _transition_matrix(self, states: np.ndarray, n_states: int) -> np.ndarray:
        counts = np.ones((n_states, n_states), dtype=float) * 1e-3
        for i in range(1, len(states)):
            counts[int(states[i - 1]), int(states[i])] += 1.0
        return counts / counts.sum(axis=1, keepdims=True)

    def _label_from_profile(self, features: Dict[str, float]) -> str:
        vol = float(features.get("volatility_20d", 0.0))
        mom = float(features.get("momentum_20d", 0.0))
        trend = float(features.get("trend_strength_20d", 0.0))

        if mom > 0.75 and trend > 0.5 and vol < 1.0:
            return "trend_following_bullish"
        if mom < -0.75 and trend < -0.5 and vol < 1.0:
            return "trend_following_bearish"
        if vol > 1.5:
            return "volatile_reversal"
        if abs(mom) < 0.35 and abs(trend) < 0.35:
            return "mean_reverting_rangebound"
        return "neutral"

    def _confidence(self, features: Dict[str, float], probability: float) -> float:
        vol = float(features.get("volatility_20d", 0.0))
        trend = abs(float(features.get("trend_strength_20d", 0.0)))
        confidence = probability + trend * 0.05 - max(0.0, vol - 1.0) * 0.03
        return float(np.clip(confidence, 0.2, 0.95))
