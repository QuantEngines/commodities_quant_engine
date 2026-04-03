from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from ...config.settings import settings
from ...data.models import DirectionalSignal
from ...features.base import CarryFeatures, MomentumFeatures, VolatilityFeatures


class DirectionalAlphaEngine:
    """Directional alpha engine with governed parameterized weights."""

    def __init__(self, parameter_state: Optional[Dict[str, object]] = None):
        self.mom_engine = MomentumFeatures()
        self.carry_engine = CarryFeatures()
        self.vol_engine = VolatilityFeatures()
        self.horizons = settings.directional_horizons
        self.parameter_state = parameter_state or {}
        self.models: Dict[int, Ridge] = {}

    def build_feature_frame(self, data: pd.DataFrame) -> pd.DataFrame:
        feature_frame = pd.concat(
            [
                self.mom_engine.compute(data),
                self.carry_engine.compute(data),
                self.vol_engine.compute(data),
            ],
            axis=1,
        )
        required = settings.signal.directional_feature_names
        for feature_name in required:
            if feature_name not in feature_frame.columns:
                feature_frame[feature_name] = 0.0
        cleaned = feature_frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return cleaned.clip(lower=-settings.signal.feature_clip_abs, upper=settings.signal.feature_clip_abs)

    def fit(self, data: pd.DataFrame, target_horizon: int = 5):
        features_df = self.build_feature_frame(data)
        target = data["close"].pct_change(target_horizon).shift(-target_horizon)
        combined = features_df.join(target.rename("target"), how="inner").dropna()
        if len(combined) < settings.adaptation.min_sample_size:
            return
        model = Ridge(alpha=settings.adaptation.ridge_alpha)
        X = combined[settings.signal.directional_feature_names]
        y = combined["target"]
        model.fit(X, y)
        self.models[target_horizon] = model

    def predict(self, data: pd.DataFrame, commodity: str) -> List[DirectionalSignal]:
        feature_frame = self.build_feature_frame(data)
        latest_features = feature_frame.iloc[-1].to_dict()
        timestamp = data.index[-1].to_pydatetime() if isinstance(data.index, pd.DatetimeIndex) else datetime.now(timezone.utc).replace(tzinfo=None)
        return [
            self.generate_signal(commodity, latest_features, timestamp, horizon)
            for horizon in self.horizons
        ]

    def generate_signal(
        self,
        commodity: str,
        features: Dict[str, float] | pd.DataFrame,
        timestamp: datetime,
        horizon: int,
    ) -> DirectionalSignal:
        if isinstance(features, pd.DataFrame):
            feature_vector = features.iloc[-1].to_dict()
        else:
            feature_vector = {name: float(value) for name, value in features.items()}

        score = self._score_feature_vector(feature_vector, horizon)
        feature_quality = self._feature_quality(feature_vector)
        base_confidence = float(1.0 / (1.0 + np.exp(-abs(score) * settings.signal.confidence_scale)))
        confidence = float(max(0.0, min(1.0, base_confidence * (0.65 + 0.35 * feature_quality))))

        return DirectionalSignal(
            commodity=commodity,
            horizon=horizon,
            score=score,
            confidence=confidence,
            features={name: float(feature_vector.get(name, 0.0)) for name in settings.signal.directional_feature_names},
            timestamp=timestamp,
            model_version=str(self.parameter_state.get("version_id", "default")),
        )

    def _score_feature_vector(self, feature_vector: Dict[str, float], horizon: int) -> float:
        clipped_vector = {
            name: float(np.clip(feature_vector.get(name, 0.0), -settings.signal.feature_clip_abs, settings.signal.feature_clip_abs))
            for name in settings.signal.directional_feature_names
        }
        if horizon in self.models:
            ordered = pd.DataFrame([clipped_vector]).reindex(columns=settings.signal.directional_feature_names, fill_value=0.0)
            model_score = float(self.models[horizon].predict(ordered)[0])
            structural_prior = self._structural_prior(clipped_vector, horizon)
            return self._normalize_score(model_score, structural_prior)

        weights = self._weights_for_horizon(horizon)
        intercept = float(self.parameter_state.get("directional_intercepts", {}).get(str(horizon), settings.signal.directional_intercepts.get(str(horizon), 0.0)))
        score = intercept
        for feature_name, weight in weights.items():
            score += float(clipped_vector.get(feature_name, 0.0)) * float(weight)
        structural_prior = self._structural_prior(clipped_vector, horizon)
        return self._normalize_score(score, structural_prior)

    def _weights_for_horizon(self, horizon: int) -> Dict[str, float]:
        state_weights = self.parameter_state.get("directional_feature_weights", {})
        if str(horizon) in state_weights:
            return dict(state_weights[str(horizon)])
        return dict(settings.signal.directional_feature_weights.get(str(horizon), {}))

    def _structural_prior(self, feature_vector: Dict[str, float], horizon: int) -> float:
        trend_score = (
            0.40 * float(feature_vector.get("momentum_20d", 0.0))
            + 0.30 * float(feature_vector.get("trend_strength_20d", 0.0))
            + 0.15 * float(feature_vector.get("carry_yield", 0.0))
            + 0.15 * float(feature_vector.get("volume_trend_20d", 0.0))
        )
        reversal_score = (
            0.65 * float(feature_vector.get("short_reversal_5d", 0.0))
            + 0.35 * float(feature_vector.get("drawdown_20d", 0.0))
        )
        trend_weight = min(1.0, max(0.0, (horizon - 1) / 19.0))
        reversal_weight = 1.0 - trend_weight
        return trend_weight * trend_score + reversal_weight * reversal_score

    def _normalize_score(self, model_score: float, structural_prior: float) -> float:
        blend = float(settings.signal.structural_prior_blend)
        raw_score = (1.0 - blend) * model_score + blend * structural_prior
        score_cap = float(settings.signal.max_directional_score_abs)
        if score_cap <= 0:
            return float(raw_score)
        return float(np.tanh(raw_score / score_cap) * score_cap)

    def _feature_quality(self, feature_vector: Dict[str, float]) -> float:
        required = settings.signal.directional_feature_names
        non_zero = sum(abs(float(feature_vector.get(name, 0.0))) > 1e-12 for name in required)
        return non_zero / max(len(required), 1)
