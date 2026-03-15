from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from ...config.settings import settings
from ...data.models import MacroFeature, MacroRegimeState, RegimeState
from ...regimes.regime_engine import RegimeEngine


class MacroRegimeOverlay:
    """Macro overlay for regime classification without peeking into future macro releases."""

    def __init__(self, base_regime_engine: RegimeEngine):
        self.base_engine = base_regime_engine
        self.macro_sensitivities = settings.macro.commodity_sensitivities

    def detect_macro_regime(
        self,
        commodity: str,
        features: Dict[str, float],
        macro_features: List[MacroFeature],
        timestamp: datetime,
    ) -> MacroRegimeState:
        base_regime = self.base_engine.detect_regime_from_features(features, commodity=commodity, timestamp=timestamp)
        macro_context = self._extract_macro_context(macro_features, timestamp)
        adjustment = self._compute_macro_regime_adjustment(commodity, base_regime, macro_context)
        combined_label = self._create_combined_regime_label(base_regime.label, adjustment["macro_regime"])
        adjusted_probability = self._adjust_probability(base_regime.probability, adjustment["contribution"])
        return MacroRegimeState(
            base_regime=base_regime.label,
            macro_overlay=adjustment["macro_regime"],
            combined_label=combined_label,
            probability=adjusted_probability,
            confidence=min(0.95, base_regime.confidence + abs(adjustment["contribution"]) * 0.1),
            macro_contribution=adjustment["contribution"],
            key_macro_drivers=adjustment["drivers"],
            timestamp=timestamp,
            features=base_regime.features,
        )

    def _extract_macro_context(self, macro_features: List[MacroFeature], timestamp: datetime) -> Dict[str, float]:
        latest_by_feature: Dict[str, MacroFeature] = {}
        for feature in macro_features:
            if feature.timestamp <= timestamp:
                current = latest_by_feature.get(feature.feature_name)
                if current is None or feature.timestamp > current.timestamp:
                    latest_by_feature[feature.feature_name] = feature
        return {name: feature.value for name, feature in latest_by_feature.items()}

    def _compute_macro_regime_adjustment(
        self,
        commodity: str,
        base_regime: RegimeState,
        macro_context: Dict[str, float],
    ) -> Dict[str, Any]:
        sensitivities = self.macro_sensitivities.get(commodity, [])
        contribution = 0.0
        drivers: List[str] = []
        macro_regime = "neutral"

        inflation = macro_context.get("cpi_yoy")
        if inflation is not None and "inflation_expectations" in sensitivities:
            if inflation > 5.5:
                contribution += 0.20
                macro_regime = "inflationary"
                drivers.append(f"Inflation is elevated at {inflation:.1f}%")
            elif inflation < 3.0:
                contribution -= 0.10

        growth = macro_context.get("gdp_yoy", macro_context.get("growth_cycle_position"))
        if growth is not None and "growth_expectations" in sensitivities:
            if growth > 0.5:
                contribution += 0.20
                macro_regime = "growth_supportive"
                drivers.append("Growth impulse supports cyclical demand")
            elif growth < -0.5:
                contribution -= 0.25
                macro_regime = "growth_slowdown"
                drivers.append("Growth impulse is weakening")

        real_rate = macro_context.get("real_rate_in")
        if real_rate is not None and "real_rates" in sensitivities:
            if real_rate < 1.0:
                contribution += 0.15
                macro_regime = "easy_real_rates"
                drivers.append(f"Real rates are supportive at {real_rate:.1f}%")
            elif real_rate > 2.0:
                contribution -= 0.20
                macro_regime = "tight_real_rates"
                drivers.append(f"Real rates are restrictive at {real_rate:.1f}%")

        if macro_context.get("macro_sentiment_score", 0.0) < -0.4 and "risk_off" in sensitivities:
            contribution += 0.10
            macro_regime = "risk_off"
            drivers.append("Risk-off tone is supporting defensive demand")

        return {
            "macro_regime": macro_regime,
            "contribution": max(-0.5, min(0.5, contribution)),
            "drivers": drivers,
        }

    def _create_combined_regime_label(self, base_regime: str, macro_regime: str) -> str:
        if macro_regime == "neutral":
            return base_regime
        return f"{base_regime} + {macro_regime}"

    def _adjust_probability(self, base_probability: float, macro_contribution: float) -> float:
        return max(0.10, min(0.90, base_probability + macro_contribution * 0.15))
