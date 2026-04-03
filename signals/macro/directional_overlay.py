from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

from ...config.commodity_universe import precious_metal_family
from ...config.settings import settings
from ...data.models import MacroDirectionalOverlay as MacroDirectionalOverlayModel, MacroFeature
from ...signals.directional.directional_alpha import DirectionalAlphaEngine


class MacroDirectionalOverlay:
    """Macro-aware directional adjustment that uses only known information at signal time."""

    def __init__(self, base_directional_engine: DirectionalAlphaEngine):
        self.base_engine = base_directional_engine
        self.macro_sensitivities = settings.macro.commodity_sensitivities

    def enhance_directional_signal(
        self,
        commodity: str,
        features: Dict[str, float],
        macro_features: List[MacroFeature],
        timestamp: datetime,
        horizon: int,
    ) -> MacroDirectionalOverlayModel:
        base_signal = self.base_engine.generate_signal(commodity, features, timestamp, horizon)
        macro_context = self._extract_macro_context(macro_features, timestamp)
        adjustment = self._compute_macro_directional_adjustment(commodity, base_signal.score, macro_context, horizon)
        adjusted_score = base_signal.score + adjustment["adjustment"]
        return MacroDirectionalOverlayModel(
            commodity=commodity,
            horizon=horizon,
            base_score=base_signal.score,
            macro_adjustment=adjustment["adjustment"],
            adjusted_score=adjusted_score,
            macro_alignment=adjustment["alignment"],
            confidence_multiplier=adjustment["confidence_multiplier"],
            key_macro_factors=adjustment["factors"],
            timestamp=timestamp,
        )

    def _extract_macro_context(self, macro_features: List[MacroFeature], timestamp: datetime) -> Dict[str, float]:
        latest_by_feature: Dict[str, MacroFeature] = {}
        for feature in macro_features:
            if feature.timestamp <= timestamp:
                current = latest_by_feature.get(feature.feature_name)
                if current is None or feature.timestamp > current.timestamp:
                    latest_by_feature[feature.feature_name] = feature
        return {name: feature.value for name, feature in latest_by_feature.items()}

    def _compute_macro_directional_adjustment(
        self,
        commodity: str,
        base_score: float,
        macro_context: Dict[str, float],
        horizon: int,
    ) -> Dict[str, Any]:
        sensitivities = self.macro_sensitivities.get(commodity, [])
        commodity_family = self._commodity_family(commodity)
        base_direction = "bullish" if base_score >= 0 else "bearish"
        adjustment = 0.0
        alignment = 0.0
        confidence_multiplier = 1.0
        factors: List[str] = []

        real_rate = macro_context.get("real_rate_in")
        if real_rate is not None and "real_rates" in sensitivities:
            if precious_metal_family(commodity):
                if base_direction == "bullish" and real_rate < 1.0:
                    adjustment += 0.20
                    alignment += 0.35
                    factors.append("Lower real rates support precious metals")
                elif base_direction == "bullish" and real_rate > 2.0:
                    adjustment -= 0.20
                    alignment -= 0.35
                    factors.append("Higher real rates oppose the bullish case")

        growth = macro_context.get("growth_cycle_position", macro_context.get("gdp_yoy"))
        if growth is not None and "growth_expectations" in sensitivities:
            if base_direction == "bullish" and growth > 0.5:
                adjustment += 0.15
                alignment += 0.25
                factors.append("Growth backdrop supports cyclical demand")
            elif base_direction == "bullish" and growth < -0.5:
                adjustment -= 0.15
                alignment -= 0.25
                factors.append("Growth slowdown weighs on cyclical demand")

        usd_inr = macro_context.get("usd_inr")
        if usd_inr is not None and "usd_strength" in sensitivities and not precious_metal_family(commodity):
            if usd_inr > 83.0 and base_direction == "bullish":
                adjustment -= 0.10
                alignment -= 0.15
                factors.append("INR weakness raises imported-cost pressure")

        if macro_context.get("event_risk_window", 0.0) > 0.0:
            confidence_multiplier *= 0.85
            factors.append("Upcoming macro event reduces near-term confidence")

        ovx_zscore = macro_context.get("ovx_zscore")
        ovx_momentum = macro_context.get("ovx_momentum_10d")
        if commodity_family == "energy" and (ovx_zscore is not None or ovx_momentum is not None):
            elevated_ovx = (ovx_zscore is not None and ovx_zscore > 1.0) or (ovx_momentum is not None and ovx_momentum > 0.08)
            if base_direction == "bullish" and elevated_ovx:
                adjustment -= 0.12
                alignment -= 0.20
                factors.append("Elevated OVX conflicts with bullish energy conviction")
            elif base_direction == "bearish" and elevated_ovx:
                adjustment += 0.08
                alignment += 0.12
                factors.append("Elevated OVX supports bearish energy stance")

        bdi_momentum = macro_context.get("bdi_momentum_20d")
        if commodity_family in {"base_metals", "agri"} and bdi_momentum is not None:
            if base_direction == "bullish" and bdi_momentum > 0.03:
                adjustment += 0.10
                alignment += 0.18
                factors.append("Rising BDI supports cyclical commodity demand")
            elif base_direction == "bullish" and bdi_momentum < -0.03:
                adjustment -= 0.10
                alignment -= 0.18
                factors.append("Falling BDI weakens cyclical demand case")
            elif base_direction == "bearish" and bdi_momentum < -0.03:
                adjustment += 0.08
                alignment += 0.12
                factors.append("Falling BDI aligns with softer demand outlook")

        if macro_context.get("bdi_shock_flag", 0.0) > 0.5 or macro_context.get("ovx_shock_flag", 0.0) > 0.5:
            confidence_multiplier *= 0.90
            factors.append("Cross-asset shock flag reduces directional confidence")

        horizon_scale = min(1.0, max(0.25, horizon / 20.0))
        return {
            "adjustment": adjustment * horizon_scale,
            "alignment": max(-1.0, min(1.0, alignment)),
            "confidence_multiplier": confidence_multiplier,
            "factors": factors,
        }

    def _commodity_family(self, commodity: str) -> str:
        config = settings.commodities.get(commodity)
        if config is None:
            return "default"
        return str(config.segment).lower()
