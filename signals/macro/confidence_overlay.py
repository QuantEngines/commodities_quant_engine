from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from ...config.commodity_universe import precious_metal_family
from ...config.settings import settings
from ...data.models import MacroConfidenceOverlay as MacroConfidenceOverlayModel, MacroEvent, MacroFeature


class MacroConfidenceOverlay:
    """Macro confidence adjustment with explicit leakage controls."""

    def __init__(self):
        self.macro_sensitivities = settings.macro.commodity_sensitivities

    def compute_confidence_adjustment(
        self,
        commodity: str,
        timestamp: datetime,
        macro_features: List[MacroFeature],
        macro_events: List[MacroEvent],
        base_regime: str,
        base_direction: str,
    ) -> MacroConfidenceOverlayModel:
        macro_context = self._extract_macro_context(macro_features, timestamp)
        alignment = self._compute_alignment_score(commodity, macro_context, base_direction)
        conflict = self._compute_conflict_score(commodity, macro_context, base_direction)
        event_risk = self._compute_event_risk_penalty(macro_events, timestamp)
        uncertainty = self._compute_uncertainty_penalty(macro_context)
        support_boost = self._compute_support_boost(alignment, macro_context)
        final_adjustment = max(
            -0.5,
            min(
                0.5,
                alignment * 0.30 - conflict * 0.40 - event_risk * 0.20 - uncertainty * 0.10 + support_boost,
            ),
        )

        return MacroConfidenceOverlayModel(
            commodity=commodity,
            timestamp=timestamp,
            macro_alignment_score=alignment,
            macro_conflict_score=conflict,
            event_risk_penalty=event_risk,
            news_uncertainty_penalty=uncertainty,
            support_boost=support_boost,
            final_confidence_adjustment=final_adjustment,
            key_macro_drivers=self._extract_key_drivers(macro_context, alignment),
            key_macro_risks=self._extract_key_risks(macro_context, event_risk, uncertainty),
            news_narrative_summary=self._generate_news_summary(macro_context),
        )

    def _extract_macro_context(self, macro_features: List[MacroFeature], timestamp: datetime) -> Dict[str, float]:
        latest_by_feature: Dict[str, MacroFeature] = {}
        for feature in macro_features:
            if feature.timestamp <= timestamp:
                current = latest_by_feature.get(feature.feature_name)
                if current is None or feature.timestamp > current.timestamp:
                    latest_by_feature[feature.feature_name] = feature
        return {name: feature.value for name, feature in latest_by_feature.items()}

    def _compute_alignment_score(self, commodity: str, macro_context: Dict[str, float], base_direction: str) -> float:
        alignment = 0.0
        sensitivities = self.macro_sensitivities.get(commodity, [])
        real_rate = macro_context.get("real_rate_in")
        if real_rate is not None and "real_rates" in sensitivities and precious_metal_family(commodity):
            if base_direction == "bullish" and real_rate < 1.0:
                alignment += 0.4
            elif base_direction == "bearish" and real_rate > 2.0:
                alignment += 0.25
        growth = macro_context.get("growth_cycle_position", macro_context.get("gdp_yoy"))
        if growth is not None and "growth_expectations" in sensitivities:
            if base_direction == "bullish" and growth > 0.5:
                alignment += 0.3
            elif base_direction == "bearish" and growth < -0.5:
                alignment += 0.3
        sentiment = macro_context.get("macro_sentiment_score")
        if sentiment is not None:
            if base_direction == "bullish" and sentiment > 0.2:
                alignment += 0.2
            elif base_direction == "bearish" and sentiment < -0.2:
                alignment += 0.2
        return min(1.0, alignment)

    def _compute_conflict_score(self, commodity: str, macro_context: Dict[str, float], base_direction: str) -> float:
        conflict = 0.0
        sensitivities = self.macro_sensitivities.get(commodity, [])
        real_rate = macro_context.get("real_rate_in")
        if real_rate is not None and "real_rates" in sensitivities and precious_metal_family(commodity):
            if base_direction == "bullish" and real_rate > 2.5:
                conflict += 0.6
        growth = macro_context.get("growth_cycle_position", macro_context.get("gdp_yoy"))
        if growth is not None and "growth_expectations" in sensitivities:
            if base_direction == "bullish" and growth < -0.5:
                conflict += 0.5
        return min(1.0, conflict)

    def _compute_event_risk_penalty(self, macro_events: List[MacroEvent], timestamp: datetime) -> float:
        penalty = 0.0
        for event in macro_events:
            days_until = (event.timestamp - timestamp).total_seconds() / 86400
            if 0 <= days_until <= settings.evaluation.event_window_days and event.expected_impact == "high":
                penalty += 0.3 / (days_until + 1.0)
        return min(1.0, penalty)

    def _compute_uncertainty_penalty(self, macro_context: Dict[str, float]) -> float:
        penalty = 0.0
        if macro_context.get("news_volume_burst", 1.0) > 2.0:
            penalty += 0.2
        if abs(macro_context.get("sentiment_volatility", 0.0)) > 0.5:
            penalty += 0.15
        return min(1.0, penalty)

    def _compute_support_boost(self, alignment: float, macro_context: Dict[str, float]) -> float:
        boost = 0.1 if alignment > 0.6 else 0.0
        if macro_context.get("macro_sentiment_score", 0.0) > 0.3:
            boost += 0.05
        return boost

    def _extract_key_drivers(self, macro_context: Dict[str, float], alignment: float) -> List[str]:
        drivers = []
        if alignment > 0.3:
            drivers.append("Macro context supports the current directional setup")
        if macro_context.get("real_rate_in") is not None:
            drivers.append(f"Real rate in India: {macro_context['real_rate_in']:.1f}%")
        if macro_context.get("macro_sentiment_score") is not None:
            drivers.append(f"Macro sentiment score: {macro_context['macro_sentiment_score']:.2f}")
        return drivers[:3]

    def _extract_key_risks(self, macro_context: Dict[str, float], event_risk: float, uncertainty: float) -> List[str]:
        risks = []
        if event_risk > 0.25:
            risks.append("High-impact scheduled macro event is near")
        if uncertainty > 0.15:
            risks.append("Macro information set is unusually noisy")
        if macro_context.get("news_volume_burst", 1.0) > 2.0:
            risks.append("News volume burst may destabilize short-horizon signals")
        return risks[:3]

    def _generate_news_summary(self, macro_context: Dict[str, float]) -> Optional[str]:
        sentiment = macro_context.get("macro_sentiment_score")
        if sentiment is None:
            return None
        tone = "positive" if sentiment > 0.2 else "negative" if sentiment < -0.2 else "mixed"
        return f"Macro news tone is {tone}."
