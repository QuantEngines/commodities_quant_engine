from __future__ import annotations

from typing import List, Sequence

from ..classification import CommodityEventClassifier
from .entity_graph import extract_entity_graph_signals
from ..schemas.events import (
    AssetScope,
    CommodityEvent,
    EventDirection,
    EventPayload,
    EventType,
    PersistenceHorizon,
    VolatilityImplication,
)


class RuleBasedEventExtractor:
    """Commodity-aware structured extractor with transparent deterministic heuristics."""

    def __init__(self):
        self.classifier = CommodityEventClassifier()

    def extract(self, payload: EventPayload, commodity_scope: Sequence[str]) -> CommodityEvent:
        text = f"{payload.headline} {payload.body}".strip().lower()
        classification = self.classifier.classify(text)
        graph_entities = extract_entity_graph_signals(text)
        direction = self._infer_direction(classification.event_type, text)
        persistence = self._infer_persistence(text)
        strength = self._infer_strength(text)
        uncertainty = self._infer_uncertainty(text)
        volatility = self._infer_volatility_implication(classification.event_type, text)
        regime_relevance = self._infer_regime_relevance(classification.event_type, persistence, volatility)

        if "complex" in text or "basket" in text or "broad" in text:
            asset_scope = AssetScope.sector_basket
        elif "macro" in text or "global" in text or "central bank" in text:
            asset_scope = AssetScope.macro_wide
        else:
            asset_scope = AssetScope.single_commodity

        summary = f"{classification.event_type.value} identified with {direction.value} bias"
        return CommodityEvent(
            event_type=classification.event_type,
            commodity_scope=list(commodity_scope),
            asset_scope=asset_scope,
            expected_direction=direction,
            confidence=classification.confidence,
            persistence_horizon=persistence,
            event_strength=strength,
            uncertainty_score=uncertainty,
            regime_relevance=regime_relevance,
            supply_demand_axis=classification.axis,
            volatility_implication=volatility,
            summary=summary,
            entities_keywords=[*classification.keywords, *graph_entities.encoded()],
            source_id=payload.source_id,
            timestamp=payload.timestamp,
            raw_text=f"{payload.headline} {payload.body}".strip(),
        )

    def _infer_direction(self, event_type: EventType, text: str) -> EventDirection:
        if any(keyword in text for keyword in ("uncertain", "mixed", "conflicting")):
            return EventDirection.mixed
        if event_type in {EventType.supply_disruption, EventType.inventory_drawdown, EventType.demand_strength, EventType.policy_supportive}:
            return EventDirection.bullish
        if event_type in {EventType.supply_recovery, EventType.inventory_buildup, EventType.demand_weakness, EventType.policy_negative}:
            return EventDirection.bearish
        if event_type in {EventType.currency_macro_shift, EventType.rates_macro_shift, EventType.inflation_macro_shift, EventType.sanctions_geopolitics}:
            if any(keyword in text for keyword in ("hawkish", "rate hike", "dollar up", "higher inflation")):
                return EventDirection.bearish
            if any(keyword in text for keyword in ("dovish", "rate cut", "dollar down", "cooling inflation")):
                return EventDirection.bullish
            return EventDirection.mixed
        return EventDirection.neutral

    def _infer_persistence(self, text: str) -> PersistenceHorizon:
        if any(keyword in text for keyword in ("structural", "multi-year", "long-term", "persistent")):
            return PersistenceHorizon.long
        if any(keyword in text for keyword in ("quarter", "season", "months")):
            return PersistenceHorizon.medium
        if any(keyword in text for keyword in ("week", "near-term")):
            return PersistenceHorizon.short
        return PersistenceHorizon.very_short

    def _infer_strength(self, text: str) -> float:
        strong = ("severe", "sharp", "major", "significant", "record")
        mild = ("minor", "moderate", "contained", "limited")
        if any(keyword in text for keyword in strong):
            return 0.85
        if any(keyword in text for keyword in mild):
            return 0.45
        return 0.60

    def _infer_uncertainty(self, text: str) -> float:
        if any(keyword in text for keyword in ("rumor", "unconfirmed", "speculation", "possibly", "could")):
            return 0.75
        if any(keyword in text for keyword in ("official", "confirmed", "reported", "verified")):
            return 0.25
        return 0.50

    def _infer_volatility_implication(self, event_type: EventType, text: str) -> VolatilityImplication:
        if event_type in {
            EventType.supply_disruption,
            EventType.sanctions_geopolitics,
            EventType.shipping_logistics_issue,
            EventType.weather_risk,
        }:
            return VolatilityImplication.higher
        if "stabilized" in text or "normalised" in text:
            return VolatilityImplication.lower
        return VolatilityImplication.unchanged

    def _infer_regime_relevance(
        self,
        event_type: EventType,
        persistence: PersistenceHorizon,
        volatility: VolatilityImplication,
    ) -> float:
        base = 0.25
        if event_type in {
            EventType.rates_macro_shift,
            EventType.inflation_macro_shift,
            EventType.currency_macro_shift,
            EventType.producer_guidance_change,
        }:
            base += 0.25
        if persistence in {PersistenceHorizon.medium, PersistenceHorizon.long}:
            base += 0.20
        if volatility == VolatilityImplication.higher:
            base += 0.10
        return min(1.0, base)
