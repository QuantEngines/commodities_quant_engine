from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

from ..schemas.events import EventType, SupplyDemandAxis


@dataclass(frozen=True)
class ClassificationResult:
    event_type: EventType
    axis: SupplyDemandAxis
    confidence: float
    keywords: List[str]


class CommodityEventClassifier:
    """Rule-first commodity-domain classifier with deterministic fallback behavior."""

    _RULES: List[Tuple[EventType, SupplyDemandAxis, Tuple[str, ...]]] = [
        (EventType.supply_disruption, SupplyDemandAxis.supply, ("outage", "disruption", "strike", "shutdown", "mine closure", "pipeline leak")),
        (EventType.supply_recovery, SupplyDemandAxis.supply, ("restart", "resumed", "recovery", "normalised", "production restored")),
        (EventType.demand_strength, SupplyDemandAxis.demand, ("strong demand", "higher demand", "demand surged", "industrial rebound", "buying spree")),
        (EventType.demand_weakness, SupplyDemandAxis.demand, ("weak demand", "demand slowdown", "recession", "orders fell", "consumption dropped")),
        (EventType.inventory_drawdown, SupplyDemandAxis.supply, ("inventory draw", "stock draw", "inventories fell", "crude drawdown")),
        (EventType.inventory_buildup, SupplyDemandAxis.supply, ("inventory build", "stocks rose", "inventories increased", "surplus stock")),
        (EventType.weather_risk, SupplyDemandAxis.supply, ("drought", "flood", "heatwave", "cyclone", "el nino", "monsoon deficit")),
        (EventType.policy_supportive, SupplyDemandAxis.macro, ("stimulus", "subsidy", "support package", "tax relief", "export incentive")),
        (EventType.policy_negative, SupplyDemandAxis.macro, ("ban", "export restriction", "tariff", "tax hike", "quota cut")),
        (EventType.sanctions_geopolitics, SupplyDemandAxis.macro, ("sanction", "conflict", "geopolitical", "ceasefire", "military")),
        (EventType.shipping_logistics_issue, SupplyDemandAxis.supply, ("shipping delay", "port congestion", "freight", "vessel", "logistics disruption")),
        (EventType.producer_guidance_change, SupplyDemandAxis.supply, ("opec", "producer guidance", "output target", "production cut", "production hike")),
        (EventType.currency_macro_shift, SupplyDemandAxis.macro, ("currency", "dollar", "fx", "exchange rate", "devaluation")),
        (EventType.rates_macro_shift, SupplyDemandAxis.macro, ("rate hike", "rate cut", "policy rate", "fed", "central bank")),
        (EventType.inflation_macro_shift, SupplyDemandAxis.macro, ("inflation", "cpi", "ppi", "wpi")),
        (EventType.industrial_activity_signal, SupplyDemandAxis.demand, ("pmi", "manufacturing", "industrial output", "factory activity")),
    ]

    def classify(self, text: str) -> ClassificationResult:
        lowered = text.lower()
        best = ClassificationResult(event_type=EventType.unknown, axis=SupplyDemandAxis.mixed, confidence=0.2, keywords=[])
        for event_type, axis, keywords in self._RULES:
            matched = [keyword for keyword in keywords if keyword in lowered]
            if not matched:
                continue
            confidence = min(0.95, 0.35 + 0.2 * len(matched))
            if confidence > best.confidence:
                best = ClassificationResult(event_type=event_type, axis=axis, confidence=confidence, keywords=matched)
        return best

    def classify_many(self, texts: Iterable[str]) -> List[ClassificationResult]:
        return [self.classify(text) for text in texts]
