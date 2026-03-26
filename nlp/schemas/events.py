from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Dict, List

from pydantic import BaseModel, Field, field_validator


class EventType(str, Enum):
    supply_disruption = "supply_disruption"
    supply_recovery = "supply_recovery"
    demand_strength = "demand_strength"
    demand_weakness = "demand_weakness"
    inventory_buildup = "inventory_buildup"
    inventory_drawdown = "inventory_drawdown"
    weather_risk = "weather_risk"
    policy_supportive = "policy_supportive"
    policy_negative = "policy_negative"
    sanctions_geopolitics = "sanctions/geopolitics"
    shipping_logistics_issue = "shipping/logistics_issue"
    producer_guidance_change = "producer_guidance_change"
    currency_macro_shift = "currency_macro_shift"
    rates_macro_shift = "rates_macro_shift"
    inflation_macro_shift = "inflation_macro_shift"
    industrial_activity_signal = "industrial_activity_signal"
    unknown = "unknown"


class AssetScope(str, Enum):
    single_commodity = "single_commodity"
    sector_basket = "sector_basket"
    macro_wide = "macro_wide"


class EventDirection(str, Enum):
    bullish = "bullish"
    bearish = "bearish"
    neutral = "neutral"
    mixed = "mixed"


class PersistenceHorizon(str, Enum):
    very_short = "very_short"
    short = "short"
    medium = "medium"
    long = "long"


class SupplyDemandAxis(str, Enum):
    supply = "supply"
    demand = "demand"
    macro = "macro"
    mixed = "mixed"


class VolatilityImplication(str, Enum):
    lower = "lower"
    unchanged = "unchanged"
    higher = "higher"


class EventPayload(BaseModel):
    source_id: str
    timestamp: datetime
    headline: str
    body: str = ""
    source: str = "unknown"


class CommodityEvent(BaseModel):
    event_type: EventType
    commodity_scope: List[str] = Field(default_factory=list)
    asset_scope: AssetScope = AssetScope.single_commodity
    expected_direction: EventDirection = EventDirection.neutral
    confidence: float = 0.5
    persistence_horizon: PersistenceHorizon = PersistenceHorizon.short
    event_strength: float = 0.0
    uncertainty_score: float = 0.5
    regime_relevance: float = 0.0
    supply_demand_axis: SupplyDemandAxis = SupplyDemandAxis.mixed
    volatility_implication: VolatilityImplication = VolatilityImplication.unchanged
    summary: str = ""
    entities_keywords: List[str] = Field(default_factory=list)
    source_id: str = ""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    raw_text: str = ""

    @field_validator("confidence", "event_strength", "uncertainty_score", "regime_relevance")
    @classmethod
    def _clip_scores(cls, value: float) -> float:
        return float(max(0.0, min(1.0, value)))

    def to_feature_dict(self) -> Dict[str, float]:
        sign = 0.0
        if self.expected_direction == EventDirection.bullish:
            sign = 1.0
        elif self.expected_direction == EventDirection.bearish:
            sign = -1.0
        persistence_weight = {
            PersistenceHorizon.very_short: 0.5,
            PersistenceHorizon.short: 0.8,
            PersistenceHorizon.medium: 1.0,
            PersistenceHorizon.long: 1.2,
        }[self.persistence_horizon]
        base_effect = sign * self.event_strength * self.confidence * persistence_weight
        return {
            "signed_event_effect": base_effect,
            "uncertainty_penalty": self.uncertainty_score,
            "regime_relevance": self.regime_relevance,
            "volatility_risk": 1.0 if self.volatility_implication == VolatilityImplication.higher else 0.0,
        }
