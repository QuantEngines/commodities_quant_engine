from __future__ import annotations

from datetime import datetime
from typing import Dict, List

from ...nlp.schemas import CommodityEvent, EventType


class MacroTextFeatureEngine:
    """Macro-event feature rollup for regime and confidence overlays."""

    def compute(self, events: List[CommodityEvent], as_of_timestamp: datetime) -> Dict[str, float]:
        vector = {
            "macro_headwind_score": 0.0,
            "macro_tailwind_score": 0.0,
            "policy_risk_score": 0.0,
            "regime_shift_probability_proxy": 0.0,
            "persistent_trend_event_score": 0.0,
        }
        for event in events:
            age_days = max(0.0, (as_of_timestamp - event.timestamp).total_seconds() / 86400.0)
            decay = 1.0 / (1.0 + age_days)
            weighted = event.event_strength * event.confidence * decay
            if event.event_type in {EventType.rates_macro_shift, EventType.currency_macro_shift, EventType.inflation_macro_shift}:
                if event.expected_direction.value == "bearish":
                    vector["macro_headwind_score"] += weighted
                elif event.expected_direction.value == "bullish":
                    vector["macro_tailwind_score"] += weighted
            if event.event_type in {EventType.policy_negative, EventType.policy_supportive, EventType.sanctions_geopolitics}:
                vector["policy_risk_score"] += weighted
            vector["regime_shift_probability_proxy"] += event.regime_relevance * weighted
            if event.persistence_horizon.value in {"medium", "long"}:
                vector["persistent_trend_event_score"] += weighted
        for key, value in vector.items():
            vector[key] = float(max(0.0, min(1.0, value)))
        return vector
