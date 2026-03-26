from __future__ import annotations

from datetime import datetime
from typing import Dict, List

from ...nlp.schemas import CommodityEvent


class CommodityEventFeatureEngine:
    """Commodity-focused event feature rollup for directional overlays."""

    def compute(self, events: List[CommodityEvent], as_of_timestamp: datetime) -> Dict[str, float]:
        if not events:
            return {
                "supply_shock_score": 0.0,
                "inventory_signal_score": 0.0,
                "uncertainty_penalty": 0.0,
            }
        output = {
            "supply_shock_score": 0.0,
            "inventory_signal_score": 0.0,
            "uncertainty_penalty": 0.0,
        }
        for event in events:
            age_days = max(0.0, (as_of_timestamp - event.timestamp).total_seconds() / 86400.0)
            decay = 1.0 / (1.0 + age_days)
            output["supply_shock_score"] += event.to_feature_dict()["signed_event_effect"] * decay
            output["inventory_signal_score"] += event.to_feature_dict()["signed_event_effect"] * 0.25 * decay
            output["uncertainty_penalty"] += event.uncertainty_score * decay
        output["supply_shock_score"] = float(max(-1.0, min(1.0, output["supply_shock_score"])))
        output["inventory_signal_score"] = float(max(-1.0, min(1.0, output["inventory_signal_score"])))
        output["uncertainty_penalty"] = float(max(0.0, min(1.0, output["uncertainty_penalty"])))
        return output
