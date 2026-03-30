from __future__ import annotations

from typing import Dict, List


class ShippingConfidenceOverlay:
    def compute(
        self,
        feature_map: Dict[str, float],
        base_direction: str,
        tightening_bias: float,
        release_bias: float,
        quality_score: float,
    ) -> Dict[str, object]:
        disruption = float(feature_map.get("port_congestion_score", 0.0) + feature_map.get("route_disruption_score", 0.0) + feature_map.get("chokepoint_stress_score", 0.0) + feature_map.get("anchorage_buildup_score", 0.0))
        flow_momentum = float(feature_map.get("tanker_flow_momentum", 0.0))
        tightening_score = max(0.0, disruption * 0.35 + max(0.0, -flow_momentum) * tightening_bias)
        release_score = max(0.0, max(0.0, flow_momentum) * release_bias + max(0.0, feature_map.get("shipping_momentum_score", 0.0)) * 0.2)

        alignment = 0.0
        conflict = 0.0
        drivers: List[str] = []
        if base_direction == "bullish":
            alignment = tightening_score
            conflict = release_score
        elif base_direction == "bearish":
            alignment = release_score
            conflict = tightening_score
        else:
            alignment = 0.5 * (tightening_score + release_score)
            conflict = 0.0

        if feature_map.get("port_congestion_score", 0.0) > 0.25:
            drivers.append("Port congestion is elevated versus its recent baseline")
        if feature_map.get("chokepoint_stress_score", 0.0) > 0.25:
            drivers.append("Chokepoint stress is contributing to logistics pressure")
        if flow_momentum < -0.1:
            drivers.append("Tanker flow momentum has softened, consistent with tighter logistics")
        if flow_momentum > 0.1:
            drivers.append("Tanker flow momentum has improved, consistent with easing logistics")

        support_boost = max(0.0, alignment - conflict) * 0.10 * max(quality_score, 0.25)
        quality_penalty = max(0.0, 1.0 - quality_score)
        return {
            "alignment": min(1.0, alignment),
            "conflict": min(1.0, conflict),
            "support_boost": support_boost,
            "quality_penalty": quality_penalty,
            "drivers": drivers[:4],
        }
