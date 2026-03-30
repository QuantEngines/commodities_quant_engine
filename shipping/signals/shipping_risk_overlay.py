from __future__ import annotations

from typing import Dict, List


class ShippingRiskOverlay:
    def compute(self, feature_map: Dict[str, float], quality_penalty: float) -> Dict[str, object]:
        corridor_stress = float(
            0.45 * feature_map.get("route_disruption_score", 0.0)
            + 0.35 * feature_map.get("chokepoint_stress_score", 0.0)
            + 0.20 * feature_map.get("anchorage_buildup_score", 0.0)
        )
        risk_penalty = min(1.0, corridor_stress * 0.35 + quality_penalty * 0.30)
        notes: List[str] = []
        if feature_map.get("route_disruption_score", 0.0) > 0.25:
            notes.append("Route disruption conditions are elevated")
        if feature_map.get("chokepoint_stress_score", 0.0) > 0.25:
            notes.append("Chokepoint stress is elevated near monitored corridors")
        if feature_map.get("port_congestion_score", 0.0) > 0.25:
            notes.append("Port congestion is elevated at monitored clusters")
        return {
            "risk_penalty": risk_penalty,
            "corridor_stress": corridor_stress,
            "notes": notes[:4],
        }
