from __future__ import annotations

from typing import Dict, List


class ShippingRiskOverlay:
    def compute(self, feature_map: Dict[str, float], quality_penalty: float) -> Dict[str, object]:
        corridor_stress = float(
            0.45 * feature_map.get("route_disruption_score", 0.0)
            + 0.35 * feature_map.get("chokepoint_stress_score", 0.0)
            + 0.20 * feature_map.get("anchorage_buildup_score", 0.0)
        )
        benchmark_divergence = abs(float(feature_map.get("shipping_market_divergence", feature_map.get("bdi_shipping_divergence", 0.0))))
        risk_penalty = min(1.0, corridor_stress * 0.35 + quality_penalty * 0.30 + benchmark_divergence * 0.10)
        notes: List[str] = []
        if feature_map.get("route_disruption_score", 0.0) > 0.25:
            notes.append("Route disruption conditions are elevated")
        if feature_map.get("chokepoint_stress_score", 0.0) > 0.25:
            notes.append("Chokepoint stress is elevated near monitored corridors")
        if feature_map.get("port_congestion_score", 0.0) > 0.25:
            notes.append("Port congestion is elevated at monitored clusters")
        if feature_map.get("shipping_market_benchmark_active", feature_map.get("bdi_benchmark_active", 0.0)) > 0.5 and benchmark_divergence > 0.35:
            if feature_map.get("bdti_benchmark_active", 0.0) > 0.5 or feature_map.get("bcti_benchmark_active", 0.0) > 0.5:
                notes.append("Tanker-market benchmarks are diverging from observed shipping stress")
            elif feature_map.get("lng_rate_benchmark_active", 0.0) > 0.5 or feature_map.get("lng_carrier_value_benchmark_active", 0.0) > 0.5:
                notes.append("LNG carrier benchmarks are diverging from observed shipping stress")
            else:
                notes.append("Dry-bulk benchmarks are diverging from observed shipping stress")
        return {
            "risk_penalty": risk_penalty,
            "corridor_stress": corridor_stress,
            "notes": notes[:4],
        }
