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
        benchmark_support = float(feature_map.get("shipping_market_benchmark_support", feature_map.get("bdi_benchmark_support", 0.0)))
        benchmark_divergence = float(feature_map.get("shipping_market_divergence", feature_map.get("bdi_shipping_divergence", 0.0)))
        tightening_score = max(0.0, disruption * 0.35 + max(0.0, -flow_momentum) * tightening_bias + benchmark_support * 0.25)
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
        if feature_map.get("shipping_market_benchmark_active", feature_map.get("bdi_benchmark_active", 0.0)) > 0.5 and benchmark_support > 0.10:
            drivers.append(self._support_driver(feature_map))
        if feature_map.get("shipping_market_benchmark_active", feature_map.get("bdi_benchmark_active", 0.0)) > 0.5 and benchmark_divergence < -0.20:
            drivers.append(self._divergence_driver(feature_map))

        support_boost = max(0.0, alignment - conflict) * 0.10 * max(quality_score, 0.25)
        quality_penalty = max(0.0, 1.0 - quality_score)
        return {
            "alignment": min(1.0, alignment),
            "conflict": min(1.0, conflict),
            "support_boost": support_boost,
            "quality_penalty": quality_penalty,
            "drivers": drivers[:4],
        }

    def _support_driver(self, feature_map: Dict[str, float]) -> str:
        if feature_map.get("bdti_benchmark_active", 0.0) > 0.5 or feature_map.get("bcti_benchmark_active", 0.0) > 0.5:
            return "Tanker benchmarks are confirming tighter crude and product logistics"
        if feature_map.get("lng_rate_benchmark_active", 0.0) > 0.5 or feature_map.get("lng_carrier_value_benchmark_active", 0.0) > 0.5:
            return "LNG carrier benchmarks are confirming tighter gas-shipping conditions"
        return "Dry-bulk benchmarks are confirming tighter seaborne freight conditions"

    def _divergence_driver(self, feature_map: Dict[str, float]) -> str:
        if feature_map.get("bdti_benchmark_active", 0.0) > 0.5 or feature_map.get("bcti_benchmark_active", 0.0) > 0.5:
            return "Observed shipping stress is lagging tanker-market benchmarks"
        if feature_map.get("lng_rate_benchmark_active", 0.0) > 0.5 or feature_map.get("lng_carrier_value_benchmark_active", 0.0) > 0.5:
            return "Observed shipping stress is lagging LNG carrier benchmarks"
        return "Observed shipping stress is lagging dry-bulk benchmarks"
