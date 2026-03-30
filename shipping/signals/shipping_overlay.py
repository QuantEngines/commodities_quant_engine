from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from ..config import load_shipping_signal_rules
from ..models import ShippingFeatureVector, ShippingSignalContext
from .shipping_confidence import ShippingConfidenceOverlay
from .shipping_risk_overlay import ShippingRiskOverlay


class ShippingOverlay:
    def __init__(self):
        self.rules = load_shipping_signal_rules()
        self.confidence_overlay = ShippingConfidenceOverlay()
        self.risk_overlay = ShippingRiskOverlay()

    def build_context(
        self,
        commodity: str,
        timestamp: datetime,
        shipping_feature_vectors: Optional[List[ShippingFeatureVector]],
        base_direction: str,
        base_regime: str,
    ) -> ShippingSignalContext:
        vector = self._latest_vector(commodity, timestamp, shipping_feature_vectors or [])
        if vector is None:
            return ShippingSignalContext.empty(commodity, timestamp)
        feature_map = {name: float(value) for name, value in vector.features.items()}
        quality_score = float(vector.quality_score)
        group_config = self._group_config_for_commodity(commodity)
        confidence = self.confidence_overlay.compute(
            feature_map=feature_map,
            base_direction=base_direction,
            tightening_bias=float(group_config.get("supply_tightening_bias", 0.5)),
            release_bias=float(group_config.get("flow_release_bias", 0.5)),
            quality_score=quality_score,
        )
        risk = self.risk_overlay.compute(feature_map=feature_map, quality_penalty=float(confidence["quality_penalty"]))
        weights = dict(self.rules.get("overlay_weights", {}) or {})
        shipping_features = dict(feature_map)
        shipping_features["shipping_data_quality_score"] = quality_score
        shipping_features["shipping_data_quality_penalty"] = float(confidence["quality_penalty"])
        explanation = self._build_explanation(base_regime, base_direction, confidence, risk, vector)
        return ShippingSignalContext(
            commodity=commodity,
            timestamp=timestamp,
            shipping_alignment_score=float(confidence["alignment"]),
            shipping_conflict_score=float(confidence["conflict"]),
            shipping_risk_penalty=float(risk["risk_penalty"]) * float(weights.get("risk", 0.18)),
            shipping_data_quality_score=quality_score,
            shipping_data_quality_penalty=float(confidence["quality_penalty"]),
            shipping_support_boost=float(confidence["support_boost"]) * float(weights.get("confidence", 0.10)),
            shipping_directional_bias=(float(confidence["alignment"]) - float(confidence["conflict"])) * float(weights.get("directional", 0.12)),
            shipping_regime_bias=(float(confidence["alignment"]) - float(confidence["conflict"])) * float(weights.get("regime", 0.15)),
            shipping_summary=self._summary_from_context(confidence, risk, quality_score),
            shipping_explanation_summary=explanation,
            key_shipping_drivers=list(confidence["drivers"]),
            route_chokepoint_notes=list(risk["notes"]),
            port_congestion_notes=self._port_notes(feature_map),
            shipping_features=shipping_features,
            provenance={"observation_window": vector.observation_window.to_dict(), "source": vector.source},
        )

    def _latest_vector(
        self,
        commodity: str,
        timestamp: datetime,
        feature_vectors: List[ShippingFeatureVector],
    ) -> Optional[ShippingFeatureVector]:
        eligible = [
            vector
            for vector in feature_vectors
            if vector.timestamp <= timestamp and (vector.commodity in {None, commodity, "ALL"} or commodity in vector.commodity_tags)
        ]
        if not eligible:
            return None
        return max(eligible, key=lambda vector: vector.timestamp)

    def _group_config_for_commodity(self, commodity: str) -> Dict[str, object]:
        groups = dict(self.rules.get("commodity_groups", {}) or {})
        for group in groups.values():
            if commodity in list(group.get("commodities", []) or []):
                return dict(group)
        return dict(groups.get("default", {}))

    def _summary_from_context(self, confidence: Dict[str, object], risk: Dict[str, object], quality_score: float) -> str:
        alignment = float(confidence["alignment"])
        conflict = float(confidence["conflict"])
        if quality_score < 0.35:
            return "Sparse coverage"
        if alignment > conflict + 0.10:
            return "Supportive"
        if conflict > alignment + 0.10:
            return "Conflicting"
        if float(risk["risk_penalty"]) > 0.20:
            return "Elevated risk"
        return "Mixed"

    def _port_notes(self, feature_map: Dict[str, float]) -> List[str]:
        notes: List[str] = []
        if feature_map.get("port_congestion_score", 0.0) > 0.25:
            notes.append("Port congestion is above its rolling baseline")
        if feature_map.get("anchorage_buildup_score", 0.0) > 0.25:
            notes.append("Anchorage buildup suggests queueing pressure")
        if feature_map.get("dwell_time_shock", 0.0) > 0.25:
            notes.append("Dwell times have risen above baseline")
        return notes[:4]

    def _build_explanation(
        self,
        base_regime: str,
        base_direction: str,
        confidence: Dict[str, object],
        risk: Dict[str, object],
        vector: ShippingFeatureVector,
    ) -> str:
        return (
            f"Shipping overlay for {base_direction or 'neutral'} {base_regime} uses the latest feature window ending "
            f"{vector.timestamp.isoformat()}. Alignment={float(confidence['alignment']):.2f}, "
            f"conflict={float(confidence['conflict']):.2f}, risk_penalty={float(risk['risk_penalty']):.2f}, "
            f"data_quality={float(vector.quality_score):.2f}."
        )
