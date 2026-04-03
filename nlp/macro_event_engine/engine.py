from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Set, Tuple, Union

from ...config.settings import settings
from ...llm.extraction import LLMExtractionAdapter, LLMInferenceClient
from ..extraction import RuleBasedEventExtractor
from ..ingestion import TextRecord, normalize_text_records
from ..preprocessing import standardize_text
from ..schemas import CommodityEvent, EventPayload, EventType, PersistenceHorizon


@dataclass
class EventIntelligenceResult:
    events: List[CommodityEvent] = field(default_factory=list)
    feature_vector: Dict[str, float] = field(default_factory=dict)
    explanations: List[str] = field(default_factory=list)
    diagnostics: Dict[str, float] = field(default_factory=dict)
    cluster_manifest: List[Dict[str, Any]] = field(default_factory=list)


class EventIntelligenceEngine:
    """Transforms unstructured commodity-relevant text into structured events and normalized features."""

    def __init__(self):
        self.rule_extractor = RuleBasedEventExtractor()
        self.llm_adapter = LLMExtractionAdapter()
        self.llm_client = LLMInferenceClient()

    def process_texts(
        self,
        raw_items: Iterable[Union[str, Mapping[str, object], TextRecord]],
        commodity_scope: Sequence[str],
        as_of_timestamp: datetime,
        llm_json_by_source_id: Mapping[str, str] | None = None,
    ) -> EventIntelligenceResult:
        records = normalize_text_records(raw_items)
        events: List[CommodityEvent] = []
        llm_json_by_source_id = llm_json_by_source_id or {}

        for record in records:
            payload = EventPayload(
                source_id=record.source_id,
                timestamp=record.timestamp,
                headline=standardize_text(record.headline),
                body=standardize_text(record.body),
                source=record.source,
            )
            raw_text = f"{payload.headline} {payload.body}".strip()
            llm_json = llm_json_by_source_id.get(record.source_id)
            if llm_json is None and settings.nlp_event.use_llm_extraction:
                llm_json = self.llm_client.generate_event_json(
                    raw_text=raw_text,
                    commodity_scope=commodity_scope,
                    source_id=record.source_id,
                )

            event = self.llm_adapter.parse_or_none(
                llm_json=llm_json,
                commodity_scope=commodity_scope,
                source_id=record.source_id,
                raw_text=raw_text,
            )
            if event is None:
                event = self.rule_extractor.extract(payload, commodity_scope=commodity_scope)
            events.append(event)

        clustered_events, cluster_diagnostics, cluster_manifest = self._deduplicate_and_cluster(events)
        feature_vector = self._aggregate_features(clustered_events, as_of_timestamp)
        explanations = self._build_explanations(clustered_events, feature_vector)
        diagnostics = {
            "event_count": float(len(clustered_events)),
            "raw_event_count": float(len(events)),
            "unknown_event_ratio": float(sum(event.event_type == EventType.unknown for event in clustered_events) / len(clustered_events)) if clustered_events else 0.0,
            "avg_confidence": float(sum(event.confidence for event in clustered_events) / len(clustered_events)) if clustered_events else 0.0,
            **cluster_diagnostics,
        }
        return EventIntelligenceResult(
            events=clustered_events,
            feature_vector=feature_vector,
            explanations=explanations,
            diagnostics=diagnostics,
            cluster_manifest=cluster_manifest,
        )

    def _deduplicate_and_cluster(
        self, events: List[CommodityEvent]
    ) -> Tuple[List[CommodityEvent], Dict[str, float], List[Dict[str, Any]]]:
        if not events:
            return [], {"cluster_count": 0.0, "dedup_ratio": 0.0}, []

        clusters: List[Dict[str, object]] = []
        for event in sorted(events, key=lambda item: item.timestamp):
            tokens = self._token_set(event.raw_text)
            matched_cluster = None
            matched_jaccard = 0.0
            for cluster in clusters:
                representative = cluster["representative"]
                if not isinstance(representative, CommodityEvent):
                    continue
                if representative.event_type != event.event_type:
                    continue
                age_hours = abs((event.timestamp - representative.timestamp).total_seconds()) / 3600.0
                if age_hours > 72.0:
                    continue
                similarity = self._jaccard(tokens, cluster["tokens"])
                if similarity >= 0.72:
                    matched_cluster = cluster
                    matched_jaccard = similarity
                    break

            if matched_cluster is None:
                clusters.append(
                    {
                        "representative": event,
                        "tokens": tokens,
                        "events": [event],
                        "member_jaccards": [1.0],
                    }
                )
                continue

            matched_cluster["events"].append(event)
            matched_cluster["tokens"] = set(matched_cluster["tokens"]).union(tokens)
            matched_cluster["member_jaccards"].append(matched_jaccard)  # type: ignore[union-attr]
            representative = matched_cluster["representative"]
            if isinstance(representative, CommodityEvent) and event.confidence > representative.confidence:
                matched_cluster["representative"] = event

        reduced_events: List[CommodityEvent] = []
        cluster_manifest: List[Dict[str, Any]] = []
        for cluster_idx, cluster in enumerate(clusters):
            representative = cluster["representative"]
            if not isinstance(representative, CommodityEvent):
                continue
            cluster_size = max(1, len(cluster["events"]))
            dedup_scale = 1.0 / (1.0 + 0.35 * (cluster_size - 1))

            member_jaccards: List[float] = list(cluster.get("member_jaccards", [1.0]))  # type: ignore[arg-type]
            non_seed_jaccards = [j for j in member_jaccards if j < 1.0]
            max_intra_jaccard = float(max(non_seed_jaccards)) if non_seed_jaccards else 1.0

            if cluster_size > 1:
                rationale = (
                    f"Highest confidence ({representative.confidence:.3f}) among"
                    f" {cluster_size} member(s); similarity threshold \u2265 0.72"
                )
            else:
                rationale = "Single unique event — no deduplication applied"

            members: List[Dict[str, Any]] = []
            for m_idx, member_event in enumerate(cluster["events"]):
                if not isinstance(member_event, CommodityEvent):
                    continue
                is_rep = (
                    member_event.source_id == representative.source_id
                    and member_event.summary == representative.summary
                )
                members.append(
                    {
                        "source_id": member_event.source_id,
                        "summary": member_event.summary,
                        "confidence": round(member_event.confidence, 4),
                        "event_strength": round(member_event.event_strength, 4),
                        "join_jaccard": round(member_jaccards[m_idx] if m_idx < len(member_jaccards) else 1.0, 4),
                        "is_representative": is_rep,
                    }
                )

            cluster_manifest.append(
                {
                    "cluster_id": f"CLU-{cluster_idx + 1:03d}",
                    "event_type": representative.event_type.value,
                    "cluster_size": cluster_size,
                    "dedup_scale": round(dedup_scale, 4),
                    "representative_source_id": representative.source_id,
                    "representative_rationale": rationale,
                    "representative_summary": representative.summary,
                    "representative_confidence_raw": round(representative.confidence, 4),
                    "representative_event_strength_raw": round(representative.event_strength, 4),
                    "max_intra_jaccard": round(max_intra_jaccard, 4),
                    "members": members,
                }
            )

            reduced_events.append(
                representative.model_copy(
                    update={
                        "event_strength": float(max(0.0, min(1.0, representative.event_strength * dedup_scale))),
                        "confidence": float(max(0.0, min(1.0, representative.confidence * dedup_scale))),
                        "summary": f"{representative.summary} [clustered x{cluster_size}]",
                    }
                )
            )

        dedup_ratio = 1.0 - (len(reduced_events) / len(events)) if events else 0.0
        return reduced_events, {
            "cluster_count": float(len(reduced_events)),
            "dedup_ratio": float(max(0.0, dedup_ratio)),
        }, cluster_manifest

    def _token_set(self, text: str) -> Set[str]:
        raw_tokens = [token.strip(" ,.;:!?()[]{}\"'`") for token in text.lower().split()]
        return {token for token in raw_tokens if token and len(token) > 2}

    def _jaccard(self, left: Set[str], right: Set[str]) -> float:
        if not left or not right:
            return 0.0
        intersection = len(left.intersection(right))
        union = len(left.union(right))
        return float(intersection / union) if union else 0.0

    def _aggregate_features(self, events: List[CommodityEvent], as_of_timestamp: datetime) -> Dict[str, float]:
        features = {
            "supply_shock_score": 0.0,
            "demand_strength_score": 0.0,
            "demand_weakness_score": 0.0,
            "macro_headwind_score": 0.0,
            "macro_tailwind_score": 0.0,
            "policy_risk_score": 0.0,
            "weather_risk_score": 0.0,
            "inventory_signal_score": 0.0,
            "geopolitics_risk_score": 0.0,
            "uncertainty_penalty": 0.0,
            "persistent_trend_event_score": 0.0,
            "regime_shift_probability_proxy": 0.0,
            "event_volatility_risk_score": 0.0,
            "entity_country_concentration": 0.0,
            "shipping_lane_risk_score": 0.0,
            "producer_concentration_risk": 0.0,
        }
        if not events:
            return features

        for event in events:
            age_days = max(0.0, (as_of_timestamp - event.timestamp).total_seconds() / 86400.0)
            decay = 1.0 / (1.0 + age_days)
            weighted = event.event_strength * event.confidence * decay
            if event.event_type == EventType.supply_disruption:
                features["supply_shock_score"] += weighted
            if event.event_type == EventType.supply_recovery:
                features["supply_shock_score"] -= weighted
            if event.event_type == EventType.demand_strength:
                features["demand_strength_score"] += weighted
            if event.event_type == EventType.demand_weakness:
                features["demand_weakness_score"] += weighted
            if event.event_type in {EventType.rates_macro_shift, EventType.currency_macro_shift, EventType.inflation_macro_shift}:
                if event.expected_direction.value == "bearish":
                    features["macro_headwind_score"] += weighted
                elif event.expected_direction.value == "bullish":
                    features["macro_tailwind_score"] += weighted
            if event.event_type in {EventType.policy_negative, EventType.policy_supportive}:
                features["policy_risk_score"] += weighted
            if event.event_type == EventType.weather_risk:
                features["weather_risk_score"] += weighted
            if event.event_type in {EventType.inventory_buildup, EventType.inventory_drawdown}:
                sign = -1.0 if event.event_type == EventType.inventory_buildup else 1.0
                features["inventory_signal_score"] += sign * weighted
            if event.event_type in {EventType.sanctions_geopolitics, EventType.shipping_logistics_issue}:
                features["geopolitics_risk_score"] += weighted
            if event.persistence_horizon in {PersistenceHorizon.medium, PersistenceHorizon.long}:
                features["persistent_trend_event_score"] += weighted
            features["uncertainty_penalty"] += event.uncertainty_score * decay
            if event.volatility_implication.value == "higher":
                features["event_volatility_risk_score"] += weighted
            features["regime_shift_probability_proxy"] += event.regime_relevance * weighted
            graph_features = self._entity_graph_feature_increment(event.entities_keywords, weighted)
            features["entity_country_concentration"] += graph_features["entity_country_concentration"]
            features["shipping_lane_risk_score"] += graph_features["shipping_lane_risk_score"]
            features["producer_concentration_risk"] += graph_features["producer_concentration_risk"]

        for key, value in features.items():
            if key == "inventory_signal_score":
                features[key] = float(max(-1.0, min(1.0, value)))
            else:
                features[key] = float(max(0.0, min(1.0, value)))
        return features

    def _entity_graph_feature_increment(self, entities_keywords: List[str], weighted: float) -> Dict[str, float]:
        countries = [token for token in entities_keywords if token.startswith("country:")]
        lanes = [token for token in entities_keywords if token.startswith("lane:")]
        producers = [token for token in entities_keywords if token.startswith("producer:")]
        return {
            "entity_country_concentration": min(1.0, len(set(countries)) * 0.2) * weighted,
            "shipping_lane_risk_score": min(1.0, len(set(lanes)) * 0.35) * weighted,
            "producer_concentration_risk": min(1.0, len(set(producers)) * 0.25) * weighted,
        }

    def _build_explanations(self, events: List[CommodityEvent], features: Dict[str, float]) -> List[str]:
        lines: List[str] = []
        if features["supply_shock_score"] > 0.35:
            lines.append("Bullish supply shock detected in commodity complex")
        if features["macro_headwind_score"] > 0.25:
            lines.append("Macro headwind increased for cyclical commodities")
        if features["weather_risk_score"] > 0.25:
            lines.append("Weather-related uncertainty raised for agri contracts")
        if features["inventory_signal_score"] < -0.2:
            lines.append("Inventory-related bearish cluster reduced conviction")
        if features["geopolitics_risk_score"] > 0.2:
            lines.append("Geopolitical event increased volatility and risk penalty")
        if not lines and events:
            lines.append("Structured event flow is neutral to mildly mixed")
        return lines
