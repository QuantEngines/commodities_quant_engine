from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Dict, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ...config.settings import settings
from ...core.composite import event_risk_penalty
from ...core.directional import event_directional_adjustment
from ...core.regimes import event_regime_adjustment
from ...data.contract_master.manager import contract_master
from ...data.models import (
    DirectionalSignal,
    MacroConfidenceOverlay,
    MacroEvent,
    MacroFeature,
    RiskPenalty,
    SignalPackage,
    SignalSnapshot,
    Suggestion,
)
from ...data.quality_checks import MarketDataValidator
from ...regimes.regime_engine import RegimeEngine
from ...shipping.models import ShippingFeatureVector, ShippingSignalContext
from ...shipping.signals import ShippingOverlay
from ...signals.directional.directional_alpha import DirectionalAlphaEngine
from ...signals.inefficiency.inefficiency_engine import InefficiencyEngine
from ...signals.macro.confidence_overlay import MacroConfidenceOverlay as MacroConfidenceOverlayEngine
from ...signals.macro.directional_overlay import MacroDirectionalOverlay
from ...signals.macro.regime_overlay import MacroRegimeOverlay
from ...nlp import EventIntelligenceEngine
from ...nlp.macro_event_engine.calibration import calibrate_overlay_weights_for_commodity


class CompositeDecisionEngine:
    """Composite signal engine with validation, explainability, and audit snapshots."""

    def __init__(self, parameter_state: Optional[Dict[str, object]] = None):
        self.parameter_state = parameter_state or {}
        self.validator = MarketDataValidator()
        self.regime_engine = RegimeEngine()
        self.directional_engine = DirectionalAlphaEngine(parameter_state=self.parameter_state)
        self.inefficiency_engine = InefficiencyEngine()
        self.macro_regime_overlay = MacroRegimeOverlay(self.regime_engine)
        self.macro_directional_overlay = MacroDirectionalOverlay(self.directional_engine)
        self.macro_confidence_overlay = MacroConfidenceOverlayEngine()
        self.shipping_overlay = ShippingOverlay()
        self.event_intelligence_engine = EventIntelligenceEngine()

    def generate_suggestion(
        self,
        data: pd.DataFrame,
        commodity: str,
        macro_features: Optional[List[MacroFeature]] = None,
        macro_events: Optional[List[MacroEvent]] = None,
        shipping_feature_vectors: Optional[List[ShippingFeatureVector]] = None,
        as_of_timestamp: Optional[datetime] = None,
    ) -> Suggestion:
        return self.generate_signal_package(
            data=data,
            commodity=commodity,
            macro_features=macro_features,
            macro_events=macro_events,
            shipping_feature_vectors=shipping_feature_vectors,
            as_of_timestamp=as_of_timestamp,
        ).suggestion

    def generate_signal_package(
        self,
        data: pd.DataFrame,
        commodity: str,
        macro_features: Optional[List[MacroFeature]] = None,
        macro_events: Optional[List[MacroEvent]] = None,
        shipping_feature_vectors: Optional[List[ShippingFeatureVector]] = None,
        raw_text_items: Optional[List[Union[str, Mapping[str, object]]]] = None,
        llm_json_by_source_id: Optional[Dict[str, str]] = None,
        as_of_timestamp: Optional[datetime] = None,
    ) -> SignalPackage:
        macro_features = macro_features or []
        macro_events = macro_events or []
        shipping_feature_vectors = shipping_feature_vectors or []
        raw_text_items = raw_text_items or []
        as_of_timestamp = as_of_timestamp or self._infer_timestamp(data)

        event_payload = self._build_event_payload(
            commodity=commodity,
            as_of_timestamp=as_of_timestamp,
            raw_text_items=raw_text_items,
            llm_json_by_source_id=llm_json_by_source_id or {},
        )
        overlay_weights = event_payload["overlay_weights"]

        quality_report = self.validator.validate(data, as_of=as_of_timestamp)
        if not quality_report.is_valid or quality_report.flag == "incomplete":
            raise ValueError(f"Signal generation blocked by market data quality issues: {quality_report.issues}")

        feature_frame = self.directional_engine.build_feature_frame(data)
        latest_features = feature_frame.iloc[-1].to_dict()
        macro_regime = self.macro_regime_overlay.detect_macro_regime(
            commodity=commodity,
            features=latest_features,
            macro_features=macro_features,
            timestamp=as_of_timestamp,
        )

        directional_signals = self._build_directional_signals(
            commodity=commodity,
            latest_features=latest_features,
            macro_features=macro_features,
            event_features=event_payload["features"],
            event_directional_weight=float(overlay_weights.get("directional_weight", settings.nlp_event.directional_overlay_weight)),
            as_of_timestamp=as_of_timestamp,
        )
        directional_scores = {signal.horizon: signal.score for signal in directional_signals}
        directional_confidences = {signal.horizon: signal.confidence for signal in directional_signals}
        inefficiency = self.inefficiency_engine.detect_inefficiency(data, commodity)
        base_direction = self._determine_direction(directional_scores)
        shipping_context = self.shipping_overlay.build_context(
            commodity=commodity,
            timestamp=as_of_timestamp,
            shipping_feature_vectors=shipping_feature_vectors,
            base_direction=base_direction,
            base_regime=macro_regime.combined_label,
        )
        macro_confidence = self.macro_confidence_overlay.compute_confidence_adjustment(
            commodity=commodity,
            timestamp=as_of_timestamp,
            macro_features=macro_features,
            macro_events=macro_events,
            base_regime=macro_regime.combined_label,
            base_direction=base_direction,
        )
        risk_penalty = self._calculate_risk_penalty(data, directional_signals, macro_confidence, macro_events, shipping_context)
        event_penalty = event_risk_penalty(event_payload["features"]) * float(overlay_weights.get("risk_weight", settings.nlp_event.risk_penalty_weight))
        risk_penalty.total_penalty = float(min(1.5, risk_penalty.total_penalty + event_penalty))
        component_scores = self._compute_component_scores(
            directional_scores,
            directional_confidences,
            inefficiency.deviation_z,
            macro_regime.macro_contribution,
            risk_penalty,
            macro_confidence,
            shipping_context,
        )
        component_scores["regime"] = float(
            component_scores["regime"] + event_regime_adjustment(event_payload["features"]) * float(overlay_weights.get("regime_weight", settings.nlp_event.regime_overlay_weight))
        )
        composite_score = self._aggregate_components(component_scores)
        directional_bias = self._directional_bias_label(directional_scores)
        entry_quality, entry_quality_score = self._assess_entry_quality(data, inefficiency.deviation_z, directional_bias)
        signal_agreement = self._signal_agreement_score(directional_scores, macro_confidence, shipping_context)
        category, direction, entry_style, horizon = self._classify_suggestion(
            composite_score,
            directional_scores,
            directional_bias,
            entry_quality,
            signal_agreement,
            risk_penalty,
            macro_confidence,
            shipping_context,
        )
        supporting, contradictory = self._extract_drivers(
            macro_regime.combined_label,
            inefficiency.deviation_z,
            directional_scores,
            directional_bias,
            entry_quality,
            signal_agreement,
            macro_confidence,
            risk_penalty,
            shipping_context,
        )
        risks = self._identify_risks(inefficiency.instability_warning, risk_penalty, macro_confidence, shipping_context, directional_bias, entry_quality)
        dominant_component, override_reason = self._dominant_component_and_override(
            component_scores=component_scores,
            entry_quality=entry_quality,
            risk_penalty=risk_penalty,
            signal_agreement=signal_agreement,
            recommendation=category,
        )
        explanation = self._generate_explanation(
            macro_regime.combined_label,
            directional_scores,
            directional_bias,
            entry_quality,
            inefficiency.deviation_z,
            risk_penalty,
            signal_agreement,
            macro_confidence,
            shipping_context,
            composite_score,
            recommendation=category,
            override_reason=override_reason,
            event_explanations=event_payload["explanations"],
        )
        macro_highlights = self._extract_macro_highlights(macro_features, as_of_timestamp)
        calibrated_regime_probability = self._calibrated_regime_probability(macro_regime.combined_label, macro_regime.probability)
        regime_probabilities = self._build_regime_probability_map(macro_regime.combined_label, calibrated_regime_probability)
        directional_confidence = self._compute_directional_confidence(directional_confidences, directional_scores)
        data_quality_confidence = self._compute_data_quality_confidence(quality_report.flag)
        raw_tradeability_confidence = self._compute_tradeability_confidence(
            directional_confidence=directional_confidence,
            entry_quality_score=entry_quality_score,
            risk_penalty=risk_penalty,
            signal_agreement=signal_agreement,
            data_quality_confidence=data_quality_confidence,
        )
        tradeability_confidence = self._apply_confidence_calibration(raw_tradeability_confidence)
        final_confidence = tradeability_confidence
        component_contributions = {name: self._label_component_contribution(name, score) for name, score in component_scores.items()}

        active_contract = contract_master.get_active_contract(commodity, as_of_timestamp.date())
        signal_id = self._build_signal_id(commodity, as_of_timestamp)
        suggestion = Suggestion(
            timestamp=as_of_timestamp,
            commodity=commodity,
            exchange=active_contract.exchange if active_contract else settings.commodities[commodity].exchange,
            active_contract=active_contract.symbol if active_contract else commodity,
            regime_label=macro_regime.combined_label,
            regime_probabilities=regime_probabilities,
            directional_scores=directional_scores,
            inefficiency_score=inefficiency.deviation_z,
            risk_penalty=risk_penalty.total_penalty,
            composite_score=composite_score,
            final_category=category,
            directional_bias=directional_bias,
            entry_quality=entry_quality,
            trade_recommendation=category,
            directional_confidence=directional_confidence,
            tradeability_confidence=tradeability_confidence,
            data_quality_confidence=data_quality_confidence,
            component_contributions=component_contributions,
            dominant_component=dominant_component,
            override_reason=override_reason,
            supportive_signals=supporting,
            contradictory_signals=contradictory,
            key_risks=risks,
            preferred_direction=direction,
            suggested_entry_style=entry_style,
            suggested_holding_horizon=horizon,
            key_supporting_drivers=supporting,
            key_contradictory_drivers=contradictory,
            principal_risks=risks,
            explanation_summary=explanation,
            data_quality_flag=quality_report.flag,
            confidence_score=final_confidence,
            signal_id=signal_id,
            model_version=str(self.parameter_state.get("version_id", "default")),
            config_version=settings.config_version,
            diagnostics={
                "quality_issues": quality_report.issues,
                "component_scores": component_scores,
                "feature_vector": {key: float(value) for key, value in latest_features.items()},
                "directional_confidences": directional_confidences,
                "contract_source": active_contract.source if active_contract else "settings_default",
                "contract_is_fallback": active_contract.is_fallback if active_contract else True,
                "raw_regime_probability": float(macro_regime.probability),
                "calibrated_regime_probability": float(calibrated_regime_probability),
                "regime_probabilities": regime_probabilities,
                "directional_bias": directional_bias,
                "entry_quality": entry_quality,
                "signal_agreement": float(signal_agreement),
                "directional_confidence": float(directional_confidence),
                "tradeability_confidence": float(tradeability_confidence),
                "data_quality_confidence": float(data_quality_confidence),
                "component_contributions": component_contributions,
                "dominant_component": dominant_component,
                "override_reason": override_reason,
                "confidence_calibrated": bool(self.parameter_state.get("confidence_calibration")),
                "event_intelligence_features": event_payload["features"],
                "event_intelligence_diagnostics": event_payload["diagnostics"],
                "event_overlay_weights": overlay_weights,
                "event_intelligence_events": event_payload["events"],
                "event_cluster_manifest": event_payload["cluster_manifest"],
                "shipping_context": shipping_context.to_dict(),
                "shipping_features": shipping_context.shipping_features,
            },
            macro_regime_summary=macro_regime.combined_label,
            macro_feature_highlights=macro_highlights,
            macro_alignment_score=macro_confidence.macro_alignment_score,
            macro_conflict_score=macro_confidence.macro_conflict_score,
            macro_event_risk_flag=macro_confidence.event_risk_penalty > 0.25,
            macro_confidence_adjustment=macro_confidence.final_confidence_adjustment,
            macro_explanation_summary=self._generate_macro_explanation(macro_confidence),
            key_macro_drivers=macro_confidence.key_macro_drivers,
            key_macro_risks=macro_confidence.key_macro_risks,
            news_narrative_summary=macro_confidence.news_narrative_summary,
            shipping_summary=shipping_context.shipping_summary,
            shipping_alignment_score=shipping_context.shipping_alignment_score,
            shipping_conflict_score=shipping_context.shipping_conflict_score,
            shipping_risk_flag=shipping_context.shipping_risk_penalty > 0.15,
            shipping_data_quality_score=shipping_context.shipping_data_quality_score,
            shipping_support_boost=shipping_context.shipping_support_boost,
            shipping_risk_penalty=shipping_context.shipping_risk_penalty,
            shipping_data_quality_penalty=shipping_context.shipping_data_quality_penalty,
            shipping_explanation_summary=shipping_context.shipping_explanation_summary,
            key_shipping_drivers=shipping_context.key_shipping_drivers,
            route_chokepoint_notes=shipping_context.route_chokepoint_notes,
            port_congestion_notes=shipping_context.port_congestion_notes,
        )
        snapshot = SignalSnapshot(
            signal_id=signal_id,
            timestamp=as_of_timestamp,
            commodity=commodity,
            contract=suggestion.active_contract,
            exchange=suggestion.exchange,
            signal_category=category,
            direction=direction,
            conviction=suggestion.confidence_score,
            regime_label=macro_regime.combined_label,
            regime_probability=calibrated_regime_probability,
            inefficiency_score=inefficiency.deviation_z,
            composite_score=composite_score,
            suggested_horizon=horizon,
            directional_scores=directional_scores,
            key_drivers=supporting,
            key_risks=risks,
            component_scores=component_scores,
            feature_vector={
                **{name: float(latest_features.get(name, 0.0)) for name in settings.signal.directional_feature_names},
                **{name: float(value) for name, value in macro_highlights.items()},
            },
            model_version=str(self.parameter_state.get("version_id", "default")),
            config_version=settings.config_version,
            data_quality_flag=quality_report.flag,
            macro_alignment_score=macro_confidence.macro_alignment_score,
            macro_conflict_score=macro_confidence.macro_conflict_score,
            shipping_alignment_score=shipping_context.shipping_alignment_score,
            shipping_conflict_score=shipping_context.shipping_conflict_score,
            shipping_data_quality_score=shipping_context.shipping_data_quality_score,
            metadata={
                "contradictory_drivers": contradictory,
                "macro_drivers": macro_confidence.key_macro_drivers,
                "macro_risks": macro_confidence.key_macro_risks,
                "shipping_drivers": shipping_context.key_shipping_drivers,
                "shipping_notes": shipping_context.route_chokepoint_notes + shipping_context.port_congestion_notes,
                "macro_feature_highlights": macro_highlights,
                "directional_confidences": directional_confidences,
                "event_explanations": event_payload["explanations"],
                "event_overlay_weights": overlay_weights,
            },
        )
        return SignalPackage(suggestion=suggestion, snapshot=snapshot, quality_report=quality_report)

    def _build_event_payload(
        self,
        commodity: str,
        as_of_timestamp: datetime,
        raw_text_items: List[Union[str, Mapping[str, object]]],
        llm_json_by_source_id: Dict[str, str],
    ) -> Dict[str, object]:
        if not settings.nlp_event.enabled or not raw_text_items:
            return {
                "features": {
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
                },
                "explanations": [],
                "diagnostics": {"event_count": 0.0},
                "events": [],
                "cluster_manifest": [],
                "overlay_weights": {
                    "directional_weight": float(settings.nlp_event.directional_overlay_weight),
                    "regime_weight": float(settings.nlp_event.regime_overlay_weight),
                    "risk_weight": float(settings.nlp_event.risk_penalty_weight),
                },
            }

        bounded_inputs = raw_text_items[: max(1, int(settings.nlp_event.max_items_per_cycle))]
        result = self.event_intelligence_engine.process_texts(
            raw_items=bounded_inputs,
            commodity_scope=[commodity],
            as_of_timestamp=as_of_timestamp,
            llm_json_by_source_id=llm_json_by_source_id,
        )
        calibrated = calibrate_overlay_weights_for_commodity(commodity)
        return {
            "features": result.feature_vector,
            "explanations": result.explanations,
            "diagnostics": result.diagnostics,
            "events": [event.model_dump(mode="json") for event in result.events],
            "cluster_manifest": result.cluster_manifest,
            "overlay_weights": {
                "directional_weight": float(calibrated.directional_weight),
                "regime_weight": float(calibrated.regime_weight),
                "risk_weight": float(calibrated.risk_weight),
            },
        }

    def _build_directional_signals(
        self,
        commodity: str,
        latest_features: Dict[str, float],
        macro_features: List[MacroFeature],
        event_features: Dict[str, float],
        event_directional_weight: float,
        as_of_timestamp: datetime,
    ) -> List[DirectionalSignal]:
        signals = []
        for horizon in settings.directional_horizons:
            base_signal = self.directional_engine.generate_signal(commodity, latest_features, as_of_timestamp, horizon)
            macro_directional = self.macro_directional_overlay.enhance_directional_signal(
                commodity=commodity,
                features=latest_features,
                macro_features=macro_features,
                timestamp=as_of_timestamp,
                horizon=horizon,
            )
            confidence = min(
                1.0,
                base_signal.confidence * macro_directional.confidence_multiplier,
            )
            if settings.nlp_event.enabled:
                directional_shift = event_directional_adjustment(event_features) * float(event_directional_weight)
                uncertainty_haircut = min(0.4, event_features.get("uncertainty_penalty", 0.0) * float(settings.nlp_event.confidence_uncertainty_weight))
            else:
                directional_shift = 0.0
                uncertainty_haircut = 0.0
            signals.append(
                DirectionalSignal(
                    commodity=commodity,
                    horizon=horizon,
                    score=macro_directional.adjusted_score + directional_shift,
                    confidence=max(0.0, min(1.0, confidence * (1.0 - uncertainty_haircut))),
                    features={name: float(latest_features.get(name, 0.0)) for name in settings.signal.directional_feature_names},
                    timestamp=as_of_timestamp,
                    model_version=str(self.parameter_state.get("version_id", "default")),
                )
            )
        return signals

    def _infer_timestamp(self, data: pd.DataFrame) -> datetime:
        if isinstance(data.index, pd.DatetimeIndex):
            return data.index[-1].to_pydatetime()
        return datetime.now(timezone.utc).replace(tzinfo=None)

    def _determine_direction(self, directional_scores: Dict[int, float]) -> str:
        weighted_score = self._weighted_directional_score(directional_scores)
        if weighted_score > settings.composite.neutral_threshold:
            return "bullish"
        if weighted_score < -settings.composite.neutral_threshold:
            return "bearish"
        return "neutral"

    def _calculate_risk_penalty(
        self,
        data: pd.DataFrame,
        directional_signals: List[DirectionalSignal],
        macro_confidence: MacroConfidenceOverlay,
        macro_events: List[MacroEvent],
        shipping_context: ShippingSignalContext,
    ) -> RiskPenalty:
        vol_penalty = max(0.0, float(data["close"].pct_change().rolling(20, min_periods=5).std(ddof=0).iloc[-1] * 10.0 - 0.15))
        disagreement_penalty = self._directional_disagreement_penalty(directional_signals)
        liquidity_penalty = 0.15 if data["volume"].iloc[-5:].mean() < max(100.0, data["volume"].median() * 0.35) else 0.0
        event_risk = macro_confidence.event_risk_penalty
        if any(event.expected_impact == "high" for event in macro_events):
            event_risk = max(event_risk, 0.1)
        shipping_risk_penalty = float(shipping_context.shipping_risk_penalty)
        shipping_data_quality_penalty = float(shipping_context.shipping_data_quality_penalty)
        total_penalty = min(1.5, vol_penalty + disagreement_penalty + liquidity_penalty + event_risk + shipping_risk_penalty + shipping_data_quality_penalty)
        return RiskPenalty(
            volatility_spike=vol_penalty,
            signal_disagreement=disagreement_penalty,
            event_risk=event_risk,
            liquidity_penalty=liquidity_penalty,
            total_penalty=total_penalty,
            shipping_risk_penalty=shipping_risk_penalty,
            shipping_data_quality_penalty=shipping_data_quality_penalty,
        )

    def _compute_component_scores(
        self,
        directional_scores: Dict[int, float],
        directional_confidences: Dict[int, float],
        inefficiency_score: float,
        macro_regime_contribution: float,
        risk_penalty: RiskPenalty,
        macro_confidence: MacroConfidenceOverlay,
        shipping_context: ShippingSignalContext,
    ) -> Dict[str, float]:
        directional_component = self._weighted_directional_score(directional_scores)
        directional_alignment = self._directional_alignment(directional_scores)
        average_confidence = float(np.mean(list(directional_confidences.values()))) if directional_confidences else 0.0
        regime_component = macro_regime_contribution + shipping_context.shipping_regime_bias
        macro_component = macro_confidence.macro_alignment_score - macro_confidence.macro_conflict_score
        shipping_component = (
            shipping_context.shipping_alignment_score
            - shipping_context.shipping_conflict_score
            + shipping_context.shipping_support_boost
            - shipping_context.shipping_data_quality_penalty * 0.5
        )
        return {
            "directional": directional_component * directional_alignment * max(0.5, average_confidence) + shipping_context.shipping_directional_bias,
            "inefficiency": float(-inefficiency_score),
            "regime": regime_component,
            "macro": macro_component,
            "shipping": shipping_component,
            "risk": float(risk_penalty.total_penalty),
        }

    def _aggregate_components(self, component_scores: Dict[str, float]) -> float:
        return float(
            component_scores["directional"] * settings.composite.directional_weight
            + component_scores["inefficiency"] * settings.composite.inefficiency_weight
            + component_scores["regime"] * settings.composite.regime_weight
            + component_scores["macro"] * settings.composite.macro_weight
            + component_scores.get("shipping", 0.0) * settings.composite.shipping_weight
            - component_scores["risk"] * settings.composite.risk_weight
        )

    def _classify_suggestion(
        self,
        composite_score: float,
        directional_scores: Dict[int, float],
        directional_bias: str,
        entry_quality: str,
        signal_agreement: float,
        risk: RiskPenalty,
        macro_confidence: MacroConfidenceOverlay,
        shipping_context: ShippingSignalContext,
    ) -> Tuple[str, str, str, int]:
        effective_score = composite_score + macro_confidence.final_confidence_adjustment + shipping_context.shipping_support_boost - shipping_context.shipping_data_quality_penalty * 0.1
        preferred_horizon = self._preferred_horizon(directional_scores, effective_score)
        if signal_agreement < 0.25 or risk.total_penalty > 1.05:
            return "Regime Conflict / Avoid", "neutral", "wait", 0
        if directional_bias == "neutral" or abs(effective_score) < settings.composite.neutral_threshold:
            return "Neutral / No Edge", "neutral", "wait", 0

        long_bias = directional_bias in {"bullish", "strong_bullish"}
        short_bias = directional_bias in {"bearish", "strong_bearish"}

        if long_bias:
            if effective_score >= settings.composite.strong_threshold and entry_quality in {"Excellent", "Good"} and risk.total_penalty < 0.65:
                return "Strong Long Candidate", "long", "market", preferred_horizon
            if effective_score >= settings.composite.weak_threshold and entry_quality in {"Excellent", "Good", "Fair"}:
                return "Long Bias", "long", "limit", preferred_horizon
            if effective_score >= settings.composite.weak_threshold and entry_quality in {"Poor", "Very Poor"}:
                return "Long Bias / Wait for Pullback", "long", "limit", preferred_horizon
            return "Watchlist Long", "long", "wait", preferred_horizon

        if short_bias:
            if effective_score <= -settings.composite.strong_threshold and entry_quality in {"Excellent", "Good"} and risk.total_penalty < 0.65:
                return "Strong Short Candidate", "short", "market", preferred_horizon
            if effective_score <= -settings.composite.weak_threshold and entry_quality in {"Excellent", "Good", "Fair"}:
                return "Short Bias", "short", "limit", preferred_horizon
            if effective_score <= -settings.composite.weak_threshold and entry_quality in {"Poor", "Very Poor"}:
                return "Short Bias / Wait for Rally", "short", "limit", preferred_horizon
            return "Watchlist Short", "short", "wait", preferred_horizon

        return "Neutral / No Edge", "neutral", "wait", settings.evaluation.primary_horizon

    def _directional_bias_label(self, directional_scores: Dict[int, float]) -> str:
        weighted = self._weighted_directional_score(directional_scores)
        if weighted >= settings.composite.strong_threshold * 0.55:
            return "strong_bullish"
        if weighted >= settings.composite.neutral_threshold:
            return "bullish"
        if weighted <= -settings.composite.strong_threshold * 0.55:
            return "strong_bearish"
        if weighted <= -settings.composite.neutral_threshold:
            return "bearish"
        return "neutral"

    def _assess_entry_quality(self, data: pd.DataFrame, inefficiency_score: float, directional_bias: str) -> Tuple[str, float]:
        close = data["close"].astype(float)
        ma20 = close.rolling(20, min_periods=5).mean()
        vol20 = close.pct_change().rolling(20, min_periods=5).std(ddof=0)
        latest_vol = float(vol20.iloc[-1]) if not pd.isna(vol20.iloc[-1]) else 0.01
        latest_vol = max(0.005, latest_vol)
        price_extension = float((close.iloc[-1] - ma20.iloc[-1]) / (close.iloc[-1] * latest_vol)) if not pd.isna(ma20.iloc[-1]) else 0.0

        directional_sign = 1.0 if "bullish" in directional_bias else (-1.0 if "bearish" in directional_bias else 0.0)
        extension_against_entry = directional_sign * price_extension
        stretch_penalty = abs(float(inefficiency_score)) * 0.45 + max(0.0, extension_against_entry) * 0.30 + max(0.0, latest_vol - 0.03) * 6.0
        entry_score = float(max(0.0, min(1.0, 1.0 - stretch_penalty)))

        if entry_score >= 0.85:
            return "Excellent", entry_score
        if entry_score >= 0.70:
            return "Good", entry_score
        if entry_score >= 0.50:
            return "Fair", entry_score
        if entry_score >= 0.30:
            return "Poor", entry_score
        return "Very Poor", entry_score

    def _signal_agreement_score(
        self,
        directional_scores: Dict[int, float],
        macro_confidence: MacroConfidenceOverlay,
        shipping_context: ShippingSignalContext,
    ) -> float:
        alignment = self._directional_alignment(directional_scores)
        macro_headwind = max(0.0, macro_confidence.macro_conflict_score - macro_confidence.macro_alignment_score)
        shipping_headwind = max(0.0, shipping_context.shipping_conflict_score - shipping_context.shipping_alignment_score)
        agreement = alignment - 0.35 * macro_headwind - 0.30 * shipping_headwind
        return float(max(0.0, min(1.0, agreement)))

    def _compute_directional_confidence(self, directional_confidences: Dict[int, float], directional_scores: Dict[int, float]) -> float:
        if not directional_confidences:
            return 0.0
        avg = float(np.mean(list(directional_confidences.values())))
        alignment = self._directional_alignment(directional_scores)
        return float(max(0.0, min(1.0, avg * (0.65 + 0.35 * alignment))))

    def _compute_data_quality_confidence(self, data_quality_flag: str) -> float:
        if data_quality_flag == "good":
            return 0.95
        if data_quality_flag == "stale":
            return 0.65
        if data_quality_flag == "incomplete":
            return 0.40
        return 0.55

    def _compute_tradeability_confidence(
        self,
        directional_confidence: float,
        entry_quality_score: float,
        risk_penalty: RiskPenalty,
        signal_agreement: float,
        data_quality_confidence: float,
    ) -> float:
        risk_haircut = max(0.0, min(0.85, risk_penalty.total_penalty / 1.5))
        blended = (
            0.40 * directional_confidence
            + 0.30 * entry_quality_score
            + 0.20 * signal_agreement
            + 0.10 * data_quality_confidence
        )
        adjusted = blended * (1.0 - 0.55 * risk_haircut)
        return float(max(0.0, min(1.0, adjusted)))

    def _build_regime_probability_map(self, selected_regime: str, selected_probability: float) -> Dict[str, float]:
        selected = float(np.clip(selected_probability, 0.05, 0.95))
        remainder = max(0.0, 1.0 - selected)
        candidates = [
            "trend_following_bullish",
            "mean_reverting_rangebound",
            "risk_off",
            "trend_following_bearish",
            "volatile_reversal",
            "neutral",
        ]
        alternatives = [name for name in candidates if name != selected_regime][:2]
        if not alternatives:
            return {selected_regime: selected}
        alt_one = remainder * 0.6
        alt_two = remainder - alt_one
        probability_map = {
            selected_regime: selected,
            alternatives[0]: alt_one,
        }
        if len(alternatives) > 1:
            probability_map[alternatives[1]] = alt_two
        return probability_map

    def _label_component_contribution(self, component_name: str, value: float) -> str:
        magnitude = abs(float(value))
        sign = "Positive" if value >= 0 else "Negative"
        if component_name == "risk":
            sign = "Negative"
        if magnitude >= 1.0:
            level = "Strong"
        elif magnitude >= 0.45:
            level = "Moderate"
        elif magnitude >= 0.15:
            level = "Mild"
        else:
            level = "Neutral"
        if level == "Neutral":
            return "Neutral"
        return f"{level} {sign}"

    def _dominant_component_and_override(
        self,
        component_scores: Dict[str, float],
        entry_quality: str,
        risk_penalty: RiskPenalty,
        signal_agreement: float,
        recommendation: str,
    ) -> Tuple[str, Optional[str]]:
        dominant = max(component_scores.items(), key=lambda item: abs(float(item[1])))[0] if component_scores else "none"
        override_reason: Optional[str] = None
        if recommendation in {"Long Bias / Wait for Pullback", "Short Bias / Wait for Rally"}:
            override_reason = "Trade setup deferred because directional edge exists but entry quality is stretched."
        elif recommendation == "Regime Conflict / Avoid" or signal_agreement < 0.25:
            override_reason = "Trade not recommended due to conflicting directional, macro, or shipping signals."
        elif entry_quality in {"Poor", "Very Poor"}:
            override_reason = "Trade timing is unattractive due to stretched pricing relative to fair-value context."
        elif risk_penalty.total_penalty > 0.95:
            override_reason = "Risk overlay dominates with elevated penalty, suppressing actionable trade setup."
        return dominant, override_reason

    def _extract_drivers(
        self,
        regime_label: str,
        inefficiency_score: float,
        directional_scores: Dict[int, float],
        directional_bias: str,
        entry_quality: str,
        signal_agreement: float,
        macro_confidence: MacroConfidenceOverlay,
        risk: RiskPenalty,
        shipping_context: ShippingSignalContext,
    ) -> Tuple[List[str], List[str]]:
        supporting = [f"Regime context: {regime_label}", f"Directional bias: {directional_bias}"]
        contradictory: List[str] = []
        avg_directional = self._weighted_directional_score(directional_scores)
        supporting.append(f"Directional score stack: {avg_directional:.2f}")
        supporting.append(f"Entry quality: {entry_quality}")
        if abs(inefficiency_score) < 0.8:
            supporting.append(f"Pricing inefficiency is contained at {inefficiency_score:.2f} z")
        if inefficiency_score < -0.5:
            contradictory.append(f"Inefficiency is negative at {inefficiency_score:.2f} z and can suppress immediate entry")
        if abs(inefficiency_score) > 1.2:
            contradictory.append(f"Price appears stretched versus fair-value context ({inefficiency_score:.2f} z)")
        if macro_confidence.key_macro_drivers:
            supporting.extend(macro_confidence.key_macro_drivers[:2])
        if shipping_context.key_shipping_drivers:
            supporting.extend(shipping_context.key_shipping_drivers[:2])
        if signal_agreement < 0.45:
            contradictory.append(f"Signal agreement is low ({signal_agreement:.2f}), indicating cross-component conflict")
        if risk.total_penalty > 0.4:
            contradictory.append(f"Aggregate risk penalty is elevated at {risk.total_penalty:.2f}")
        if macro_confidence.macro_conflict_score > 0.25:
            contradictory.append("Macro context partially conflicts with the technical setup")
        if shipping_context.shipping_conflict_score > 0.25:
            contradictory.append("Shipping flow context conflicts with the current setup")
        return supporting[:4], contradictory[:3]

    def _identify_risks(
        self,
        inefficiency_instability: bool,
        risk: RiskPenalty,
        macro_confidence: MacroConfidenceOverlay,
        shipping_context: ShippingSignalContext,
        directional_bias: str,
        entry_quality: str,
    ) -> List[str]:
        risks = []
        if inefficiency_instability:
            risks.append("Fair-value relationship is unstable")
        if entry_quality in {"Poor", "Very Poor"}:
            risks.append("Entry quality is weak; waiting for better price location is prudent")
        if risk.volatility_spike > 0.25:
            risks.append("Recent realized volatility is elevated")
        if risk.signal_disagreement > 0.5:
            risks.append("Directional horizons are not aligned")
        risks.extend(macro_confidence.key_macro_risks[:2])
        if shipping_context.shipping_data_quality_penalty > 0.5:
            risks.append("Shipping coverage is sparse or noisy")
        if shipping_context.route_chokepoint_notes:
            risks.append(shipping_context.route_chokepoint_notes[0])
        if not risks:
            if directional_bias == "neutral":
                risks.append("Directional edge is weak, reducing expected payoff")
            else:
                risks.append("No major acute risk flags, but execution timing still matters")
        return risks[:4]

    def _generate_explanation(
        self,
        regime_label: str,
        directional_scores: Dict[int, float],
        directional_bias: str,
        entry_quality: str,
        inefficiency_score: float,
        risk: RiskPenalty,
        signal_agreement: float,
        macro_confidence: MacroConfidenceOverlay,
        shipping_context: ShippingSignalContext,
        composite_score: float,
        recommendation: str,
        override_reason: Optional[str],
        event_explanations: Optional[List[str]] = None,
    ) -> str:
        avg_directional = self._weighted_directional_score(directional_scores)
        macro_clause = ""
        if macro_confidence.macro_alignment_score > 0.25:
            macro_clause = " Macro context is supportive."
        elif macro_confidence.macro_conflict_score > 0.25:
            macro_clause = " Macro context is a headwind."
        shipping_clause = ""
        if shipping_context.shipping_alignment_score > 0.25:
            shipping_clause = " Shipping context is supportive."
        elif shipping_context.shipping_conflict_score > 0.25:
            shipping_clause = " Shipping context is a headwind."
        explanation = (
            f"{self._human_bias_label(directional_bias)} directional momentum is observed across horizons (avg score {avg_directional:.2f}) under {regime_label}. "
            f"Entry quality is {entry_quality} with inefficiency at {inefficiency_score:.2f} z and risk penalty at {risk.total_penalty:.2f}. "
            f"Signal agreement is {signal_agreement:.2f}, leading to recommendation: {recommendation}."
            f"{macro_clause}{shipping_clause}"
        )
        if override_reason:
            explanation = explanation + f" Override: {override_reason}"
        if event_explanations:
            explanation = explanation + " Event intelligence: " + " | ".join(event_explanations[:2])
        return explanation

    def _human_bias_label(self, directional_bias: str) -> str:
        labels = {
            "strong_bullish": "Strong bullish",
            "bullish": "Bullish",
            "strong_bearish": "Strong bearish",
            "bearish": "Bearish",
            "neutral": "Neutral",
        }
        return labels.get(directional_bias, "Neutral")

    def _compute_final_confidence(
        self,
        composite_score: float,
        macro_confidence: MacroConfidenceOverlay,
        shipping_context: ShippingSignalContext,
        data_quality_flag: str,
    ) -> float:
        shipping_adjustment = shipping_context.shipping_support_boost - shipping_context.shipping_conflict_score * 0.10 - shipping_context.shipping_data_quality_penalty * 0.20
        raw = 1.0 / (1.0 + np.exp(-(abs(composite_score) + macro_confidence.final_confidence_adjustment + shipping_adjustment) * 1.5))
        if data_quality_flag == "stale":
            raw *= 0.8
        bounded = float(max(0.0, min(1.0, raw)))
        return self._apply_confidence_calibration(bounded)

    def _apply_confidence_calibration(self, raw_confidence: float) -> float:
        calibration = self.parameter_state.get("confidence_calibration", {})
        anchors = calibration.get("anchors", []) if isinstance(calibration, dict) else []
        if not anchors:
            return float(raw_confidence)

        points = sorted(
            (
                float(anchor.get("confidence", 0.0)),
                float(anchor.get("calibrated_hit_rate", 0.0)),
            )
            for anchor in anchors
            if isinstance(anchor, dict)
        )
        if not points:
            return float(raw_confidence)
        x = np.array([point[0] for point in points], dtype=float)
        y = np.array([point[1] for point in points], dtype=float)
        calibrated = float(np.interp(raw_confidence, x, y))
        return float(np.clip(calibrated, 0.0, 1.0))

    def _calibrated_regime_probability(self, regime_label: str, base_probability: float) -> float:
        calibration = self.parameter_state.get("regime_calibration", {})
        if not isinstance(calibration, dict):
            return float(base_probability)
        regime_map = calibration.get("regime_map", {})
        if not isinstance(regime_map, dict):
            return float(base_probability)
        calibrated = regime_map.get(regime_label)
        if calibrated is None:
            return float(base_probability)
        return float(np.clip(float(calibrated), 0.0, 1.0))

    def _weighted_directional_score(self, directional_scores: Dict[int, float]) -> float:
        if not directional_scores:
            return 0.0
        weights = settings.signal.directional_horizon_weights
        weighted_sum = 0.0
        total_weight = 0.0
        for horizon, score in directional_scores.items():
            weight = float(weights.get(str(horizon), 1.0))
            weighted_sum += weight * float(score)
            total_weight += weight
        return weighted_sum / total_weight if total_weight else 0.0

    def _directional_alignment(self, directional_scores: Dict[int, float]) -> float:
        if not directional_scores:
            return 0.0
        weighted_score = self._weighted_directional_score(directional_scores)
        dominant_sign = 0 if abs(weighted_score) < 1e-12 else int(np.sign(weighted_score))
        if dominant_sign == 0:
            return 0.5
        weights = settings.signal.directional_horizon_weights
        aligned_weight = 0.0
        total_weight = 0.0
        for horizon, score in directional_scores.items():
            weight = float(weights.get(str(horizon), 1.0))
            total_weight += weight
            if int(np.sign(score)) == dominant_sign:
                aligned_weight += weight
        return aligned_weight / total_weight if total_weight else 0.0

    def _directional_disagreement_penalty(self, directional_signals: List[DirectionalSignal]) -> float:
        if not directional_signals:
            return 0.0
        scores = {signal.horizon: signal.score for signal in directional_signals}
        confidence = {signal.horizon: signal.confidence for signal in directional_signals}
        dispersion = float(np.std(list(scores.values())))
        alignment = self._directional_alignment(scores)
        avg_confidence = float(np.mean(list(confidence.values()))) if confidence else 0.0
        disagreement = (1.0 - alignment) * (0.25 + avg_confidence)
        return float(min(0.75, 0.5 * dispersion + disagreement))

    def _preferred_horizon(self, directional_scores: Dict[int, float], effective_score: float) -> int:
        if not directional_scores:
            return settings.evaluation.primary_horizon
        target_sign = int(np.sign(effective_score)) if abs(effective_score) > 1e-12 else 0
        weights = settings.signal.directional_horizon_weights
        ranked = sorted(
            directional_scores.items(),
            key=lambda item: abs(float(item[1])) * float(weights.get(str(item[0]), 1.0)),
            reverse=True,
        )
        for horizon, score in ranked:
            if target_sign == 0 or int(np.sign(score)) == target_sign:
                return int(horizon)
        return int(ranked[0][0])

    def _extract_macro_highlights(self, macro_features: List[MacroFeature], timestamp: datetime) -> Dict[str, float]:
        highlights: Dict[str, float] = {}
        latest_by_feature: Dict[str, MacroFeature] = {}
        for feature in macro_features:
            if feature.timestamp <= timestamp:
                current = latest_by_feature.get(feature.feature_name)
                if current is None or feature.timestamp > current.timestamp:
                    latest_by_feature[feature.feature_name] = feature
        for feature_name in ("cpi_yoy", "gdp_yoy", "real_rate_in", "usd_inr", "macro_sentiment_score"):
            if feature_name in latest_by_feature:
                highlights[feature_name] = float(latest_by_feature[feature_name].value)
        return highlights

    def _generate_macro_explanation(self, macro_confidence: MacroConfidenceOverlay) -> str:
        if not macro_confidence.key_macro_drivers and not macro_confidence.key_macro_risks:
            return "No dominant macro overlay."
        parts = []
        if macro_confidence.key_macro_drivers:
            parts.append("Drivers: " + "; ".join(macro_confidence.key_macro_drivers[:2]))
        if macro_confidence.key_macro_risks:
            parts.append("Risks: " + "; ".join(macro_confidence.key_macro_risks[:2]))
        return " ".join(parts)

    def _build_signal_id(self, commodity: str, timestamp: datetime) -> str:
        return f"{commodity}-{timestamp.strftime('%Y%m%d%H%M%S')}-{uuid.uuid4().hex[:8]}"
