from __future__ import annotations

import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...config.settings import settings
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
from ...signals.directional.directional_alpha import DirectionalAlphaEngine
from ...signals.inefficiency.inefficiency_engine import InefficiencyEngine
from ...signals.macro.confidence_overlay import MacroConfidenceOverlay as MacroConfidenceOverlayEngine
from ...signals.macro.directional_overlay import MacroDirectionalOverlay
from ...signals.macro.regime_overlay import MacroRegimeOverlay


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

    def generate_suggestion(
        self,
        data: pd.DataFrame,
        commodity: str,
        macro_features: Optional[List[MacroFeature]] = None,
        macro_events: Optional[List[MacroEvent]] = None,
        as_of_timestamp: Optional[datetime] = None,
    ) -> Suggestion:
        return self.generate_signal_package(
            data=data,
            commodity=commodity,
            macro_features=macro_features,
            macro_events=macro_events,
            as_of_timestamp=as_of_timestamp,
        ).suggestion

    def generate_signal_package(
        self,
        data: pd.DataFrame,
        commodity: str,
        macro_features: Optional[List[MacroFeature]] = None,
        macro_events: Optional[List[MacroEvent]] = None,
        as_of_timestamp: Optional[datetime] = None,
    ) -> SignalPackage:
        macro_features = macro_features or []
        macro_events = macro_events or []
        as_of_timestamp = as_of_timestamp or self._infer_timestamp(data)

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
            as_of_timestamp=as_of_timestamp,
        )
        directional_scores = {signal.horizon: signal.score for signal in directional_signals}
        directional_confidences = {signal.horizon: signal.confidence for signal in directional_signals}
        inefficiency = self.inefficiency_engine.detect_inefficiency(data, commodity)
        base_direction = self._determine_direction(directional_scores)
        macro_confidence = self.macro_confidence_overlay.compute_confidence_adjustment(
            commodity=commodity,
            timestamp=as_of_timestamp,
            macro_features=macro_features,
            macro_events=macro_events,
            base_regime=macro_regime.combined_label,
            base_direction=base_direction,
        )
        risk_penalty = self._calculate_risk_penalty(data, directional_signals, macro_confidence, macro_events)
        component_scores = self._compute_component_scores(directional_scores, directional_confidences, inefficiency.deviation_z, macro_regime.macro_contribution, risk_penalty, macro_confidence)
        composite_score = self._aggregate_components(component_scores)
        category, direction, entry_style, horizon = self._classify_suggestion(composite_score, directional_scores, risk_penalty, macro_confidence)
        supporting, contradictory = self._extract_drivers(macro_regime.combined_label, inefficiency.deviation_z, directional_scores, macro_confidence, risk_penalty)
        risks = self._identify_risks(inefficiency.instability_warning, risk_penalty, macro_confidence)
        explanation = self._generate_explanation(macro_regime.combined_label, directional_scores, inefficiency.deviation_z, risk_penalty, macro_confidence, composite_score)

        active_contract = contract_master.get_active_contract(commodity, as_of_timestamp.date())
        signal_id = self._build_signal_id(commodity, as_of_timestamp)
        suggestion = Suggestion(
            timestamp=as_of_timestamp,
            commodity=commodity,
            exchange=active_contract.exchange if active_contract else settings.commodities[commodity].exchange,
            active_contract=active_contract.symbol if active_contract else commodity,
            regime_label=macro_regime.combined_label,
            regime_probabilities={macro_regime.combined_label: macro_regime.probability},
            directional_scores=directional_scores,
            inefficiency_score=inefficiency.deviation_z,
            risk_penalty=risk_penalty.total_penalty,
            composite_score=composite_score,
            final_category=category,
            preferred_direction=direction,
            suggested_entry_style=entry_style,
            suggested_holding_horizon=horizon,
            key_supporting_drivers=supporting,
            key_contradictory_drivers=contradictory,
            principal_risks=risks,
            explanation_summary=explanation,
            data_quality_flag=quality_report.flag,
            confidence_score=self._compute_final_confidence(composite_score, macro_confidence, quality_report.flag),
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
            },
            macro_regime_summary=macro_regime.combined_label,
            macro_feature_highlights=self._extract_macro_highlights(macro_features, as_of_timestamp),
            macro_alignment_score=macro_confidence.macro_alignment_score,
            macro_conflict_score=macro_confidence.macro_conflict_score,
            macro_event_risk_flag=macro_confidence.event_risk_penalty > 0.25,
            macro_confidence_adjustment=macro_confidence.final_confidence_adjustment,
            macro_explanation_summary=self._generate_macro_explanation(macro_confidence),
            key_macro_drivers=macro_confidence.key_macro_drivers,
            key_macro_risks=macro_confidence.key_macro_risks,
            news_narrative_summary=macro_confidence.news_narrative_summary,
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
            regime_probability=macro_regime.probability,
            inefficiency_score=inefficiency.deviation_z,
            composite_score=composite_score,
            suggested_horizon=horizon,
            directional_scores=directional_scores,
            key_drivers=supporting,
            key_risks=risks,
            component_scores=component_scores,
            feature_vector={name: float(latest_features.get(name, 0.0)) for name in settings.signal.directional_feature_names},
            model_version=str(self.parameter_state.get("version_id", "default")),
            config_version=settings.config_version,
            data_quality_flag=quality_report.flag,
            macro_alignment_score=macro_confidence.macro_alignment_score,
            macro_conflict_score=macro_confidence.macro_conflict_score,
            metadata={
                "contradictory_drivers": contradictory,
                "macro_drivers": macro_confidence.key_macro_drivers,
                "macro_risks": macro_confidence.key_macro_risks,
                "directional_confidences": directional_confidences,
            },
        )
        return SignalPackage(suggestion=suggestion, snapshot=snapshot, quality_report=quality_report)

    def _build_directional_signals(
        self,
        commodity: str,
        latest_features: Dict[str, float],
        macro_features: List[MacroFeature],
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
            signals.append(
                DirectionalSignal(
                    commodity=commodity,
                    horizon=horizon,
                    score=macro_directional.adjusted_score,
                    confidence=confidence,
                    features={name: float(latest_features.get(name, 0.0)) for name in settings.signal.directional_feature_names},
                    timestamp=as_of_timestamp,
                    model_version=str(self.parameter_state.get("version_id", "default")),
                )
            )
        return signals

    def _infer_timestamp(self, data: pd.DataFrame) -> datetime:
        if isinstance(data.index, pd.DatetimeIndex):
            return data.index[-1].to_pydatetime()
        return datetime.utcnow()

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
    ) -> RiskPenalty:
        vol_penalty = max(0.0, float(data["close"].pct_change().rolling(20, min_periods=5).std(ddof=0).iloc[-1] * 10.0 - 0.15))
        disagreement_penalty = self._directional_disagreement_penalty(directional_signals)
        liquidity_penalty = 0.15 if data["volume"].iloc[-5:].mean() < max(100.0, data["volume"].median() * 0.35) else 0.0
        event_risk = macro_confidence.event_risk_penalty
        if any(event.expected_impact == "high" for event in macro_events):
            event_risk = max(event_risk, 0.1)
        total_penalty = min(1.5, vol_penalty + disagreement_penalty + liquidity_penalty + event_risk)
        return RiskPenalty(
            volatility_spike=vol_penalty,
            signal_disagreement=disagreement_penalty,
            event_risk=event_risk,
            liquidity_penalty=liquidity_penalty,
            total_penalty=total_penalty,
        )

    def _compute_component_scores(
        self,
        directional_scores: Dict[int, float],
        directional_confidences: Dict[int, float],
        inefficiency_score: float,
        macro_regime_contribution: float,
        risk_penalty: RiskPenalty,
        macro_confidence: MacroConfidenceOverlay,
    ) -> Dict[str, float]:
        directional_component = self._weighted_directional_score(directional_scores)
        directional_alignment = self._directional_alignment(directional_scores)
        average_confidence = float(np.mean(list(directional_confidences.values()))) if directional_confidences else 0.0
        regime_component = macro_regime_contribution
        macro_component = macro_confidence.macro_alignment_score - macro_confidence.macro_conflict_score
        return {
            "directional": directional_component * directional_alignment * max(0.5, average_confidence),
            "inefficiency": float(-inefficiency_score),
            "regime": regime_component,
            "macro": macro_component,
            "risk": float(risk_penalty.total_penalty),
        }

    def _aggregate_components(self, component_scores: Dict[str, float]) -> float:
        return float(
            component_scores["directional"] * settings.composite.directional_weight
            + component_scores["inefficiency"] * settings.composite.inefficiency_weight
            + component_scores["regime"] * settings.composite.regime_weight
            + component_scores["macro"] * settings.composite.macro_weight
            - component_scores["risk"] * settings.composite.risk_weight
        )

    def _classify_suggestion(
        self,
        composite_score: float,
        directional_scores: Dict[int, float],
        risk: RiskPenalty,
        macro_confidence: MacroConfidenceOverlay,
    ) -> Tuple[str, str, str, int]:
        effective_score = composite_score + macro_confidence.final_confidence_adjustment
        preferred_horizon = self._preferred_horizon(directional_scores, effective_score)
        if abs(effective_score) < settings.composite.neutral_threshold or risk.total_penalty > 0.95:
            return "Neutral / No Edge", "neutral", "wait", 0
        if effective_score >= settings.composite.strong_threshold:
            return "Strong Long Candidate", "long", "market", preferred_horizon
        if effective_score >= settings.composite.weak_threshold:
            return "Long Bias, wait for pullback", "long", "limit", preferred_horizon
        if effective_score <= -settings.composite.strong_threshold:
            return "Strong Short Candidate", "short", "market", preferred_horizon
        if effective_score <= -settings.composite.weak_threshold:
            return "Short Bias, wait for rally", "short", "limit", preferred_horizon
        return "Weak Relative Edge", "neutral", "wait", settings.evaluation.primary_horizon

    def _extract_drivers(
        self,
        regime_label: str,
        inefficiency_score: float,
        directional_scores: Dict[int, float],
        macro_confidence: MacroConfidenceOverlay,
        risk: RiskPenalty,
    ) -> Tuple[List[str], List[str]]:
        supporting = [f"Regime: {regime_label}"]
        contradictory: List[str] = []
        avg_directional = self._weighted_directional_score(directional_scores)
        supporting.append(f"Average directional score: {avg_directional:.2f}")
        if abs(inefficiency_score) > 1.0:
            supporting.append(f"Deviation from fair value is {inefficiency_score:.2f} z")
        if macro_confidence.key_macro_drivers:
            supporting.extend(macro_confidence.key_macro_drivers[:2])
        if risk.total_penalty > 0.4:
            contradictory.append(f"Aggregate risk penalty is elevated at {risk.total_penalty:.2f}")
        if macro_confidence.macro_conflict_score > 0.25:
            contradictory.append("Macro context partially conflicts with the technical setup")
        return supporting[:4], contradictory[:3]

    def _identify_risks(
        self,
        inefficiency_instability: bool,
        risk: RiskPenalty,
        macro_confidence: MacroConfidenceOverlay,
    ) -> List[str]:
        risks = []
        if inefficiency_instability:
            risks.append("Fair-value relationship is unstable")
        if risk.volatility_spike > 0.25:
            risks.append("Recent realized volatility is elevated")
        if risk.signal_disagreement > 0.5:
            risks.append("Directional horizons are not aligned")
        risks.extend(macro_confidence.key_macro_risks[:2])
        return risks[:4]

    def _generate_explanation(
        self,
        regime_label: str,
        directional_scores: Dict[int, float],
        inefficiency_score: float,
        risk: RiskPenalty,
        macro_confidence: MacroConfidenceOverlay,
        composite_score: float,
    ) -> str:
        avg_directional = self._weighted_directional_score(directional_scores)
        macro_clause = ""
        if macro_confidence.macro_alignment_score > 0.25:
            macro_clause = " Macro context is supportive."
        elif macro_confidence.macro_conflict_score > 0.25:
            macro_clause = " Macro context is a headwind."
        return (
            f"Regime is {regime_label}. Average directional score is {avg_directional:.2f}, "
            f"inefficiency score is {inefficiency_score:.2f}, and aggregate risk penalty is {risk.total_penalty:.2f}. "
            f"Composite score is {composite_score:.2f}.{macro_clause}"
        )

    def _compute_final_confidence(
        self,
        composite_score: float,
        macro_confidence: MacroConfidenceOverlay,
        data_quality_flag: str,
    ) -> float:
        raw = 1.0 / (1.0 + np.exp(-(abs(composite_score) + macro_confidence.final_confidence_adjustment) * 1.5))
        if data_quality_flag == "stale":
            raw *= 0.8
        return float(max(0.0, min(1.0, raw)))

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
