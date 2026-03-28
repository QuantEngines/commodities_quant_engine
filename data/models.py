from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from typing import Any, Dict, List, Optional


def _format_optional_number(value: Optional[float], digits: int = 2, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    format_spec = f"{'+' if signed else ''}.{digits}f"
    return format(float(value), format_spec)


def _compact_list(items: List[str], limit: int = 4) -> str:
    filtered = [str(item).strip() for item in items if str(item).strip()]
    if not filtered:
        return "None"
    return "; ".join(filtered[:limit])


def _sorted_score_items(scores: Dict[Any, Any]) -> List[tuple[Any, float]]:
    normalized: List[tuple[Any, float]] = []
    for key, value in scores.items():
        try:
            normalized.append((key, float(value)))
        except (TypeError, ValueError):
            continue
    return sorted(
        normalized,
        key=lambda item: (0, int(item[0])) if str(item[0]).isdigit() else (1, str(item[0])),
    )


def _top_abs_items(values: Dict[str, Any], limit: int = 5) -> Dict[str, float]:
    normalized: List[tuple[str, float]] = []
    for key, value in values.items():
        try:
            normalized.append((str(key), float(value)))
        except (TypeError, ValueError):
            continue
    normalized.sort(key=lambda item: abs(item[1]), reverse=True)
    return {key: round(value, 4) for key, value in normalized[:limit]}


def _format_score_map(values: Dict[str, Any], limit: int = 5) -> str:
    top_items = list(_top_abs_items(values, limit=limit).items())
    if not top_items:
        return "n/a"
    return ", ".join(f"{key}={value:.2f}" for key, value in top_items)


def _format_directional_scores(scores: Dict[Any, Any]) -> str:
    ordered = _sorted_score_items(scores)
    if not ordered:
        return "n/a"
    return ", ".join(f"{key}D={value:.2f}" for key, value in ordered)


def _primary_regime_probability(regime_label: str, probabilities: Dict[str, float]) -> Optional[float]:
    if regime_label in probabilities:
        return float(probabilities[regime_label])
    if probabilities:
        return float(max(probabilities.values()))
    return None


@dataclass
class Commodity:
    symbol: str
    name: str
    exchange: str
    segment: str
    base_currency: str = "INR"
    contract_multiplier: int = 1
    tick_size: float = 0.01
    expiry_rule: str = "monthly"
    roll_days_before_expiry: int = 5
    liquidity_threshold: int = 1000
    seasonality_class: str = "neutral"
    macro_sensitivity: List[str] = field(default_factory=list)


@dataclass
class Contract:
    commodity: str
    symbol: str
    expiry_date: date
    lot_size: int
    tick_size: float
    multiplier: int
    exchange: str
    segment: str
    first_notice_date: Optional[date] = None
    last_trading_date: Optional[date] = None
    quote_currency: str = "INR"
    settlement_type: Optional[str] = None
    source: str = "fallback"
    is_fallback: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def contract_code(self) -> str:
        return self.symbol

    @property
    def active_until(self) -> date:
        return self.last_trading_date or self.expiry_date


@dataclass
class OHLCV:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    open_interest: Optional[int] = None
    contract: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RegimeState:
    label: str
    probability: float
    confidence: float
    features: Dict[str, float]
    timestamp: datetime


@dataclass
class DirectionalSignal:
    commodity: str
    horizon: int
    score: float
    confidence: float
    features: Dict[str, float]
    timestamp: datetime
    model_version: str = "default"


@dataclass
class InefficiencySignal:
    commodity: str
    deviation_z: float
    persistence: int
    timestamp: datetime
    instability_warning: bool
    fair_value_gap: float = 0.0
    cross_sectional_score: Optional[float] = None


@dataclass
class RiskPenalty:
    volatility_spike: float
    signal_disagreement: float
    event_risk: float
    liquidity_penalty: float
    total_penalty: float


@dataclass
class DataQualityReport:
    flag: str
    issues: List[str]
    stats: Dict[str, float]
    as_of: datetime
    is_valid: bool = True


@dataclass
class Suggestion:
    timestamp: datetime
    commodity: str
    exchange: str
    active_contract: str
    regime_label: str
    regime_probabilities: Dict[str, float]
    directional_scores: Dict[int, float]
    inefficiency_score: float
    risk_penalty: float
    composite_score: float
    final_category: str
    preferred_direction: str
    suggested_entry_style: str
    suggested_holding_horizon: int
    key_supporting_drivers: List[str]
    key_contradictory_drivers: List[str]
    principal_risks: List[str]
    explanation_summary: str
    data_quality_flag: str
    confidence_score: float
    signal_id: Optional[str] = None
    model_version: Optional[str] = None
    config_version: Optional[str] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    macro_regime_summary: Optional[str] = None
    macro_feature_highlights: Dict[str, float] = field(default_factory=dict)
    macro_alignment_score: Optional[float] = None
    macro_conflict_score: Optional[float] = None
    macro_event_risk_flag: bool = False
    macro_confidence_adjustment: float = 0.0
    macro_explanation_summary: Optional[str] = None
    key_macro_drivers: List[str] = field(default_factory=list)
    key_macro_risks: List[str] = field(default_factory=list)
    news_narrative_summary: Optional[str] = None

    def to_markdown(self) -> str:
        diagnostics = self.diagnostics or {}
        component_scores = diagnostics.get("component_scores", {})
        directional_confidences = diagnostics.get("directional_confidences", {})
        feature_vector = diagnostics.get("feature_vector", {})
        event_features = diagnostics.get("event_intelligence_features", {})
        quality_issues = diagnostics.get("quality_issues", [])
        regime_probability = _primary_regime_probability(self.regime_label, self.regime_probabilities or {})
        return f"""
# {self.commodity} Trading Suggestion

## Decision Snapshot
- **Timestamp:** {self.timestamp}
- **Exchange:** {self.exchange}
- **Active Contract:** {self.active_contract}
- **Signal ID:** {self.signal_id or "n/a"}
- **Decision:** {self.final_category} | direction={self.preferred_direction} | confidence={self.confidence_score:.2f} | score={self.composite_score:.2f}
- **Execution:** entry={self.suggested_entry_style} | horizon={self.suggested_holding_horizon}D | data_quality={self.data_quality_flag}
- **Regime:** {self.regime_label} | p={_format_optional_number(regime_probability, digits=2)} | macro_align={_format_optional_number(self.macro_alignment_score, digits=2)} | macro_conflict={_format_optional_number(self.macro_conflict_score, digits=2)} | event_risk={'high' if self.macro_event_risk_flag else 'low'} | macro_conf_adj={_format_optional_number(self.macro_confidence_adjustment, digits=2, signed=True)}
- **Model Lineage:** model={self.model_version or "n/a"} | config={self.config_version or "n/a"}

## Signal Anatomy
- **Directional Term Structure:** {_format_directional_scores(self.directional_scores)}
- **Directional Confidence:** {_format_directional_scores(directional_confidences)}
- **Inefficiency Score:** {self.inefficiency_score:.2f}
- **Risk Penalty:** {self.risk_penalty:.2f}
- **Component Stack:** {_format_score_map(component_scores, limit=5)}
- **Feature Highlights:** {_format_score_map(feature_vector, limit=5)}
- **Event Overlay:** {_format_score_map(event_features, limit=4)}
- **Macro Features:** {_format_score_map(self.macro_feature_highlights, limit=5)}

## Thesis
{self.explanation_summary}

- **Supporting Drivers:** {_compact_list(self.key_supporting_drivers, limit=4)}
- **Contradictions:** {_compact_list(self.key_contradictory_drivers, limit=3)}
- **Principal Risks:** {_compact_list(self.principal_risks, limit=4)}
- **Quality Issues:** {_compact_list(quality_issues, limit=4)}

## Macro Context
- **Macro Regime:** {self.macro_regime_summary or "Not available"}
- **Macro Alignment:** {_format_optional_number(self.macro_alignment_score, digits=2)}
- **Macro Conflict:** {_format_optional_number(self.macro_conflict_score, digits=2)}
- **Macro Confidence Adjustment:** {_format_optional_number(self.macro_confidence_adjustment, digits=2, signed=True)}
- **Macro Explanation:** {self.macro_explanation_summary or "Not available"}
- **Key Macro Drivers:** {_compact_list(self.key_macro_drivers, limit=4)}
- **Key Macro Risks:** {_compact_list(self.key_macro_risks, limit=4)}
- **News Narrative:** {self.news_narrative_summary or "No significant macro news"}
""".strip()


@dataclass
class SignalSnapshot:
    signal_id: str
    timestamp: datetime
    commodity: str
    contract: str
    exchange: str
    signal_category: str
    direction: str
    conviction: float
    regime_label: str
    regime_probability: float
    inefficiency_score: float
    composite_score: float
    suggested_horizon: int
    directional_scores: Dict[int, float]
    key_drivers: List[str]
    key_risks: List[str]
    component_scores: Dict[str, float]
    feature_vector: Dict[str, float]
    model_version: str
    config_version: str
    data_quality_flag: str
    macro_alignment_score: Optional[float] = None
    macro_conflict_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SignalPackage:
    suggestion: Suggestion
    snapshot: SignalSnapshot
    quality_report: DataQualityReport


@dataclass
class SignalEvaluationRecord:
    signal_id: str
    timestamp: datetime
    commodity: str
    horizon: int
    direction: str
    confidence: float
    composite_score: float
    realized_return: float
    signed_return: float
    direction_correct: bool
    excess_return: float
    volatility_adjusted_return: float
    max_favorable_excursion: float
    max_adverse_excursion: float
    follow_through_ratio: float
    reversal_probability: float
    event_window_flag: bool
    regime_label: str
    realized_regime_label: str
    regime_alignment: bool
    confidence_bucket: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationArtifact:
    commodity: str
    created_at: datetime
    horizons: List[int]
    summary_metrics: Dict[str, Any]
    degradation_alerts: List[str]
    scorecards: Dict[str, Any]
    detailed_path: Optional[str] = None
    summary_path: Optional[str] = None


@dataclass
class ParameterVersion:
    version_id: str
    commodity: str
    created_at: datetime
    parent_version_id: Optional[str]
    parameters: Dict[str, Any]
    evidence: Dict[str, Any]
    metrics: Dict[str, float]
    mode: str
    approved: bool
    active: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AdaptationDecision:
    commodity: str
    created_at: datetime
    incumbent_version_id: Optional[str]
    candidate_version_id: Optional[str]
    promoted: bool
    approved: bool
    reason: str
    evidence: Dict[str, Any]
    safety_checks: Dict[str, bool]
    mode: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MacroSeries:
    series_id: str
    timestamp: datetime
    value: float
    unit: str
    frequency: str
    source: str
    is_revised: bool = False
    original_timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MacroEvent:
    event_id: str
    event_type: str
    country: str
    timestamp: datetime
    title: str
    description: Optional[str] = None
    expected_impact: str = "medium"
    actual_value: Optional[float] = None
    consensus_value: Optional[float] = None
    previous_value: Optional[float] = None
    source: str = "official"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class NewsItem:
    news_id: str
    timestamp: datetime
    headline: str
    source: str
    url: Optional[str] = None
    content: Optional[str] = None
    relevance_score: float = 0.0
    sentiment_score: Optional[float] = None
    topics: List[str] = field(default_factory=list)
    commodity_tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MacroFeature:
    feature_name: str
    timestamp: datetime
    value: float
    commodity: Optional[str] = None
    source_series: List[str] = field(default_factory=list)
    transform: str = ""
    lag_days: int = 0
    frequency: str = "daily"
    missing_data_policy: str = "forward_fill"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MacroRegimeState:
    base_regime: str
    macro_overlay: str
    combined_label: str
    probability: float
    confidence: float
    macro_contribution: float
    key_macro_drivers: List[str]
    timestamp: datetime
    features: Dict[str, float] = field(default_factory=dict)


@dataclass
class MacroDirectionalOverlay:
    commodity: str
    horizon: int
    base_score: float
    macro_adjustment: float
    adjusted_score: float
    macro_alignment: float
    confidence_multiplier: float
    key_macro_factors: List[str]
    timestamp: datetime


@dataclass
class MacroConfidenceOverlay:
    commodity: str
    timestamp: datetime
    macro_alignment_score: float
    macro_conflict_score: float
    event_risk_penalty: float
    news_uncertainty_penalty: float
    support_boost: float
    final_confidence_adjustment: float
    key_macro_drivers: List[str]
    key_macro_risks: List[str]
    news_narrative_summary: Optional[str] = None
