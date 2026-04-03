"""Non-binding portfolio position suggestions derived from model signals."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd

from ...config.settings import settings
from ...data.storage.local import LocalStorage


class PositionDirection(str, Enum):
    """Directional stance for a recommendation."""

    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class PositionSuggestion:
    """Budget-aware, non-executable recommendation for user review."""

    suggestion_id: str
    commodity: str
    exchange: str
    direction: PositionDirection
    target_weight: float
    target_notional: float
    target_quantity: int
    current_quantity: int
    rebalance_delta: int
    reference_price: float
    signal_score: float
    confidence_score: float
    stop_reference: Optional[float] = None
    take_profit_reference: Optional[float] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc).replace(tzinfo=None))
    rationale: List[str] = field(default_factory=list)
    notes: str = "For user review only. The engine never places orders."

    def to_dict(self) -> Dict[str, object]:
        return {
            "suggestion_id": self.suggestion_id,
            "commodity": self.commodity,
            "exchange": self.exchange,
            "direction": self.direction.value,
            "target_weight": self.target_weight,
            "target_notional": self.target_notional,
            "target_quantity": self.target_quantity,
            "current_quantity": self.current_quantity,
            "rebalance_delta": self.rebalance_delta,
            "reference_price": self.reference_price,
            "signal_score": self.signal_score,
            "confidence_score": self.confidence_score,
            "stop_reference": self.stop_reference,
            "take_profit_reference": self.take_profit_reference,
            "timestamp": self.timestamp.isoformat(),
            "rationale": self.rationale,
            "notes": self.notes,
        }


@dataclass
class SuggestionMetrics:
    """Summary metrics for generated recommendations."""

    total_suggestions_generated: int = 0
    total_target_notional: float = 0.0
    directions: Dict[str, int] = field(default_factory=dict)
    last_generation_time: Optional[datetime] = None

    def to_dict(self) -> Dict[str, object]:
        return {
            "total_suggestions_generated": self.total_suggestions_generated,
            "total_target_notional": self.total_target_notional,
            "directions": self.directions,
            "last_generation_time": (
                self.last_generation_time.isoformat()
                if self.last_generation_time
                else None
            ),
        }


class PositionSuggestionEngine:
    """Converts optimized portfolio outputs into non-binding position suggestions."""

    def __init__(self, storage: Optional[LocalStorage] = None):
        self.storage = storage or LocalStorage()
        self.metrics = SuggestionMetrics()
        self.suggestion_history: List[PositionSuggestion] = []

    def generate_portfolio_suggestions(
        self,
        portfolio_weights: Dict[str, float],
        signal_data: Dict[str, Dict[str, object]],
        price_data: Dict[str, float],
        portfolio_budget: float,
        market_volatility: Optional[Dict[str, float]] = None,
        current_positions: Optional[Dict[str, int]] = None,
    ) -> Dict[str, PositionSuggestion]:
        """Build position suggestions without generating broker instructions."""
        suggestions: Dict[str, PositionSuggestion] = {}
        current_positions = current_positions or {}

        for commodity, weight in portfolio_weights.items():
            if abs(weight) <= 0:
                continue

            commodity_signal = signal_data.get(commodity, {})
            signal_score = float(commodity_signal.get("composite_score", 0.0))
            confidence_score = float(commodity_signal.get("confidence_score", 0.0))
            if abs(signal_score) < 0.05:
                continue

            reference_price = float(price_data.get(commodity, 0.0))
            if reference_price <= 0:
                continue

            direction = self._resolve_direction(signal_score)
            if direction == PositionDirection.FLAT:
                continue

            target_notional = max(0.0, float(portfolio_budget) * abs(float(weight)))
            volatility = None
            if market_volatility:
                volatility = market_volatility.get(commodity)
            if volatility and volatility > 0:
                target_notional *= min(1.0, settings.evaluation_pricing.target_annualized_vol / volatility)

            commodity_config = settings.commodities.get(commodity)
            lot_size = (
                commodity_config.contract_multiplier
                if commodity_config and commodity_config.contract_multiplier > 0
                else 1
            )
            lots_needed = target_notional / (reference_price * lot_size)
            signed_target_quantity = max(1, int(lots_needed))
            if signed_target_quantity <= 0:
                continue
            if direction == PositionDirection.SHORT:
                signed_target_quantity *= -1

            current_quantity = int(current_positions.get(commodity, 0))
            rebalance_delta = signed_target_quantity - current_quantity

            suggestion = PositionSuggestion(
                suggestion_id=f"{commodity}_{datetime.now(timezone.utc).replace(tzinfo=None).strftime('%Y%m%d_%H%M%S')}",
                commodity=commodity,
                exchange=commodity_config.exchange if commodity_config else "MCX",
                direction=direction,
                target_weight=float(weight),
                target_notional=abs(signed_target_quantity) * reference_price,
                target_quantity=abs(signed_target_quantity),
                current_quantity=current_quantity,
                rebalance_delta=rebalance_delta,
                reference_price=reference_price,
                signal_score=signal_score,
                confidence_score=confidence_score,
                stop_reference=self._compute_stop_reference(direction, reference_price, volatility),
                take_profit_reference=self._compute_take_profit_reference(direction, reference_price, volatility),
                rationale=self._build_rationale(weight, signal_score, confidence_score, rebalance_delta, volatility),
            )
            suggestions[commodity] = suggestion
            self.suggestion_history.append(suggestion)

        self._update_metrics(suggestions)
        return suggestions

    def get_suggestions_frame(self, commodity: Optional[str] = None) -> pd.DataFrame:
        """Return generated suggestions as a dataframe."""
        rows = []
        for suggestion in self.suggestion_history:
            if commodity and suggestion.commodity != commodity:
                continue
            rows.append(suggestion.to_dict())
        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def get_metrics(self) -> SuggestionMetrics:
        return self.metrics

    def persist_suggestions(self, suggestions: Dict[str, PositionSuggestion], name: str = "latest_position_suggestions") -> None:
        """Persist the current suggestion snapshot for research and review."""
        payload = {
            "generated_at": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            "metrics": self.metrics.to_dict(),
            "suggestions": {commodity: suggestion.to_dict() for commodity, suggestion in suggestions.items()},
        }
        self.storage.write_json(settings.storage.report_store, name, payload)

    def _update_metrics(self, suggestions: Dict[str, PositionSuggestion]) -> None:
        directions = {"long": 0, "short": 0, "flat": 0}
        total_target_notional = 0.0
        for suggestion in suggestions.values():
            directions[suggestion.direction.value] = directions.get(suggestion.direction.value, 0) + 1
            total_target_notional += suggestion.target_notional

        self.metrics.total_suggestions_generated = len(suggestions)
        self.metrics.total_target_notional = total_target_notional
        self.metrics.directions = directions
        self.metrics.last_generation_time = datetime.now(timezone.utc).replace(tzinfo=None)

    def _resolve_direction(self, signal_score: float) -> PositionDirection:
        if signal_score > 0:
            return PositionDirection.LONG
        if signal_score < 0:
            return PositionDirection.SHORT
        return PositionDirection.FLAT

    def _compute_stop_reference(
        self,
        direction: PositionDirection,
        reference_price: float,
        volatility: Optional[float],
    ) -> float:
        volatility = volatility or 0.02
        distance = reference_price * volatility * 2.0
        if direction == PositionDirection.LONG:
            return reference_price - distance
        return reference_price + distance

    def _compute_take_profit_reference(
        self,
        direction: PositionDirection,
        reference_price: float,
        volatility: Optional[float],
    ) -> float:
        volatility = volatility or 0.02
        distance = reference_price * volatility
        if direction == PositionDirection.LONG:
            return reference_price + distance
        return reference_price - distance

    def _build_rationale(
        self,
        weight: float,
        signal_score: float,
        confidence_score: float,
        rebalance_delta: int,
        volatility: Optional[float],
    ) -> List[str]:
        rationale = [
            f"Target portfolio weight {weight:.1%} based on optimizer output.",
            f"Signal score {signal_score:.3f} with confidence {confidence_score:.3f}.",
            f"Suggested position change {rebalance_delta} contracts from current holdings.",
        ]
        if volatility:
            rationale.append(f"Sizing adjusted against realized volatility estimate {volatility:.2%}.")
        rationale.append("This is an analytical suggestion only and is never transmitted to any broker or order system.")
        return rationale


position_suggestion_engine = PositionSuggestionEngine()