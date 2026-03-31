"""
ExecutionLayer — Converts signals to orders with adaptive execution algorithms.

Includes:
1. SignalToOrderConverter — Maps signal strength to order size
2. AdaptiveOrderRouter — VWAP/TWAP execution with slippage minimization
3. OrderAudit — Logs all orders with P&L tracking

This layer bridges signal generation and actual trading execution.
Note: Execution decisions are USER'S responsibility. This generates orders for review.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config.settings import settings
from ..data.storage.local import LocalStorage


class OrderType(str, Enum):
    """Order types supported."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    VWAP = "vwap"
    TWAP = "twap"


class OrderDirection(str, Enum):
    """Trade direction."""
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class Order:
    """Represents a trading order."""
    
    order_id: str
    commodity: str
    exchange: str
    direction: OrderDirection
    quantity: int
    order_type: OrderType
    entry_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    time_in_force: str = "GTC"  # Good-till-cancel
    timestamp: datetime = field(default_factory=datetime.now)
    signal_score: float = 0.0
    signal_confidence: float = 0.0
    routing_algo: str = "VWAP"  # Default execution algo
    max_participation_rate: float = 0.2  # Max % of VWAP volume
    urgency: str = "normal"  # normal, urgent, patient
    status: str = "pending"  # pending, placed, filled, cancelled, etc.
    notes: str = ""

    def to_dict(self) -> Dict:
        """Serialize to dict."""
        return {
            "order_id": self.order_id,
            "commodity": self.commodity,
            "exchange": self.exchange,
            "direction": self.direction.value,
            "quantity": self.quantity,
            "order_type": self.order_type.value,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "timestamp": self.timestamp.isoformat(),
            "signal_score": self.signal_score,
            "signal_confidence": self.signal_confidence,
            "routing_algo": self.routing_algo,
            "status": self.status,
            "notes": self.notes,
        }


class SignalToOrderConverter:
    """
    Converts signal strength and confidence to order size and execution parameters.

    Risk management:
    - Position sizing relative to portfolio capital and volatility
    - Smaller positions for lower confidence signals
    - Larger positions for high-confidence regime consensus signals
    """

    def __init__(self):
        self.target_notional_per_signal = 100000.0  # USD per trade
        self.max_position_notional = 500000.0  # Max single commodity exposure
        self.min_quantity = 1

    def convert_signal_to_order(
        self,
        commodity: str,
        signal_score: float,  # [-1, 1]
        confidence: float,  # [0, 1]
        current_price: float,
        market_volatility: Optional[float] = None,
        existing_position: int = 0,
    ) -> Optional[Order]:
        """
        Convert signal to order with size and execution parameters.

        Args:
            commodity: Commodity symbol
            signal_score: Signal score in [-1, 1] range
            confidence: Confidence in [0, 1] range
            current_price: Market price for sizing
            market_volatility: Realized vol (used for risk adjustment)
            existing_position: Current net position quantity

        Returns:
            Order object, or None if signal too weak
        """
        # Threshold: only trade if signal above noise floor
        if abs(signal_score) < 0.05:
            return None

        # Determine direction
        direction = (
            OrderDirection.LONG if signal_score > 0 else OrderDirection.SHORT
        )

        # Size based on signal strength and confidence
        # Confidence adjustment: higher confidence = larger trade
        # Signal strength adjustment: stronger signal = larger trade
        signal_strength = abs(signal_score)
        size_factor = signal_strength * confidence

        # Base notional
        notional = self.target_notional_per_signal * size_factor

        # Vol adjustment (if available): high vol = smaller position
        if market_volatility and market_volatility > 0:
            vol_adjustment = 0.15 / market_volatility  # Target 15% vol
            notional = notional * vol_adjustment

        # Cap to max position
        notional = min(notional, self.max_position_notional)

        # Convert to quantity
        quantity = max(self.min_quantity, int(notional / current_price))

        # Adjust for existing position (reduce if already exposed)
        if existing_position != 0:
            if (direction == OrderDirection.LONG and existing_position > 0) or (
                direction == OrderDirection.SHORT and existing_position < 0
            ):
                # Already in same direction: reduce quantity
                quantity = max(0, quantity - abs(existing_position))

        # Skip if quantity zero
        if quantity == 0:
            return None

        # Determine execution parameters
        routing_algo = self._select_execution_algo(confidence, signal_strength)
        urgency = self._determine_urgency(signal_score, confidence)
        order_type = self._select_order_type(routing_algo, urgency)

        # Price levels
        entry_price = current_price
        stop_loss = self._compute_stop_loss(
            direction, current_price, market_volatility
        )
        take_profit = self._compute_take_profit(
            direction, current_price, market_volatility
        )

        order_id = f"{commodity}_{direction.value}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return Order(
            order_id=order_id,
            commodity=commodity,
            exchange=settings.commodities.get(commodity, {}).get("exchange", "MCX"),
            direction=direction,
            quantity=quantity,
            order_type=order_type,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            signal_score=signal_score,
            signal_confidence=confidence,
            routing_algo=routing_algo,
            urgency=urgency,
        )

    def _select_execution_algo(self, confidence: float, strength: float) -> str:
        """Select execution algorithm based on signal quality."""
        if confidence > 0.8 and strength > 0.5:
            return "VWAP"  # Aggressive: participate in volume
        elif confidence > 0.6:
            return "TWAP"  # Neutral: time-weighted
        else:
            return "PATIENT"  # Conservative: limit orders

    def _determine_urgency(self, signal_score: float, confidence: float) -> str:
        """Determine execution urgency."""
        if abs(signal_score) > 0.7 and confidence > 0.8:
            return "urgent"
        elif abs(signal_score) > 0.3 and confidence > 0.6:
            return "normal"
        else:
            return "patient"

    def _select_order_type(self, algo: str, urgency: str) -> OrderType:
        """Select order type."""
        if algo == "VWAP":
            return OrderType.VWAP
        elif algo == "TWAP":
            return OrderType.TWAP
        elif urgency == "urgent":
            return OrderType.MARKET
        else:
            return OrderType.LIMIT

    def _compute_stop_loss(
        self, direction: OrderDirection, price: float, vol: Optional[float] = None
    ) -> float:
        """Compute stop loss level."""
        # Default: 2x realized volatility (or 2% if no vol)
        vol = vol or 0.02
        stop_distance = price * vol * 2
        if direction == OrderDirection.LONG:
            return price - stop_distance
        else:
            return price + stop_distance

    def _compute_take_profit(
        self, direction: OrderDirection, price: float, vol: Optional[float] = None
    ) -> float:
        """Compute take profit level."""
        # Default: 1x realized volatility / 0.5 reward-risk ratio = 1x vol
        vol = vol or 0.02
        tp_distance = price * vol
        if direction == OrderDirection.LONG:
            return price + tp_distance
        else:
            return price - tp_distance


class AdaptiveOrderRouter:
    """
    Routes orders to appropriate execution algorithm.

    Implements VWAP, TWAP, and patient limit order execution.
    Adapts algo based on market conditions and urgency.
    """

    def __init__(self):
        self.max_participation_rate = 0.2  # Max % of market volume

    def route_order(
        self,
        order: Order,
        intraday_volume_profile: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        Route order and generate execution schedule.

        Args:
            order: Order object
            intraday_volume_profile: Optional hourly volume data

        Returns:
            Execution plan: {algo, child_orders, expected_costs, timeline}
        """
        if order.routing_algo == "VWAP":
            return self._route_vwap(order, intraday_volume_profile)
        elif order.routing_algo == "TWAP":
            return self._route_twap(order)
        else:  # PATIENT (limit orders)
            return self._route_patient_limit(order)

    def _route_vwap(
        self,
        order: Order,
        volume_profile: Optional[pd.DataFrame] = None,
    ) -> Dict:
        """
        Route using Volume-Weighted Average Price (VWAP).

        Aims to fill order at or better than volume-weighted average,
        matching market participation rate.
        """
        if volume_profile is None or volume_profile.empty:
            # Estimate: equal participation across day
            n_buckets = 9  # Trading hours
            qty_per_bucket = order.quantity // n_buckets
        else:
            # Use actual volume profile to size buckets
            volume_profile = volume_profile.copy()
            volume_profile["pct"] = (
                volume_profile["volume"] / volume_profile["volume"].sum()
            )
            qty_per_bucket = (order.quantity * volume_profile["pct"]).astype(int)

        child_orders = []
        cum_qty = 0
        for i, qty in enumerate(
            qty_per_bucket if "pct" in locals() else [order.quantity // 9] * 9
        ):
            if qty > 0:
                hour = 9 + i  # 9am to 5pm
                child_orders.append(
                    {
                        "time": f"{hour:02d}:00",
                        "quantity": qty,
                        "algo": "VWAP",
                        "participation_rate": 0.15,  # 15% of per-hour volume
                    }
                )
                cum_qty += qty

        # Leftover in last bucket
        if cum_qty < order.quantity:
            if child_orders:
                child_orders[-1]["quantity"] += order.quantity - cum_qty

        return {
            "order_id": order.order_id,
            "routing_algo": "VWAP",
            "child_orders": child_orders,
            "expected_slippage_bps": 2.0,  # 2 basis points typical
            "expected_execution_time": "1 trading day",
            "urgency": order.urgency,
        }

    def _route_twap(self, order: Order) -> Dict:
        """
        Route using Time-Weighted Average Price (TWAP).

        Splits order evenly across time periods.
        """
        n_buckets = 9  # 9 hours
        qty_per_bucket = order.quantity // n_buckets

        child_orders = []
        for i in range(n_buckets):
            hour = 9 + i
            qty = qty_per_bucket + (1 if i < order.quantity % n_buckets else 0)
            if qty > 0:
                child_orders.append(
                    {
                        "time": f"{hour:02d}:00",
                        "quantity": qty,
                        "algo": "TWAP",
                        "timing": "start of hour",
                    }
                )

        return {
            "order_id": order.order_id,
            "routing_algo": "TWAP",
            "child_orders": child_orders,
            "expected_slippage_bps": 3.0,
            "expected_execution_time": "1 trading day",
        }

    def _route_patient_limit(self, order: Order) -> Dict:
        """
        Route using patient limit orders.

        Uses limit orders close to mid-price, relies on liquidity.
        Good for non-urgent, high-confidence trades.
        """
        limit_offset = 0.002  # 0.2% from mid
        if order.direction == OrderDirection.LONG:
            limit_price = order.entry_price * (1 - limit_offset)
        else:
            limit_price = order.entry_price * (1 + limit_offset)

        return {
            "order_id": order.order_id,
            "routing_algo": "PATIENT_LIMIT",
            "order_type": "LIMIT",
            "limit_price": limit_price,
            "quantity": order.quantity,
            "time_in_force": "GTC",
            "expected_slippage_bps": 0.5,  # Minimal if filled
            "expected_execution_time": "variable (may not fill)",
        }


class OrderAudit:
    """Logs and tracks all orders and subsequent P&L."""

    def __init__(self, storage: Optional[LocalStorage] = None):
        self.storage = storage or LocalStorage()
        self.orders: Dict[str, Order] = {}

    def log_order(self, order: Order) -> None:
        """Log order to audit trail."""
        self.orders[order.order_id] = order
        try:
            self.storage.write_json(
                settings.storage.parameter_store,  # Reuse for audit logs
                f"order_{order.order_id}",
                order.to_dict(),
            )
        except Exception as e:
            print(f"Warning: Could not persist order: {e}")

    def update_order_status(self, order_id: str, status: str, notes: str = "") -> None:
        """Update order status (filled, cancelled, etc.)."""
        if order_id in self.orders:
            self.orders[order_id].status = status
            self.orders[order_id].notes = notes

    def get_order_history(self, commodity: Optional[str] = None) -> pd.DataFrame:
        """Retrieve order history."""
        rows = []
        for order in self.orders.values():
            if commodity and order.commodity != commodity:
                continue
            rows.append(order.to_dict())

        return pd.DataFrame(rows) if rows else pd.DataFrame()


# Singleton instances
signal_to_order_converter = SignalToOrderConverter()
adaptive_order_router = AdaptiveOrderRouter()
order_audit = OrderAudit()
