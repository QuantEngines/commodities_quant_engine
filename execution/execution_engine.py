"""
ExecutionEngine — Orchestrates order generation from signal workflow.

Integration point:
  Portfolio-Level Signals → Order Generation → Execution Routing → Audit

This is the final stage of the signal generation pipeline.
Orders are recommendations for user review before placement.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ..config.settings import settings
from ..data.storage.local import LocalStorage
from .order_generator import (
    Order,
    OrderDirection,
    adaptive_order_router,
    order_audit,
    signal_to_order_converter,
)


@dataclass
class ExecutionMetrics:
    """Performance metrics for execution layer."""

    total_orders_generated: int = 0
    total_orders_placed: int = 0
    avg_slippage_bps: float = 0.0
    positions_by_side: Dict[str, int] = field(default_factory=dict)
    orders_by_algo: Dict[str, int] = field(default_factory=dict)
    turnover_notional: float = 0.0
    last_execution_time: Optional[datetime] = None

    def to_dict(self) -> Dict:
        """Serialize metrics."""
        return {
            "total_orders_generated": self.total_orders_generated,
            "total_orders_placed": self.total_orders_placed,
            "avg_slippage_bps": self.avg_slippage_bps,
            "positions_by_side": self.positions_by_side,
            "orders_by_algo": self.orders_by_algo,
            "turnover_notional": self.turnover_notional,
            "last_execution_time": (
                self.last_execution_time.isoformat()
                if self.last_execution_time
                else None
            ),
        }


class ExecutionEngine:
    """
    End-to-end orchestration: Signals → Orders → Routing → Audit.

    Public methods:
    - execute_portfolio_signals()  — Main entry point
    - get_order_recommendations()  — Retrieve pending orders
    - get_execution_metrics()      — Performance summary
    - clear_positions()            — Flatten all positions
    """

    def __init__(self, storage: Optional[LocalStorage] = None):
        self.storage = storage or LocalStorage()
        self.metrics = ExecutionMetrics()
        self.current_positions: Dict[str, int] = {}  # commodity -> net qty
        self.execution_history: List[Order] = []

    def execute_portfolio_signals(
        self,
        portfolio_weights: Dict[str, float],  # commodity -> weight [0, 1]
        signal_data: Dict,  # From CompositeDecisionEngine
        price_data: Dict[str, float],  # commodity -> price
        market_volatility: Optional[Dict[str, float]] = None,  # commodity -> vol
        intraday_profiles: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> Dict[str, Order]:
        """
        Core execution orchestration.

        Args:
            portfolio_weights: Portfolio allocation (optimized weights)
            signal_data: Full signal output including scores, confidence, regime
            price_data: Current market prices
            market_volatility: Optional realized vol by commodity
            intraday_profiles: Optional hourly volume data

        Returns:
            Dict mapping commodity -> Order object (only commodities with orders)
        """
        orders = {}
        total_notional = 0

        # For each commodity in portfolio
        for commodity, weight in portfolio_weights.items():
            if weight == 0:
                # Skip zero-weight positions
                continue

            # Extract signal info
            commodity_signal = signal_data.get(commodity, {})
            signal_score = commodity_signal.get("composite_score", 0.0)
            confidence = commodity_signal.get("confidence_score", 0.5)

            # Skip if no signal
            if abs(signal_score) < 0.05:
                continue

            # Get price
            price = price_data.get(commodity)
            if price is None or price <= 0:
                print(f"Warning: No valid price for {commodity}")
                continue

            # Get volatility
            vol = market_volatility.get(commodity) if market_volatility else None

            # Get existing position
            existing_qty = self.current_positions.get(commodity, 0)

            # Convert signal to order
            order = signal_to_order_converter.convert_signal_to_order(
                commodity=commodity,
                signal_score=signal_score,
                confidence=confidence,
                current_price=price,
                market_volatility=vol,
                existing_position=existing_qty,
            )

            if order is None:
                continue

            # Route order to execution algorithm
            execution_plan = adaptive_order_router.route_order(
                order,
                intraday_volume_profile=intraday_profiles.get(commodity)
                if intraday_profiles
                else None,
            )

            # Attach execution plan to order
            order.notes = str(execution_plan)

            # Add to results
            orders[commodity] = order
            total_notional += order.quantity * price

            # Log order
            order_audit.log_order(order)
            self.execution_history.append(order)

            # Update metrics
            self.metrics.total_orders_generated += 1
            self.metrics.orders_by_algo[order.routing_algo] = (
                self.metrics.orders_by_algo.get(order.routing_algo, 0) + 1
            )
            self.metrics.turnover_notional += order.quantity * price

        # Update metrics
        self.metrics.last_execution_time = datetime.now()
        if orders:
            self._update_position_metrics(orders)

        return orders

    def _update_position_metrics(self, orders: Dict[str, Order]) -> None:
        """Update position and side metrics."""
        sides = {"long": 0, "short": 0}
        for order in orders.values():
            if order.direction == OrderDirection.LONG:
                sides["long"] += order.quantity
                self.current_positions[order.commodity] = (
                    self.current_positions.get(order.commodity, 0) + order.quantity
                )
            elif order.direction == OrderDirection.SHORT:
                sides["short"] += abs(order.quantity)
                self.current_positions[order.commodity] = (
                    self.current_positions.get(order.commodity, 0) - order.quantity
                )

        self.metrics.positions_by_side = sides

    def get_order_recommendations(
        self, status: str = "pending", commodity: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get pending orders for user review.

        Args:
            status: Filter by order status (default: pending)
            commodity: Optional filter by commodity

        Returns:
            DataFrame of recommended orders
        """
        filtered_orders = []
        for order in self.execution_history:
            if order.status != status:
                continue
            if commodity and order.commodity != commodity:
                continue
            filtered_orders.append(order.to_dict())

        return pd.DataFrame(filtered_orders) if filtered_orders else pd.DataFrame()

    def get_execution_metrics(self) -> ExecutionMetrics:
        """Return current execution metrics."""
        return self.metrics

    def clear_positions(self, reason: str = "manual liquidation") -> Dict[str, Order]:
        """
        Generate liquidation orders to flatten all positions.

        Args:
            reason: Reason for liquidation

        Returns:
            Dict mapping commodity -> liquidation Order
        """
        liquidation_orders = {}

        for commodity, qty in self.current_positions.items():
            if qty == 0:
                continue

            # Opposite direction to flatten
            direction = (
                OrderDirection.SHORT if qty > 0 else OrderDirection.LONG
            )

            # Create liquidation order
            order = Order(
                order_id=f"LIQ_{commodity}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                commodity=commodity,
                exchange=settings.commodities.get(commodity, {}).get("exchange", "MCX"),
                direction=direction,
                quantity=abs(qty),
                order_type="market",
                entry_price=0.0,  # Will be filled at market
                urgency="urgent",
                routing_algo="MARKET",
                notes=f"Liquidation: {reason}",
                status="pending",
            )

            liquidation_orders[commodity] = order
            order_audit.log_order(order)

        return liquidation_orders

    def simulate_execution(
        self,
        orders: Dict[str, Order],
        simulated_prices: Dict[str, float],
    ) -> Tuple[float, float]:
        """
        Simulate execution-to-completion P&L.

        Args:
            orders: Orders to simulate
            simulated_prices: Post-execution prices

        Returns:
            (total_pnl, total_notional)
        """
        total_pnl = 0.0
        total_notional = 0.0

        for commodity, order in orders.items():
            sim_price = simulated_prices.get(commodity, order.entry_price)
            notional = order.quantity * order.entry_price

            if order.direction == OrderDirection.LONG:
                pnl = order.quantity * (sim_price - order.entry_price)
            else:
                pnl = order.quantity * (order.entry_price - sim_price)

            total_pnl += pnl
            total_notional += abs(notional)

        return total_pnl, total_notional

    def persist_execution_state(self) -> None:
        """Persist execution state and metrics to storage."""
        try:
            # Persist metrics
            self.storage.write_json(
                settings.storage.parameter_store,
                "execution_metrics",
                self.metrics.to_dict(),
            )

            # Persist positions
            self.storage.write_json(
                settings.storage.parameter_store,
                "current_positions",
                self.current_positions,
            )

            print("Execution state persisted.")
        except Exception as e:
            print(f"Warning: Could not persist execution state: {e}")

    def load_execution_state(self) -> None:
        """Load execution state from storage."""
        try:
            metrics_dict = self.storage.read_json(
                settings.storage.parameter_store, "execution_metrics"
            )
            if metrics_dict:
                self.metrics = ExecutionMetrics(**metrics_dict)

            positions = self.storage.read_json(
                settings.storage.parameter_store, "current_positions"
            )
            if positions:
                self.current_positions = positions

            print("Execution state loaded.")
        except Exception as e:
            print(f"Warning: Could not load execution state: {e}")


# Singleton instance
execution_engine = ExecutionEngine()
