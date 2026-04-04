#!/usr/bin/env python3
"""
run_world_class.py — Signal generator with Shipping Integration, Factor Timing, and Portfolio Optimization.

This demonstrates the three-sprint world-class commodities signal engine:
1. Sprint 1: Automatic shipping intelligence (always-on, graceful fallback)
2. Sprint 2: Factor timing (regime-aware factor reweighting)
3. Sprint 3: Portfolio optimization (joint commodity allocation)

Output: Multi-commodity signal package with portfolio-level insights.
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from commodities_quant_engine.analytics.factor_timing import FactorTimingEngine
from commodities_quant_engine.config.settings import settings
from commodities_quant_engine.data.storage.local import LocalStorage
from commodities_quant_engine.portfolio.optimization_engine import portfolio_optimization_engine
from commodities_quant_engine.main import build_demo_price_data
from commodities_quant_engine.workflow import ResearchWorkflow


def build_demo_multi_commodity_prices(
    commodities: List[str], periods: int = 280
) -> Dict[str, pd.DataFrame]:
    """Generate realistic multi-commodity price data with cross-correlations."""
    end_date = pd.Timestamp.today().normalize()
    dates = pd.bdate_range(end=end_date, periods=periods)

    prices_dict = {}
    base_trend = np.linspace(0.0, 18.0, periods)

    # Base price levels by commodity family
    base_prices = {
        "GOLD": 65000.0,
        "SILVER": 28000.0,
        "COPPER": 9800.0,
        "ZINC": 2650.0,
        "CRUDE OIL": 85.0,
        "NATURALGAS": 3.2,
        "WHEAT": 650.0,
        "COTTON": 85.0,
    }

    for commodity in commodities:
        base = base_prices.get(commodity, 10000.0)
        
        # Add correlated shocks (energy up -> copper up, etc.)
        if commodity in ["COPPER", "ZINC", "NATURALGAS"]:
            shock = np.sin(np.linspace(0, 4 * np.pi, periods)) * 2.0  # Synchronized
        elif commodity in ["GOLD", "SILVER"]:
            shock = np.sin(np.linspace(0, 2 * np.pi, periods)) * 1.0  # Different phase
        else:
            shock = np.sin(np.linspace(np.pi, 3 * np.pi, periods)) * 1.5  # Offset phase

        close = base + base_trend * 10.0 + shock * 50.0
        
        frame = pd.DataFrame(
            {
                "open": close - 5.0,
                "high": close + 10.0,
                "low": close - 10.0,
                "close": close,
                "volume": np.linspace(50000, 100000, periods).astype(int),
                "open_interest": np.linspace(200000, 300000, periods).astype(int),
            },
            index=dates,
        )
        frame.index.name = "timestamp"
        prices_dict[commodity] = frame

    return prices_dict


def run_multi_commodity_signals(
    commodities: List[str], price_data: Dict[str, pd.DataFrame]
) -> Dict[str, Dict]:
    """
    Generate signals for multiple commodities (Sprint 1, 2, 3 integrated).

    Returns:
        Dict[commodity, signal_package]
    """
    storage = LocalStorage()
    workflow = ResearchWorkflow(storage=storage)
    factor_timing = FactorTimingEngine(storage=storage)

    signals_by_commodity = {}
    as_of_timestamp = datetime.now()

    print("\n" + "=" * 80)
    print("WORLD-CLASS COMMODITIES SIGNAL ENGINE")
    print("=" * 80)

    for commodity in commodities:
        if commodity not in price_data:
            print(f"⚠️  No price data for {commodity}")
            continue

        price_frame = price_data[commodity]
        print(f"\n▶ {commodity:<15} — Generating signal...")

        # Sprint 1: Auto-generates shipping context (integrated into workflow)
        # Sprint 2: Factor timing weights (currently using defaults, would be populated from evaluations)
        # Sprint 3: Portfolio optimization (applied after all commodities generated)

        try:
            signal_package = workflow.run_signal_cycle(
                commodity=commodity,
                price_data=price_frame,
                as_of_timestamp=as_of_timestamp,
                persist_snapshot=False,
                persist_report=False,
                # No explicit shipping_feature_vectors → auto-generated via ShippingContextBuilder (Sprint 1)
            )

            signals_by_commodity[commodity] = {
                "package": signal_package,
                "score": signal_package.suggestion.composite_score,
                "confidence": signal_package.suggestion.confidence_score,
                "regime": signal_package.suggestion.regime_label,
                "direction": signal_package.suggestion.preferred_direction,
            }

            print(
                f"  ✓ Score: {signal_package.suggestion.composite_score:+.3f} | "
                f"Confidence: {signal_package.suggestion.confidence_score:.2%} | "
                f"Regime: {signal_package.suggestion.regime_label}"
            )
        except Exception as e:
            print(f"  ✗ Error: {e}")
            continue

    # Sprint 3: Portfolio-level optimization
    print("\n" + "-" * 80)
    print("PORTFOLIO OPTIMIZATION (Sprint 3)")
    print("-" * 80)

    commodity_scores = {c: sig["score"] for c, sig in signals_by_commodity.items()}
    print(f"\nIndividual commodity scores:\n  {commodity_scores}")

    # Estimate covariance from demo prices
    price_arrays = {c: price_data[c]["close"].values for c in commodities if c in price_data}
    weights = portfolio_optimization_engine.optimize_commodity_weights(
        commodity_scores, price_history=price_arrays
    )

    print(f"\nPortfolio-optimized weights:\n  {weights}")

    sector_exposures = portfolio_optimization_engine.get_sector_exposures(weights)
    print(f"\nSector exposures:\n  {sector_exposures}")

    return {
        "signals_by_commodity": signals_by_commodity,
        "portfolio_weights": weights,
        "sector_exposures": sector_exposures,
    }


def print_summary(result: Dict) -> None:
    """Print comprehensive signal summary."""
    print("\n" + "=" * 80)
    print("SIGNAL SUMMARY")
    print("=" * 80)

    signals = result["signals_by_commodity"]
    weights = result["portfolio_weights"]

    print("\nIndividual Signals:")
    print(f"{'Commodity':<15} {'Score':>10} {'Conf':>8} {'Regime':>12} {'Direction':>10} {'Weight':>10}")
    print("-" * 80)

    for commodity in sorted(signals.keys()):
        sig = signals[commodity]
        weight = weights.get(commodity, 0.0)
        print(
            f"{commodity:<15} {sig['score']:>10.3f} {sig['confidence']:>7.1%} "
            f"{sig['regime']:>12} {sig['direction']:>10} {weight:>9.1%}"
        )

    print("\n" + "-" * 80)
    print("Sector Allocations:")
    for sector, exposure in result["sector_exposures"].items():
        print(f"  {sector:<20} {exposure:>6.1%}")

    print("\n" + "=" * 80)
    print("SPRINT INTEGRATION SUMMARY")
    print("=" * 80)
    print("""
✓ Sprint 1 (Shipping Integration): Auto-generated via ShippingContextBuilder
  → Shipping context always available (graceful fallback if no data)
  → Integrated into all signal_cycle() calls

✓ Sprint 2 (Factor Timing): Engine ready for backtested evaluations
  → Tracks per-factor Sharpe ratios by regime
  → Dynamically reweights factors (placeholder; populated from evaluations)

✓ Sprint 3 (Portfolio Optimization): Joint commodity allocation
  → Markowitz optimization with sector constraints
  → Correlation-aware position sizing
  → Cross-commodity risk budgeting
    """)


def main() -> None:
    """Main world-class demo."""
    # Featured commodities across all sectors
    commodities = [
        "GOLD",     # Bullion
        "COPPER",   # Base metals
        "CRUDEOIL", # Energy
        "WHEAT",    # Agri
    ]

    print("\nBuilding multi-commodity price data...")
    price_data = build_demo_multi_commodity_prices(commodities, periods=280)

    print("Running world-class signal generation...")
    result = run_multi_commodity_signals(commodities, price_data)

    print_summary(result)


if __name__ == "__main__":
    main()
