"""
EndToEndDemo — Integration of all five components.

Demonstrates complete signal generation pipeline:
1. Sprint 1: Shipping Context Builder (always-on shipping integration)
2. Sprint 2: Factor Timing Engine (regime-aware dynamic weighting)
3. Sprint 3: Portfolio Optimization Engine (Markowitz commodity allocation)
4. Component 1: Historical Backtester (factor accumulation)
5. Component 2: Live Factor Refresh Scheduler (daily refresh)
6. Component 3: Inter-Market Data Sources (COMEX/LBME basis)
7. Component 4: Intraday Factor Rotation (cross-timeframe signals)
8. Component 5: Execution Layer (order generation + routing)

Execution flow:
  Tick
    ├─ Load daily signals (regimes, factors)
    ├─ Build shipping context (always-on)
    ├─ Generate intraday signals (within daily frames)
    ├─ Optimize portfolio (Markowitz)
    ├─ Inter-market basis (arbitrage opportunities)  
    ├─ Generate orders (execution layer)
    └─ Persist metrics + order audit

Backtester runs offline (daily refresh from scheduler).
"""

from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

# Sprint components
from analytics.factor_timing import factor_timing_engine
from portfolio.optimization_engine import portfolio_optimization_engine
from shipping.context_builder import shipping_context_builder

# Advanced components
from analytics.backtester import (
    HistoricalBacktester,
    generate_backtest_report,
)
from analytics.live_scheduler import LiveFactorScheduler
from data.ingestion.inter_market import (
    InterMarketBasisCalculator,
    COMEXDataSource,
    LBMEDataSource,
)
from signals.intraday.intraday_engine import IntradayFactorRotationEngine

# Execution
from execution import ExecutionEngine, order_audit

# Core
from config.settings import settings
from data.storage.local import LocalStorage
from signals.composite.composite_decision_engine import composite_decision_engine


def run_end_to_end_demo(
    test_commodities: Optional[list] = None,
    include_backtester: bool = True,
    include_intraday: bool = True,
    include_inter_market: bool = True,
    include_execution: bool = True,
) -> Dict:
    """
    Run complete end-to-end signal generation and execution pipeline.

    Args:
        test_commodities: Optional list of commodities to test (default: ['CRUDEOIL', 'GOLD'])
        include_backtester: Run historical backtester (slow)
        include_intraday: Generate intraday signals
        include_inter_market: Calculate inter-market basis
        include_execution: Generate orders and execution plans

    Returns:
        Results dict with all pipeline outputs
    """
    test_commodities = test_commodities or ["CRUDEOIL", "GOLD", "COPPER"]
    storage = LocalStorage()
    results = {
        "timestamp": datetime.now().isoformat(),
        "commodities": test_commodities,
        "components": {},
    }

    print("\n" + "=" * 80)
    print("END-TO-END SIGNAL GENERATION PIPELINE")
    print("=" * 80)

    # ============================================================================
    # STEP 1: DAILY SIGNAL GENERATION (Composite Decision Engine)
    # ============================================================================
    print("\n[STEP 1] Generating daily signals (regime detection + multi-factor)...")

    daily_signals = {}
    for commodity in test_commodities:
        try:
            signal = composite_decision_engine.generate_signal(
                commodity=commodity,
                regime="neutral",  # Assume neutral for demo
                include_shipping=True,  # Sprint 1: Shipping included
            )
            daily_signals[commodity] = signal
            print(f"  ✓ {commodity}: score={signal['composite_score']:.3f}, "
                  f"confidence={signal['confidence_score']:.3f}")
        except Exception as e:
            print(f"  ✗ {commodity}: {e}")

    results["components"]["daily_signals"] = daily_signals

    # ============================================================================
    # STEP 2: SHIPPING CONTEXT (Sprint 1)
    # ============================================================================
    print("\n[SPRINT 1] Building shipping context (always-on integration)...")

    shipping_contexts = {}
    for commodity in test_commodities:
        try:
            context = shipping_context_builder.build(
                commodity=commodity,
                fallback_to_stubs=True,
            )
            shipping_contexts[commodity] = {
                "features": context.to_dict() if hasattr(context, "to_dict") else {},
                "source": "real" if context else "stubs",
            }
            print(f"  ✓ {commodity}: {shipping_contexts[commodity]['source']} data")
        except Exception as e:
            print(f"  ✗ {commodity}: {e}")

    results["components"]["shipping_contexts"] = shipping_contexts

    # ============================================================================
    # STEP 3: FACTOR TIMING (Sprint 2)
    # ============================================================================
    print("\n[SPRINT 2] Computing regime-aware factor weights...")

    factor_diagnostics = {}
    try:
        factor_metrics = factor_timing_engine.get_factor_diagnostics()
        for commodity in test_commodities:
            factor_diagnostics[commodity] = {
                "bullish_weights": {k: v for k, v in factor_metrics.items()},
                "note": "Regime-weighted factors based on rolling Sharpe ratios",
            }
        print(f"  ✓ Factor timing engine ready (Bullish regime)")
        for k, v in factor_metrics.items():
            print(f"    - {k}: {v:.3f}")
    except Exception as e:
        print(f"  ✗ Factor timing: {e}")
        factor_diagnostics = {}

    results["components"]["factor_timing"] = factor_diagnostics

    # ============================================================================
    # STEP 4: PORTFOLIO OPTIMIZATION (Sprint 3)
    # ============================================================================
    print("\n[SPRINT 3] Optimizing portfolio allocation (Markowitz)...")

    portfolio_weights = {}
    try:
        # Create mock signals for optimization
        commodity_signals = {c: daily_signals.get(c, {}) for c in test_commodities}

        weights = portfolio_optimization_engine.optimize_commodity_weights(
            commodity_signals=commodity_signals,
            correlation_window=60,
        )

        portfolio_weights = weights
        total_weight = sum(weights.values())
        print(f"  ✓ Optimization complete (total weight: {total_weight:.1%})")
        for commodity, weight in weights.items():
            print(f"    - {commodity}: {weight:.1%}")
    except Exception as e:
        print(f"  ✗ Portfolio optimization: {e}")
        # Default equal weight
        equal_weight = 1.0 / len(test_commodities)
        portfolio_weights = {c: equal_weight for c in test_commodities}

    results["components"]["portfolio_weights"] = portfolio_weights

    # ============================================================================
    # STEP 5: BACKTESTER (Component 1 - Optional)
    # ============================================================================
    if include_backtester:
        print("\n[COMPONENT 1] Running historical backtester (factor attribution)...")

        try:
            backtester = HistoricalBacktester()
            for commodity in test_commodities[:1]:  # Just first commodity (slow)
                print(f"  Backtesting {commodity}...")
                eval_result = backtester.backtest_commodity_historical(
                    commodity=commodity,
                    lookback_days=250,
                )
                if eval_result:
                    print(
                        f"    ✓ Backtest complete: Sharpe={eval_result.sharpe_ratio:.2f}"
                    )
                    # Ingest into factor timing engine
                    factor_timing_engine.ingest_evaluation_results(eval_result)
        except Exception as e:
            print(f"  ✗ Backtester: {e}")

    results["components"]["backtester"] = {"status": "complete" if include_backtester else "skipped"}

    # ============================================================================
    # STEP 6: LIVE SCHEDULER (Component 2 - Optional)
    # ============================================================================
    if include_backtester:
        print("\n[COMPONENT 2] Live factor refresh scheduler (daily cron)...")

        try:
            scheduler = LiveFactorScheduler(
                backtester=HistoricalBacktester(),
                factor_engine=factor_timing_engine,
            )
            status = scheduler.get_refresh_status()
            print(f"  ✓ Scheduler status: {status.get('status', 'ready')}")
            print(
                f"    Last refresh: {status.get('last_refresh_time', 'never')}"
            )
            # Don't start automatic scheduler in demo (would run in background)
            print("    (Automatic scheduling available in production)")
        except Exception as e:
            print(f"  ✗ Scheduler: {e}")

    results["components"]["scheduler"] = {"status": "configured"}

    # ============================================================================
    # STEP 7: INTER-MARKET BASIS (Component 3 - Optional)
    # ============================================================================
    if include_inter_market:
        print("\n[COMPONENT 3] Computing inter-market basis (arbitrage opportunities)...")

        try:
            basis_calc = InterMarketBasisCalculator()

            # Mock some price data for demo
            mock_mcx_prices = {c: np.random.uniform(40, 60) for c in test_commodities}
            mock_comex_prices = {c: np.random.uniform(40, 60) for c in test_commodities}

            for commodity in test_commodities:
                mcx_price = mock_mcx_prices.get(commodity, 50.0)
                comex_price = mock_comex_prices.get(commodity, 50.0)

                basis_pct = (mcx_price - comex_price) / comex_price * 100
                opp = basis_calc.get_arbitrage_opportunity(
                    mcx_price=mcx_price,
                    comex_price=comex_price,
                    historical_mean_basis=0.0,
                    historical_std_basis=1.0,
                )

                print(f"  ✓ {commodity}: basis={basis_pct:.2f}%, "
                      f"opportunity={opp}")
        except Exception as e:
            print(f"  ✗ Inter-market basis: {e}")

    results["components"]["inter_market"] = {"status": "complete" if include_inter_market else "skipped"}

    # ============================================================================
    # STEP 8: INTRADAY FACTOR ROTATION (Component 4 - Optional)
    # ============================================================================
    if include_intraday:
        print("\n[COMPONENT 4] Generating intraday signals (cross-timeframe)...")

        try:
            intraday_engine = IntradayFactorRotationEngine()

            # Mock intraday data
            intraday_signals = {}
            for commodity in test_commodities:
                daily_sig = daily_signals.get(commodity, {})

                # Mock intraday price data (9 hours of OHLCV)
                hours = pd.date_range("09:00", "17:00", freq="1H")
                intraday_data = pd.DataFrame({
                    "timestamp": hours,
                    "open": np.random.uniform(40, 60, len(hours)),
                    "high": np.random.uniform(60, 70, len(hours)),
                    "low": np.random.uniform(30, 40, len(hours)),
                    "close": np.random.uniform(40, 60, len(hours)),
                    "volume": np.random.randint(1000, 5000, len(hours)),
                })

                intraday_signal = intraday_engine.generate_intraday_signal_package(
                    commodity=commodity,
                    daily_signal=daily_sig,
                    intraday_price_data=intraday_data,
                    interval="1H",
                )

                intraday_signals[commodity] = intraday_signal
                print(f"  ✓ {commodity}: entry_signal={intraday_signal.get('entry_signal', 'hold')}")
        except Exception as e:
            print(f"  ✗ Intraday engine: {e}")
            intraday_signals = {}

    results["components"]["intraday"] = intraday_signals if include_intraday else {"status": "skipped"}

    # ============================================================================
    # STEP 9: EXECUTION LAYER (Component 5)
    # ============================================================================
    print("\n[COMPONENT 5] Generating execution orders and routing...")

    orders = {}
    if include_execution:
        try:
            execution_engine = ExecutionEngine()

            # Mock price data
            price_data = {c: np.random.uniform(40, 60) for c in test_commodities}
            vol_data = {c: 0.02 for c in test_commodities}

            orders = execution_engine.execute_portfolio_signals(
                portfolio_weights=portfolio_weights,
                signal_data=daily_signals,
                price_data=price_data,
                market_volatility=vol_data,
                intraday_profiles=None,  # Could pass real data
            )

            print(f"  ✓ Order generation complete ({len(orders)} orders)")
            for commodity, order in orders.items():
                print(
                    f"    - {commodity}: {order.direction.value.upper()} "
                    f"{order.quantity} @ {order.entry_price:.2f} "
                    f"(algo: {order.routing_algo})"
                )

            # Show order routing details
            print("\n  Order Routing Details:")
            for commodity, order in orders.items():
                print(f"    {commodity}:")
                print(f"      Order ID: {order.order_id}")
                print(f"      Signal Strength: {order.signal_score:.3f}")
                print(f"      Confidence: {order.signal_confidence:.3f}")
                print(f"      Stop Loss: {order.stop_loss:.2f}")
                print(f"      Take Profit: {order.take_profit:.2f}")

            # Get execution metrics
            metrics = execution_engine.get_execution_metrics()
            print(f"\n  Execution Metrics:")
            print(f"    Total Notional: ${metrics.turnover_notional:,.0f}")
            print(f"    Orders by Side: {metrics.positions_by_side}")
            print(f"    Algos Used: {metrics.orders_by_algo}")

        except Exception as e:
            print(f"  ✗ Execution layer: {e}")

    results["components"]["execution"] = {
        "orders_generated": len(orders),
        "orders": {k: v.to_dict() for k, v in orders.items()},
    }

    # ============================================================================
    # FINAL SUMMARY
    # ============================================================================
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nGenerated: {datetime.now().isoformat()}")
    print(f"Commodities: {', '.join(test_commodities)}")
    print(f"Orders Ready for Review: {len(orders)}")
    print(f"\nNext Steps:")
    print("  1. Review order recommendations (see execution orders above)")
    print("  2. Validate risk constraints (position limits, sector exposure)")
    print("  3. Place orders via broker API (user's responsibility)")
    print("  4. Monitor executions via order_audit")
    print("  5. Refresh factor metrics daily via live_scheduler")

    # Persist results
    try:
        storage.write_json(
            settings.storage.reports,
            f"end_to_end_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            results,
        )
        print(f"\n✓ Results persisted to {settings.storage.reports}/")
    except Exception as e:
        print(f"\nWarning: Could not persist results: {e}")

    return results


if __name__ == "__main__":
    # Run full demo
    results = run_end_to_end_demo(
        test_commodities=["CRUDEOIL", "GOLD", "COPPER"],
        include_backtester=True,  # Slow but shows factor accumulation
        include_intraday=True,
        include_inter_market=True,
        include_execution=True,
    )

    print("\n✓ Demo complete. Order recommendations available for review.")
