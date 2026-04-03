"""End-to-end demo for the suggestion-only research pipeline."""

from dataclasses import asdict
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

from commodities_quant_engine.analytics.backtester import HistoricalBacktester
from commodities_quant_engine.analytics.factor_timing import factor_timing_engine
from commodities_quant_engine.analytics.live_scheduler import LiveFactorScheduler
from commodities_quant_engine.config.settings import settings
from commodities_quant_engine.data.ingestion.inter_market import InterMarketBasisCalculator
from commodities_quant_engine.data.storage.local import LocalStorage
from commodities_quant_engine.portfolio.optimization_engine import portfolio_optimization_engine
from commodities_quant_engine.portfolio.position_suggestion import PositionSuggestionEngine
from commodities_quant_engine.shipping.context_builder import shipping_context_builder
from commodities_quant_engine.signals.composite.composite_decision import CompositeDecisionEngine
from commodities_quant_engine.signals.intraday.intraday_engine import IntradayFactorRotationEngine


def _make_mock_daily_frame(seed: int, periods: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.utcnow().normalize(), periods=periods, freq="B")
    returns = rng.normal(loc=0.0004, scale=0.012, size=periods)
    close = 100.0 * np.cumprod(1.0 + returns)
    open_ = close * (1.0 + rng.normal(0.0, 0.0025, size=periods))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.001, 0.01, size=periods))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.001, 0.01, size=periods))
    volume = rng.integers(5_000, 50_000, size=periods)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )


def _make_mock_intraday_frame(seed: int, periods: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2026-01-01 09:00", periods=periods, freq="1h")
    close = 50.0 + np.cumsum(rng.normal(0.0, 0.8, size=periods))
    open_ = close + rng.normal(0.0, 0.3, size=periods)
    high = np.maximum(open_, close) + rng.uniform(0.1, 0.8, size=periods)
    low = np.minimum(open_, close) - rng.uniform(0.1, 0.8, size=periods)
    volume = rng.integers(1_000, 5_000, size=periods)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def run_end_to_end_demo(
    test_commodities: Optional[list] = None,
    include_backtester: bool = True,
    include_intraday: bool = True,
    include_inter_market: bool = True,
    include_position_suggestions: bool = True,
    portfolio_budget: float = 1_000_000.0,
) -> Dict:
    test_commodities = test_commodities or ["CRUDEOIL", "GOLD", "COPPER"]
    storage = LocalStorage()
    composite_engine = CompositeDecisionEngine()
    suggestion_engine = PositionSuggestionEngine(storage=storage)
    results = {
        "timestamp": datetime.now().isoformat(),
        "commodities": test_commodities,
        "components": {},
    }

    daily_price_data = {
        commodity: _make_mock_daily_frame(seed=100 + idx)
        for idx, commodity in enumerate(test_commodities)
    }
    signal_snapshots: Dict[str, list] = {}

    print("\n" + "=" * 80)
    print("END-TO-END SIGNAL GENERATION PIPELINE")
    print("=" * 80)

    print("\n[STEP 1] Generating daily signals (regime detection + multi-factor)...")
    daily_signals = {}
    for idx, commodity in enumerate(test_commodities):
        try:
            price_frame = daily_price_data[commodity]
            as_of_timestamp = price_frame.index[-21].to_pydatetime()
            shipping_vectors = shipping_context_builder.build(
                commodity=commodity,
                as_of_timestamp=as_of_timestamp,
            )
            signal_package = composite_engine.generate_signal_package(
                data=price_frame.loc[:as_of_timestamp],
                commodity=commodity,
                shipping_feature_vectors=shipping_vectors,
                as_of_timestamp=as_of_timestamp,
            )
            signal = asdict(signal_package.suggestion)
            signal["factor_weights"] = factor_timing_engine.get_factor_weights(
                commodity=commodity,
                regime=signal_package.suggestion.regime_label,
                use_defaults=True,
            )
            daily_signals[commodity] = signal
            signal_snapshots[commodity] = [signal_package.snapshot.to_dict()]
            print(
                f"  ✓ {commodity}: score={signal['composite_score']:.3f}, confidence={signal['confidence_score']:.3f}"
            )
        except Exception as exc:
            print(f"  ✗ {commodity}: {exc}")

    results["components"]["daily_signals"] = daily_signals

    print("\n[SPRINT 1] Building shipping context (always-on integration)...")
    shipping_contexts = {}
    for commodity in test_commodities:
        try:
            context = shipping_context_builder.build(
                commodity=commodity,
                as_of_timestamp=daily_price_data[commodity].index[-21].to_pydatetime(),
            )
            shipping_contexts[commodity] = {
                "features": [item.to_dict() if hasattr(item, "to_dict") else {} for item in context],
                "source": "stubs" if context else "none",
            }
            print(f"  ✓ {commodity}: {shipping_contexts[commodity]['source']} data")
        except Exception as exc:
            print(f"  ✗ {commodity}: {exc}")
    results["components"]["shipping_contexts"] = shipping_contexts

    print("\n[SPRINT 2] Computing regime-aware factor weights...")
    factor_diagnostics = {}
    try:
        for commodity in test_commodities:
            regime = daily_signals.get(commodity, {}).get("regime_label", "mean_reverting_rangebound")
            factor_diagnostics[commodity] = {
                "diagnostics": factor_timing_engine.get_factor_diagnostics(commodity, regime),
                "weights": factor_timing_engine.get_factor_weights(commodity, regime, use_defaults=True),
                "note": "Regime-weighted factors based on rolling Sharpe ratios.",
            }
        print("  ✓ Factor timing engine ready")
    except Exception as exc:
        print(f"  ✗ Factor timing: {exc}")
        factor_diagnostics = {}
    results["components"]["factor_timing"] = factor_diagnostics

    print("\n[SPRINT 3] Optimizing portfolio allocation (Markowitz)...")
    portfolio_weights = {}
    try:
        commodity_signals = {
            commodity: max(0.0, float(daily_signals.get(commodity, {}).get("composite_score", 0.0)))
            for commodity in test_commodities
        }
        if not any(score > 0 for score in commodity_signals.values()):
            commodity_signals = {
                commodity: max(0.01, abs(float(daily_signals.get(commodity, {}).get("composite_score", 0.0))))
                for commodity in test_commodities
            }
        portfolio_weights = portfolio_optimization_engine.optimize_commodity_weights(
            commodity_signals=commodity_signals,
            price_history={
                commodity: daily_price_data[commodity]["close"].astype(float).to_numpy()
                for commodity in test_commodities
            },
        )
        total_weight = sum(portfolio_weights.values())
        print(f"  ✓ Optimization complete (total weight: {total_weight:.1%})")
        for commodity, weight in portfolio_weights.items():
            print(f"    - {commodity}: {weight:.1%}")
    except Exception as exc:
        print(f"  ✗ Portfolio optimization: {exc}")
        equal_weight = 1.0 / len(test_commodities)
        portfolio_weights = {commodity: equal_weight for commodity in test_commodities}
    results["components"]["portfolio_weights"] = portfolio_weights

    if include_backtester:
        print("\n[COMPONENT 1] Running historical backtester (factor attribution)...")
        try:
            backtester = HistoricalBacktester(storage=storage)
            backtest_summary = {}
            for commodity in test_commodities[:1]:
                print(f"  Backtesting {commodity}...")
                eval_result = backtester.backtest_commodity_historical(
                    commodity=commodity,
                    price_data=daily_price_data[commodity],
                    signal_snapshots=signal_snapshots.get(commodity),
                    horizons=[1, 3, 5, 10],
                )
                backtest_summary[commodity] = {
                    regime: len(evals) for regime, evals in eval_result.items()
                }
                print(f"    ✓ Backtest complete: {sum(len(evals) for evals in eval_result.values())} evaluations")
            results["components"]["backtester"] = backtest_summary
        except Exception as exc:
            print(f"  ✗ Backtester: {exc}")
            results["components"]["backtester"] = {"status": "failed", "error": str(exc)}
    else:
        results["components"]["backtester"] = {"status": "skipped"}

    if include_backtester:
        print("\n[COMPONENT 2] Live factor refresh scheduler (daily cron)...")
        try:
            scheduler = LiveFactorScheduler(storage=storage)
            status = scheduler.get_refresh_status()
            print(f"  ✓ Scheduler status: {status.get('scheduler_type', 'manual')}")
            print(f"    Last refresh: {status.get('last_refresh', 'never')}")
            print("    (Automatic scheduling available in production)")
            results["components"]["scheduler"] = status
        except Exception as exc:
            print(f"  ✗ Scheduler: {exc}")
            results["components"]["scheduler"] = {"status": "failed", "error": str(exc)}
    else:
        results["components"]["scheduler"] = {"status": "skipped"}

    if include_inter_market:
        print("\n[COMPONENT 3] Computing inter-market basis (arbitrage opportunities)...")
        inter_market_results = {}
        try:
            basis_calc = InterMarketBasisCalculator()
            for idx, commodity in enumerate(test_commodities):
                mcx_price = float(daily_price_data[commodity]["close"].iloc[-1])
                comex_price = mcx_price * (1.0 + 0.01 * ((idx % 3) - 1))
                basis_metrics = basis_calc.calculate_commodity_basis(
                    commodity=commodity,
                    mcx_price=mcx_price,
                    comex_price=comex_price,
                    fx_rate=84.0,
                )
                basis_history = pd.DataFrame(
                    {
                        "basis": np.linspace(
                            basis_metrics["basis"] - 0.5,
                            basis_metrics["basis"],
                            num=30,
                        )
                    },
                    index=pd.date_range(end=pd.Timestamp.utcnow().normalize(), periods=30, freq="B"),
                )
                opportunity = basis_calc.get_arbitrage_opportunity(
                    basis_history=basis_history,
                    commodity=commodity,
                )
                inter_market_results[commodity] = {
                    **basis_metrics,
                    "opportunity": opportunity,
                }
                print(f"  ✓ {commodity}: basis={basis_metrics['basis_pct']:.2f}%, opportunity={opportunity['arbitrage_signal']}")
        except Exception as exc:
            print(f"  ✗ Inter-market basis: {exc}")
            inter_market_results = {"status": "failed", "error": str(exc)}
        results["components"]["inter_market"] = inter_market_results
    else:
        results["components"]["inter_market"] = {"status": "skipped"}

    if include_intraday:
        print("\n[COMPONENT 4] Generating intraday signals (cross-timeframe)...")
        intraday_results = {}
        try:
            intraday_engine = IntradayFactorRotationEngine()
            for idx, commodity in enumerate(test_commodities):
                intraday_signal = intraday_engine.generate_intraday_signal_package(
                    commodity=commodity,
                    daily_signal=daily_signals.get(commodity, {}),
                    intraday_price_data=_make_mock_intraday_frame(seed=500 + idx),
                    interval="1H",
                )
                intraday_results[commodity] = intraday_signal
                print(f"  ✓ {commodity}: entry_signal={intraday_signal.get('entry_signal', 'hold')}")
        except Exception as exc:
            print(f"  ✗ Intraday engine: {exc}")
            intraday_results = {"status": "failed", "error": str(exc)}
        results["components"]["intraday"] = intraday_results
    else:
        results["components"]["intraday"] = {"status": "skipped"}

    print("\n[COMPONENT 5] Generating budget-aware position suggestions...")
    position_suggestions = {}
    if include_position_suggestions:
        try:
            latest_prices = {
                commodity: float(frame["close"].iloc[-1])
                for commodity, frame in daily_price_data.items()
            }
            volatility = {
                commodity: max(0.01, float(frame["close"].pct_change().dropna().std(ddof=0)))
                for commodity, frame in daily_price_data.items()
            }
            position_suggestions = suggestion_engine.generate_portfolio_suggestions(
                portfolio_weights=portfolio_weights,
                signal_data=daily_signals,
                price_data=latest_prices,
                portfolio_budget=portfolio_budget,
                market_volatility=volatility,
                current_positions=None,
            )
            suggestion_engine.persist_suggestions(position_suggestions)
            print(f"  ✓ Suggestion generation complete ({len(position_suggestions)} recommendations)")
            for commodity, suggestion in position_suggestions.items():
                print(
                    f"    - {commodity}: {suggestion.direction.value.upper()} target {suggestion.target_quantity} @ {suggestion.reference_price:.2f} for ${suggestion.target_notional:,.0f}"
                )
            metrics = suggestion_engine.get_metrics()
            print("\n  Suggestion Metrics:")
            print(f"    Total Target Notional: ${metrics.total_target_notional:,.0f}")
            print(f"    Direction Mix: {metrics.directions}")
        except Exception as exc:
            print(f"  ✗ Position suggestion layer: {exc}")
    results["components"]["position_suggestions"] = {
        "suggestions_generated": len(position_suggestions),
        "suggestions": {
            commodity: suggestion.to_dict()
            for commodity, suggestion in position_suggestions.items()
        },
    }

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"\nGenerated: {datetime.now().isoformat()}")
    print(f"Commodities: {', '.join(test_commodities)}")
    print(f"Suggestions Ready for Review: {len(position_suggestions)}")
    print("\nNext Steps:")
    print("  1. Review position suggestions against your discretionary workflow")
    print("  2. Validate risk constraints (position limits, sector exposure)")
    print("  3. Execute independently outside this engine if you choose to trade")
    print("  4. Refresh factor metrics daily via live_scheduler")

    try:
        storage.write_json(
            settings.storage.report_store,
            f"end_to_end_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            results,
        )
        print(f"\n✓ Results persisted to {settings.storage.report_store}/")
    except Exception as exc:
        print(f"\nWarning: Could not persist results: {exc}")

    return results


if __name__ == "__main__":
    run_end_to_end_demo(
        test_commodities=["CRUDEOIL", "GOLD", "COPPER"],
        include_backtester=True,
        include_intraday=True,
        include_inter_market=True,
        include_position_suggestions=True,
    )

    print("\n✓ Demo complete. Position suggestions available for review.")
