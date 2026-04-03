import pandas as pd
from types import SimpleNamespace

from ..analytics.backtest import MacroBacktester
from ..data.storage.local import LocalStorage


def make_price_frame(periods: int = 180) -> pd.DataFrame:
    index = pd.date_range("2024-01-01", periods=periods, freq="B")
    close = pd.Series(range(periods), index=index, dtype=float) * 0.3 + 100
    return pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.4,
            "low": close - 0.4,
            "close": close,
            "volume": 1500,
        },
        index=index,
    )


def test_backtester_uses_evaluation_workflow(tmp_path):
    storage = LocalStorage(str(tmp_path))
    backtester = MacroBacktester(storage=storage)
    result = backtester.run_backtest(commodity="GOLD", price_data=make_price_frame(), persist=True)

    assert result.total_trades > 0
    assert result.evaluation_summary["sample_size"] > 0
    assert result.signal_accuracy >= 0


def test_backtester_applies_signal_from_next_bar(tmp_path, monkeypatch):
    storage = LocalStorage(str(tmp_path))
    backtester = MacroBacktester(storage=storage)
    price_data = pd.DataFrame(
        {
            "open": [100.0, 110.0, 121.0],
            "high": [101.0, 111.0, 126.0],
            "low": [99.0, 109.0, 120.0],
            "close": [100.0, 110.0, 125.0],
            "volume": [1000, 1000, 1000],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="B"),
    )
    signals = [SimpleNamespace(timestamp=price_data.index[1], preferred_direction="long", confidence_score=1.0)]
    monkeypatch.setattr("commodities_quant_engine.config.settings.settings.evaluation_pricing.entry_slippage_bps", 0.0)
    monkeypatch.setattr("commodities_quant_engine.config.settings.settings.evaluation_pricing.exit_slippage_bps", 0.0)
    monkeypatch.setattr("commodities_quant_engine.config.settings.settings.evaluation_pricing.entry_spread_bps", 0.0)
    monkeypatch.setattr("commodities_quant_engine.config.settings.settings.evaluation_pricing.exit_spread_bps", 0.0)
    monkeypatch.setattr("commodities_quant_engine.config.settings.settings.evaluation_pricing.impact_coefficient_bps", 0.0)
    monkeypatch.setattr("commodities_quant_engine.config.settings.settings.evaluation_pricing.max_slippage_from_range_fraction", 0.0)

    portfolio = backtester._simulate_portfolio(price_data, signals)

    assert portfolio.iloc[1] == 0.0
    expected = float(price_data["close"].iloc[2] / price_data["open"].iloc[2] - 1.0) - backtester.transaction_costs
    assert abs(portfolio.iloc[2] - expected) < 1e-12
