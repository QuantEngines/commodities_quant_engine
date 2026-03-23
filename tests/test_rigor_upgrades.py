from types import SimpleNamespace

import pandas as pd

from ..analytics.adaptation import AdaptiveParameterEngine
from ..analytics.backtest import MacroBacktester
from ..analytics.execution import slippage_rate
from ..data.quality_checks import MarketDataValidator
from ..data.storage.local import LocalStorage


def test_adaptation_uses_purged_walkforward_splits(tmp_path):
    storage = LocalStorage(str(tmp_path))
    engine = AdaptiveParameterEngine(storage=storage)

    splits = engine._purged_walkforward_splits(length=120, horizon=5)

    assert splits
    for train_idx, holdout_idx in splits:
        assert len(train_idx) > 0
        assert len(holdout_idx) > 0
        assert train_idx.max() < holdout_idx.min()
        # Purge+embargo leave a non-trivial gap between train and holdout segments.
        assert holdout_idx.min() - train_idx.max() >= 10


def test_vol_target_sizing_scales_with_realized_vol(tmp_path, monkeypatch):
    storage = LocalStorage(str(tmp_path))
    backtester = MacroBacktester(storage=storage)

    monkeypatch.setattr("commodities_quant_engine.config.settings.settings.execution.target_annualized_vol", 0.20)
    monkeypatch.setattr("commodities_quant_engine.config.settings.settings.execution.annualization_days", 252)
    monkeypatch.setattr("commodities_quant_engine.config.settings.settings.execution.max_abs_position", 1.0)
    monkeypatch.setattr("commodities_quant_engine.config.settings.settings.execution.min_trade_confidence", 0.0)

    signal = SimpleNamespace(preferred_direction="long", confidence_score=0.7)
    low_vol_position = backtester._signal_to_position(signal, realized_vol=0.005)
    high_vol_position = backtester._signal_to_position(signal, realized_vol=0.03)

    assert abs(low_vol_position) >= abs(high_vol_position)
    assert abs(low_vol_position) <= 1.0
    assert abs(high_vol_position) <= 1.0


def test_execution_slippage_increases_with_participation():
    row = pd.Series({"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 1200.0})
    base = slippage_rate(row, phase="entry", median_volume=1200.0, participation=0.0)
    higher = slippage_rate(row, phase="entry", median_volume=1200.0, participation=1.0)
    assert higher > base


def test_market_data_validator_fails_on_duplicate_timestamp_and_nan_close():
    validator = MarketDataValidator()
    index = pd.DatetimeIndex(["2025-01-01", "2025-01-01", "2025-01-02"])
    frame = pd.DataFrame(
        {
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.0, None, 102.0],
            "volume": [1000.0, 1000.0, 1000.0],
        },
        index=index,
    )

    report = validator.validate(frame)

    assert not report.is_valid
    assert report.flag == "incomplete"
    assert any("duplicate timestamps" in issue.lower() for issue in report.issues)
    assert any("missing values" in issue.lower() for issue in report.issues)
