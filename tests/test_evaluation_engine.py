from datetime import datetime

import pandas as pd

from ..analytics.evaluation import SignalEvaluationEngine
from ..data.models import SignalSnapshot
from ..data.storage.local import LocalStorage


def make_uptrend_frame(periods: int = 15) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=periods, freq="B")
    close = pd.Series(range(periods), index=index, dtype=float) + 100.0
    return pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.2,
            "close": close,
            "volume": 1000,
        },
        index=index,
    )


def make_snapshot(signal_id: str, timestamp: datetime, direction: str) -> SignalSnapshot:
    return SignalSnapshot(
        signal_id=signal_id,
        timestamp=timestamp,
        commodity="GOLD",
        contract="GOLDAPR26",
        exchange="MCX",
        signal_category="Test",
        direction=direction,
        conviction=0.7,
        regime_label="trend_following_bullish" if direction == "long" else "trend_following_bearish",
        regime_probability=0.7,
        inefficiency_score=-0.5,
        composite_score=0.9 if direction == "long" else -0.9,
        suggested_horizon=5,
        directional_scores={1: 0.5, 3: 0.7, 5: 0.9},
        key_drivers=["trend"],
        key_risks=["none"],
        component_scores={"directional": 0.7},
        feature_vector={"momentum_5d": 1.0},
        model_version="default",
        config_version="test",
        data_quality_flag="good",
    )


def test_evaluation_engine_is_timestamp_safe(tmp_path):
    storage = LocalStorage(str(tmp_path))
    engine = SignalEvaluationEngine(storage=storage)
    price_data = make_uptrend_frame()
    snapshots = [
        make_snapshot("sig-long", price_data.index[5].to_pydatetime(), "long"),
        make_snapshot("sig-short", price_data.index[10].to_pydatetime(), "short"),
    ]

    records = engine._build_evaluation_records(
        price_data=price_data,
        snapshots=snapshots,
        horizons=[1, 3, 5],
        macro_events=[],
        as_of_timestamp=price_data.index[12].to_pydatetime(),
    )

    assert len(records) == 4
    assert any(record.signal_id == "sig-long" and record.direction_correct for record in records)
    assert all(not (record.signal_id == "sig-short" and record.horizon == 5) for record in records)

    artifact = engine.evaluate_signals(
        commodity="GOLD",
        price_data=price_data,
        signal_snapshots=snapshots,
        horizons=[1, 3, 5],
        as_of_timestamp=price_data.index[12].to_pydatetime(),
        persist=False,
    )
    assert artifact.summary_metrics["sample_size"] == 4


def test_evaluation_engine_uses_next_bar_entry_by_default(tmp_path, monkeypatch):
    storage = LocalStorage(str(tmp_path))
    engine = SignalEvaluationEngine(storage=storage)
    price_data = make_uptrend_frame()
    snapshot = make_snapshot("sig-long", price_data.index[5].to_pydatetime(), "long")
    monkeypatch.setattr("commodities_quant_engine.config.settings.settings.execution.entry_slippage_bps", 0.0)
    monkeypatch.setattr("commodities_quant_engine.config.settings.settings.execution.exit_slippage_bps", 0.0)
    monkeypatch.setattr("commodities_quant_engine.config.settings.settings.execution.max_slippage_from_range_fraction", 0.0)

    records = engine._build_evaluation_records(
        price_data=price_data,
        snapshots=[snapshot],
        horizons=[1],
        macro_events=[],
        as_of_timestamp=price_data.index[10].to_pydatetime(),
    )

    expected = float(price_data["close"].iloc[7] / price_data["open"].iloc[6] - 1.0)
    assert len(records) == 1
    assert abs(records[0].realized_return - expected) < 1e-12
    assert records[0].metadata["entry_timestamp"].startswith(str(price_data.index[6].date()))
