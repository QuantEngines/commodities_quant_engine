from datetime import date, datetime

from ..data.models import Commodity, Contract, OHLCV, SignalSnapshot


def test_commodity_creation():
    comm = Commodity(
        symbol="GOLD",
        name="Gold",
        exchange="MCX",
        segment="bullion",
        contract_multiplier=100,
    )
    assert comm.symbol == "GOLD"
    assert comm.contract_multiplier == 100


def test_contract_creation():
    contract = Contract(
        commodity="GOLD",
        symbol="GOLDAPR26",
        expiry_date=date(2026, 4, 28),
        lot_size=100,
        tick_size=0.01,
        multiplier=100,
        exchange="MCX",
        segment="bullion",
    )
    assert contract.contract_code == "GOLDAPR26"


def test_ohlcv_creation():
    ohlcv = OHLCV(
        timestamp=datetime(2026, 3, 15, 10, 0),
        open=50000.0,
        high=50200.0,
        low=49900.0,
        close=50100.0,
        volume=1000,
        open_interest=5000,
    )
    assert ohlcv.close == 50100.0
    assert ohlcv.to_dict()["close"] == 50100.0


def test_signal_snapshot_to_dict():
    snapshot = SignalSnapshot(
        signal_id="sig-1",
        timestamp=datetime(2026, 3, 15, 10, 0),
        commodity="GOLD",
        contract="GOLDAPR26",
        exchange="MCX",
        signal_category="Strong Long Candidate",
        direction="long",
        conviction=0.72,
        regime_label="trend_following_bullish",
        regime_probability=0.7,
        inefficiency_score=-1.1,
        composite_score=1.3,
        suggested_horizon=5,
        directional_scores={5: 1.1},
        key_drivers=["Momentum positive"],
        key_risks=["Volatility elevated"],
        component_scores={"directional": 0.8},
        feature_vector={"momentum_5d": 1.2},
        model_version="default",
        config_version="test",
        data_quality_flag="good",
    )
    assert snapshot.to_dict()["signal_id"] == "sig-1"
