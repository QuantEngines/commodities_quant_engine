from datetime import date, datetime

from ..data.models import Commodity, Contract, OHLCV, SignalSnapshot, Suggestion


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


def test_suggestion_markdown_includes_enriched_snapshot_sections():
    suggestion = Suggestion(
        timestamp=datetime(2026, 3, 15, 10, 0),
        commodity="GOLD",
        exchange="MCX",
        active_contract="GOLDAPR26",
        regime_label="trend_following_bullish",
        regime_probabilities={"trend_following_bullish": 0.73},
        directional_scores={1: 0.2, 5: 0.8},
        inefficiency_score=-1.1,
        risk_penalty=0.25,
        composite_score=1.3,
        final_category="Strong Long Candidate",
        preferred_direction="long",
        suggested_entry_style="buy_pullbacks",
        suggested_holding_horizon=5,
        key_supporting_drivers=["Momentum positive", "Macro tailwind"],
        key_contradictory_drivers=["Crowded positioning"],
        principal_risks=["Volatility elevated"],
        explanation_summary="Trend, regime, and macro all align for a long setup.",
        data_quality_flag="good",
        confidence_score=0.72,
        signal_id="sig-1",
        model_version="default",
        config_version="test",
        diagnostics={
            "component_scores": {"directional": 0.8, "macro": 0.3},
            "directional_confidences": {1: 0.55, 5: 0.71},
            "feature_vector": {"momentum_5d": 1.2, "carry_20d": -0.4},
            "event_intelligence_features": {"policy_risk_score": 0.2},
            "quality_issues": [],
        },
        macro_regime_summary="bullish growth impulse",
        macro_feature_highlights={"usd_inr_z": -0.3, "real_rate_z": -0.4},
        macro_alignment_score=0.42,
        macro_conflict_score=0.11,
        macro_event_risk_flag=False,
        macro_confidence_adjustment=0.08,
        macro_explanation_summary="Rates and FX support bullion strength.",
        key_macro_drivers=["Falling real rates"],
        key_macro_risks=["Surprise hawkish policy"],
        news_narrative_summary="Central-bank tone remains supportive.",
    )

    markdown = suggestion.to_markdown()

    assert "## Decision Snapshot" in markdown
    assert "## Signal Anatomy" in markdown
    assert "Directional Term Structure" in markdown
    assert "Component Stack" in markdown
    assert "## Thesis" in markdown
    assert "## Macro Context" in markdown
