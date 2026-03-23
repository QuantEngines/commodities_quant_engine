import json

import pandas as pd

from ..analytics.adaptation import AdaptiveParameterEngine
from ..analytics.evaluation import SignalEvaluationEngine
from ..data.models import SignalSnapshot
from ..data.storage.local import LocalStorage


def make_snapshot(i: int) -> SignalSnapshot:
    timestamp = pd.Timestamp("2025-01-01") + pd.Timedelta(days=i)
    feature_value = 1.0 if i % 2 == 0 else -1.0
    return SignalSnapshot(
        signal_id=f"sig-{i}",
        timestamp=timestamp.to_pydatetime(),
        commodity="GOLD",
        contract="GOLDAPR26",
        exchange="MCX",
        signal_category="Test",
        direction="long" if feature_value > 0 else "short",
        conviction=0.6,
        regime_label="trend_following_bullish" if feature_value > 0 else "trend_following_bearish",
        regime_probability=0.7,
        inefficiency_score=-0.1,
        composite_score=feature_value,
        suggested_horizon=5,
        directional_scores={5: 0.0},
        key_drivers=["momentum"],
        key_risks=["none"],
        component_scores={"directional": feature_value},
        feature_vector={
            "momentum_5d": feature_value,
            "momentum_20d": feature_value,
            "trend_strength_20d": feature_value,
            "short_reversal_5d": -feature_value * 0.2,
            "volatility_20d": 0.1,
            "drawdown_20d": -0.1,
            "volume_trend_20d": 0.2,
            "carry_yield": 0.0,
        },
        model_version="default",
        config_version="test",
        data_quality_flag="good",
    )


def test_adaptive_engine_requires_evidence_and_creates_candidate(tmp_path, monkeypatch):
    storage = LocalStorage(str(tmp_path))
    evaluation_engine = SignalEvaluationEngine(storage=storage)
    adaptation_engine = AdaptiveParameterEngine(storage=storage)
    monkeypatch.setattr("commodities_quant_engine.config.settings.settings.adaptation.max_feature_drift", 100.0)
    monkeypatch.setattr("commodities_quant_engine.config.settings.settings.adaptation.min_hit_rate_improvement", 0.0)
    monkeypatch.setattr("commodities_quant_engine.config.settings.settings.adaptation.min_rank_ic_improvement", 0.0)
    snapshots = [make_snapshot(i) for i in range(60)]
    evaluation_engine.persist_signal_snapshots(snapshots, commodity="GOLD")

    evaluation_rows = []
    for i, snapshot in enumerate(snapshots):
        realized_return = (0.01 + i * 0.0005) if snapshot.direction == "long" else -(0.01 + i * 0.0005)
        evaluation_rows.append(
            {
                "signal_id": snapshot.signal_id,
                "timestamp": snapshot.timestamp,
                "commodity": "GOLD",
                "horizon": 5,
                "direction": snapshot.direction,
                "confidence": snapshot.conviction,
                "composite_score": snapshot.composite_score,
                "realized_return": realized_return,
                "signed_return": 0.02,
                "direction_correct": True,
                "excess_return": 0.02,
                "volatility_adjusted_return": 1.0,
                "max_favorable_excursion": 0.03,
                "max_adverse_excursion": -0.01,
                "follow_through_ratio": 0.8,
                "reversal_probability": 0.0,
                "event_window_flag": False,
                "regime_label": snapshot.regime_label,
                "realized_regime_label": snapshot.regime_label,
                "regime_alignment": True,
                "confidence_bucket": "0.40-0.60",
                "metadata": json.dumps({"signal_category": "Test"}),
            }
        )
    storage.append_dataframe(
        pd.DataFrame(evaluation_rows),
        "evaluations",
        "GOLD_detailed",
        dedupe_on=["signal_id", "horizon"],
    )

    decision = adaptation_engine.recommend_update("GOLD", dry_run=True, approve=False)
    assert decision.candidate_version_id is not None
    assert not decision.promoted
    assert decision.safety_checks["min_sample_size"]


def test_adaptive_engine_rejects_flat_non_improving_candidate(tmp_path):
    storage = LocalStorage(str(tmp_path))
    evaluation_engine = SignalEvaluationEngine(storage=storage)
    adaptation_engine = AdaptiveParameterEngine(storage=storage)
    snapshots = [make_snapshot(i) for i in range(60)]
    evaluation_engine.persist_signal_snapshots(snapshots, commodity="GOLD")

    evaluation_rows = []
    for snapshot in snapshots:
        evaluation_rows.append(
            {
                "signal_id": snapshot.signal_id,
                "timestamp": snapshot.timestamp,
                "commodity": "GOLD",
                "horizon": 5,
                "direction": snapshot.direction,
                "confidence": snapshot.conviction,
                "composite_score": snapshot.composite_score,
                "realized_return": 0.0,
                "signed_return": 0.0,
                "direction_correct": False,
                "excess_return": 0.0,
                "volatility_adjusted_return": 0.0,
                "max_favorable_excursion": 0.0,
                "max_adverse_excursion": 0.0,
                "follow_through_ratio": 0.0,
                "reversal_probability": 0.0,
                "event_window_flag": False,
                "regime_label": snapshot.regime_label,
                "realized_regime_label": "neutral",
                "regime_alignment": False,
                "confidence_bucket": "0.40-0.60",
                "metadata": json.dumps({"signal_category": "Test"}),
            }
        )
    storage.append_dataframe(
        pd.DataFrame(evaluation_rows),
        "evaluations",
        "GOLD_detailed",
        dedupe_on=["signal_id", "horizon"],
    )

    decision = adaptation_engine.recommend_update("GOLD", dry_run=True, approve=False)
    assert decision.candidate_version_id is None
    assert not decision.promoted
