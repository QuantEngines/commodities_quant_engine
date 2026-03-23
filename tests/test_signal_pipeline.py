from datetime import datetime

import pandas as pd

from ..data.storage.local import LocalStorage
from ..signals.composite.composite_decision import CompositeDecisionEngine
from ..workflow import ResearchWorkflow


def make_price_frame(periods: int = 260) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=periods, freq="B")
    close = pd.Series(range(periods), index=index, dtype=float) * 0.4 + 100.0
    frame = pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1000 + pd.Series(range(periods), index=index) * 3,
        },
        index=index,
    )
    return frame


def test_composite_engine_generates_auditable_signal_package():
    engine = CompositeDecisionEngine()
    package = engine.generate_signal_package(make_price_frame(), "GOLD")
    assert package.suggestion.signal_id
    assert package.snapshot.signal_id == package.suggestion.signal_id
    assert "directional" in package.snapshot.component_scores
    assert package.quality_report.flag in {"good", "stale"}


def test_research_workflow_persists_signal_and_evaluation(tmp_path):
    storage = LocalStorage(str(tmp_path))
    workflow = ResearchWorkflow(storage=storage)
    price_data = make_price_frame()

    package = workflow.run_signal_cycle("GOLD", price_data.iloc[:-10])
    assert package.snapshot.signal_id

    artifact = workflow.run_evaluation_cycle("GOLD", price_data, as_of_timestamp=price_data.index[-1].to_pydatetime())
    assert artifact.summary_metrics["sample_size"] >= 1


def test_research_workflow_applies_persisted_calibration_maps(tmp_path):
    storage = LocalStorage(str(tmp_path))
    workflow = ResearchWorkflow(storage=storage)
    price_data = make_price_frame()
    storage.write_json(
        "evaluations",
        "GOLD_calibration",
        {
            "confidence_calibration": {
                "anchors": [
                    {"confidence": 0.0, "calibrated_hit_rate": 0.2},
                    {"confidence": 1.0, "calibrated_hit_rate": 0.2},
                ]
            },
            "regime_calibration": {
                "regime_map": {
                    "trend_following_bullish": 0.33,
                    "trend_following_bearish": 0.44,
                    "mean_reverting_rangebound": 0.55,
                    "volatile_reversal": 0.66,
                    "neutral": 0.5,
                }
            },
        },
    )

    package = workflow.run_signal_cycle("GOLD", price_data.iloc[:-10])
    calibrated_prob = package.suggestion.regime_probabilities[package.suggestion.regime_label]

    assert abs(package.suggestion.confidence_score - 0.2) < 1e-9
    assert 0.0 <= calibrated_prob <= 1.0
    assert package.suggestion.diagnostics.get("confidence_calibrated") is True
