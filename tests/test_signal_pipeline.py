from datetime import datetime

import pandas as pd

from ..config.settings import settings
from ..data.storage.local import LocalStorage
from ..regimes.regime_engine import RegimeEngine
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
    assert "diagnostics_summary_path" in artifact.scorecards
    assert "attribution_summary_path" in artifact.scorecards


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


def test_research_workflow_persists_structured_events_with_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(settings.nlp_event, "enabled", True)
    storage = LocalStorage(str(tmp_path))
    workflow = ResearchWorkflow(storage=storage)
    price_data = make_price_frame()

    package = workflow.run_signal_cycle(
        "CRUDEOIL",
        price_data.iloc[:-5],
        raw_text_items=[
            "Pipeline outage and sanctions risk near Hormuz disrupt exports",
            "Inventory drawdown deepens crude tightness",
        ],
    )

    assert package.snapshot.signal_id
    persisted = storage.load_jsonl("signals", "CRUDEOIL_structured_events")
    assert persisted
    assert all(item.get("signal_id") == package.snapshot.signal_id for item in persisted)


def test_research_workflow_runs_suggestion_only_portfolio_cycle(tmp_path):
    storage = LocalStorage(str(tmp_path))
    workflow = ResearchWorkflow(storage=storage)

    price_data_by_commodity = {
        "GOLD": make_price_frame(),
        "CRUDEOIL": make_price_frame(),
        "COPPER": make_price_frame(),
    }

    result = workflow.run_portfolio_cycle(
        price_data_by_commodity=price_data_by_commodity,
        portfolio_budget=1_000_000.0,
        current_positions={"GOLD": 0, "CRUDEOIL": 0, "COPPER": 0},
    )

    assert result["signal_packages"]
    assert result["portfolio_weights"]
    assert abs(sum(result["portfolio_weights"].values()) - 1.0) < 1e-6
    assert "portfolio_report_name" in result

    report_payload = storage.read_json("reports", result["portfolio_report_name"])
    assert report_payload.get("suggestion_only") is True
    assert "No broker orders" in report_payload.get("note", "")
    assert isinstance(report_payload.get("suggestions", {}), dict)
    assert isinstance(report_payload.get("ranking_table", []), list)
    assert "ranking_markdown" in report_payload
    assert isinstance(report_payload.get("dashboard", {}), dict)


def test_research_workflow_runs_intraday_signal_cycle(tmp_path):
    storage = LocalStorage(str(tmp_path))
    workflow = ResearchWorkflow(storage=storage)

    daily = make_price_frame(periods=260)
    idx = pd.date_range("2025-09-01 09:00", periods=20, freq="1h")
    base = pd.Series(range(len(idx)), index=idx, dtype=float) * 0.1 + 100.0
    intraday = pd.DataFrame(
        {
            "open": base - 0.05,
            "high": base + 0.15,
            "low": base - 0.15,
            "close": base,
            "volume": 500 + pd.Series(range(len(idx)), index=idx) * 5,
        },
        index=idx,
    )

    result = workflow.run_intraday_signal_cycle(
        commodity="GOLD",
        daily_price_data=daily,
        intraday_price_data=intraday,
        interval="1H",
    )

    assert result["daily_package"].snapshot.signal_id
    intraday_signal = result["intraday_signal"]
    assert intraday_signal["commodity"] == "GOLD"
    assert "combined_signal" in intraday_signal
    assert "entry_signal" in intraday_signal


def test_regime_engine_supports_hmm_mode(monkeypatch):
    monkeypatch.setattr(settings.signal, "regime_model", "hmm")
    monkeypatch.setattr(settings.signal, "hmm_min_history_rows", 60)
    monkeypatch.setattr(settings.signal, "hmm_states", 4)

    data = make_price_frame(periods=260)
    engine = RegimeEngine()
    regime = engine.detect_regime(data, commodity="GOLD")

    assert regime.label in {
        "trend_following_bullish",
        "trend_following_bearish",
        "volatile_reversal",
        "mean_reverting_rangebound",
        "neutral",
    }
    assert 0.0 <= regime.probability <= 1.0
    assert 0.0 <= regime.confidence <= 1.0
