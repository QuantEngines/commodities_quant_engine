from datetime import datetime

import pandas as pd

from ..config.settings import settings
from ..nlp import EventIntelligenceEngine
from ..nlp.schemas import CommodityEvent, EventType
from ..signals.composite.composite_decision import CompositeDecisionEngine


def _price_frame(periods: int = 280) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=periods, freq="B")
    close = pd.Series(range(periods), index=index, dtype=float) * 0.35 + 100.0
    return pd.DataFrame(
        {
            "open": close - 0.1,
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
            "volume": 1200 + pd.Series(range(periods), index=index) * 2,
        },
        index=index,
    )


def test_event_schema_clips_scores():
    event = CommodityEvent(
        event_type=EventType.supply_disruption,
        commodity_scope=["CRUDEOIL"],
        confidence=2.5,
        event_strength=-1.0,
        uncertainty_score=3.0,
        regime_relevance=-2.0,
    )
    assert 0.0 <= event.confidence <= 1.0
    assert 0.0 <= event.event_strength <= 1.0
    assert 0.0 <= event.uncertainty_score <= 1.0
    assert 0.0 <= event.regime_relevance <= 1.0


def test_event_classification_and_aggregation_pipeline():
    engine = EventIntelligenceEngine()
    result = engine.process_texts(
        raw_items=[
            "Major refinery outage causes severe supply disruption in crude complex",
            "EIA reports weekly inventory drawdown at Cushing hub",
            "Central bank signals hawkish rate hike path",
        ],
        commodity_scope=["CRUDEOIL"],
        as_of_timestamp=datetime.utcnow(),
    )
    assert result.diagnostics["event_count"] == 3
    assert result.feature_vector["supply_shock_score"] > 0.0
    assert result.feature_vector["inventory_signal_score"] > -1.0
    assert "uncertainty_penalty" in result.feature_vector


def test_event_deduplication_and_clustering_reduces_repeated_headline_weight():
    engine = EventIntelligenceEngine()
    repeated = "Major refinery outage causes severe supply disruption in crude complex"
    result = engine.process_texts(
        raw_items=[repeated, repeated, repeated],
        commodity_scope=["CRUDEOIL"],
        as_of_timestamp=datetime.utcnow(),
    )
    assert result.diagnostics["raw_event_count"] == 3
    assert result.diagnostics["event_count"] == 1
    assert result.diagnostics["dedup_ratio"] > 0.0


def test_persistence_and_regime_proxy_features():
    engine = EventIntelligenceEngine()
    result = engine.process_texts(
        raw_items=[
            {
                "source_id": "p1",
                "timestamp": "2026-02-01T00:00:00",
                "headline": "OPEC announces structural multi-year production cuts",
                "body": "Policy likely persistent across seasons",
                "source": "mock",
            }
        ],
        commodity_scope=["CRUDEOIL"],
        as_of_timestamp=datetime.fromisoformat("2026-02-02T00:00:00"),
    )
    assert result.feature_vector["persistent_trend_event_score"] > 0.0
    assert result.feature_vector["regime_shift_probability_proxy"] > 0.0


def test_composite_engine_fallback_without_event_inputs(monkeypatch):
    monkeypatch.setattr(settings.nlp_event, "enabled", False)
    engine = CompositeDecisionEngine()
    package = engine.generate_signal_package(_price_frame(), "GOLD")
    event_features = package.suggestion.diagnostics.get("event_intelligence_features", {})
    assert event_features.get("supply_shock_score", 0.0) == 0.0


def test_composite_engine_event_overlay_and_explanations(monkeypatch):
    monkeypatch.setattr(settings.nlp_event, "enabled", True)
    monkeypatch.setattr(settings.nlp_event, "max_items_per_cycle", 10)
    engine = CompositeDecisionEngine()
    package = engine.generate_signal_package(
        _price_frame(),
        "CRUDEOIL",
        raw_text_items=[
            "Severe shipping delay and sanctions risk disrupts energy exports",
            "Inventories increased sharply while demand slowed",
        ],
    )
    diagnostics = package.suggestion.diagnostics
    assert diagnostics["event_intelligence_diagnostics"]["event_count"] >= 1
    assert "event_intelligence_features" in diagnostics
    assert "event_overlay_weights" in diagnostics
    assert diagnostics["event_overlay_weights"]["directional_weight"] > 0.0
    assert "Event intelligence:" in package.suggestion.explanation_summary


def test_entity_graph_features_populated_from_policy_geopolitical_text(monkeypatch):
    monkeypatch.setattr(settings.nlp_event, "enabled", True)
    engine = CompositeDecisionEngine()
    package = engine.generate_signal_package(
        _price_frame(),
        "CRUDEOIL",
        raw_text_items=[
            "Sanctions on Iran and disruption risk near Strait of Hormuz as OPEC meets in Saudi Arabia",
        ],
    )
    features = package.suggestion.diagnostics["event_intelligence_features"]
    assert features["entity_country_concentration"] > 0.0
    assert features["shipping_lane_risk_score"] > 0.0
    assert features["producer_concentration_risk"] > 0.0


def test_missing_data_fallback_yields_empty_events(monkeypatch):
    monkeypatch.setattr(settings.nlp_event, "enabled", True)
    engine = EventIntelligenceEngine()
    result = engine.process_texts(raw_items=[], commodity_scope=["GOLD"], as_of_timestamp=datetime.utcnow())
    assert result.events == []
    assert result.feature_vector["supply_shock_score"] == 0.0


def test_cluster_manifest_has_correct_structure_for_repeated_headlines():
    engine = EventIntelligenceEngine()
    repeated = "Major refinery outage causes severe supply disruption in crude complex"
    result = engine.process_texts(
        raw_items=[repeated, repeated, repeated],
        commodity_scope=["CRUDEOIL"],
        as_of_timestamp=datetime.utcnow(),
    )
    assert len(result.cluster_manifest) == 1
    cluster = result.cluster_manifest[0]
    assert cluster["cluster_size"] == 3
    assert cluster["dedup_scale"] < 1.0
    assert cluster["cluster_id"] == "CLU-001"
    assert "rationale" in cluster["representative_rationale"].lower() or "member" in cluster["representative_rationale"].lower()
    assert len(cluster["members"]) == 3
    # Exactly one member must be marked as representative
    rep_count = sum(1 for m in cluster["members"] if m["is_representative"])
    assert rep_count == 1
    # All members must have join_jaccard field
    for member in cluster["members"]:
        assert "join_jaccard" in member


def test_cluster_manifest_has_one_entry_per_unique_event_type():
    engine = EventIntelligenceEngine()
    result = engine.process_texts(
        raw_items=[
            "Major refinery outage disrupts crude supply significantly",
            "Central bank signals hawkish rate hike path ahead",
            "EIA weekly inventory drawdown at Cushing hub below expectations",
        ],
        commodity_scope=["CRUDEOIL"],
        as_of_timestamp=datetime.utcnow(),
    )
    # Three distinct event types — each should be its own cluster
    assert len(result.cluster_manifest) == len(result.events)
    for cluster in result.cluster_manifest:
        assert cluster["cluster_size"] >= 1
        assert cluster["dedup_scale"] == 1.0  # no dedup for singletons
        assert cluster["max_intra_jaccard"] == 1.0


def test_cluster_report_generator_writes_markdown_and_json(tmp_path, monkeypatch):
    from ..nlp.macro_event_engine.cluster_report import ClusterReportGenerator
    from ..data.storage.local import LocalStorage

    storage = LocalStorage(base_dir=str(tmp_path))
    engine = EventIntelligenceEngine()
    repeated = "Refinery outage disrupts crude oil supply chain significantly at major hub"
    result = engine.process_texts(
        raw_items=[repeated, repeated],
        commodity_scope=["CRUDEOIL"],
        as_of_timestamp=datetime.utcnow(),
    )

    generator = ClusterReportGenerator()
    paths = generator.generate(
        cluster_manifest=result.cluster_manifest,
        diagnostics=result.diagnostics,
        commodity="CRUDEOIL",
        signal_id="TEST-SIGNAL-001",
        as_of_timestamp=datetime.utcnow(),
        storage=storage,
    )

    assert paths["json"].exists()
    assert paths["markdown"].exists()

    md_text = paths["markdown"].read_text(encoding="utf-8")
    assert "Event Cluster Inspection Report" in md_text
    assert "CLU-001" in md_text
    assert "TEST-SIGNAL-001" in md_text
    # Dedup ratio should be > 0 for 2 identical headlines
    assert "50.0%" in md_text or "Dedup ratio:" in md_text

    import json
    json_data = json.loads(paths["json"].read_text())
    assert json_data["signal_id"] == "TEST-SIGNAL-001"
    assert json_data["commodity"] == "CRUDEOIL"
    assert len(json_data["clusters"]) == 1
    assert json_data["clusters"][0]["cluster_size"] == 2


def test_research_workflow_generates_cluster_report_file(tmp_path, monkeypatch):
    from ..workflow.research_cycle import ResearchWorkflow
    from ..data.storage.local import LocalStorage

    monkeypatch.setattr(settings.nlp_event, "enabled", True)
    monkeypatch.setattr(settings.nlp_event, "max_items_per_cycle", 10)

    storage = LocalStorage(base_dir=str(tmp_path))
    workflow = ResearchWorkflow(storage=storage)

    repeated = "Pipeline outage disrupts crude supply severely at main hub"
    package = workflow.run_signal_cycle(
        commodity="CRUDEOIL",
        price_data=_price_frame(),
        raw_text_items=[repeated, repeated],
        persist_snapshot=False,
        persist_report=True,
    )

    signal_id = package.snapshot.signal_id
    report_dir = tmp_path / settings.storage.report_store
    md_files = list(report_dir.glob(f"CRUDEOIL_{signal_id}_cluster_report.md"))
    json_files = list(report_dir.glob(f"CRUDEOIL_{signal_id}_cluster_report.json"))
    assert md_files, "Markdown cluster report was not written"
    assert json_files, "JSON cluster report was not written"
