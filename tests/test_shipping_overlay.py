from datetime import datetime

import pandas as pd

from ..data.models import MacroFeature
from ..shipping import ShippingFeaturePipeline
from ..shipping.models import ShippingFeatureVector, ShippingObservationWindow
from ..signals.composite.composite_decision import CompositeDecisionEngine


def make_price_frame(periods: int = 260) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=periods, freq="B")
    close = pd.Series(range(periods), index=index, dtype=float) * 0.4 + 100.0
    return pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": 1000 + pd.Series(range(periods), index=index) * 3,
        },
        index=index,
    )


def make_shipping_vector(timestamp: datetime, quality_score: float = 0.82) -> ShippingFeatureVector:
    return ShippingFeatureVector(
        timestamp=timestamp,
        commodity="CRUDEOIL",
        source="unit_test",
        observation_window=ShippingObservationWindow(
            start_time=timestamp - pd.Timedelta(days=10),
            end_time=timestamp,
            frequency="1D",
            source="unit_test",
            commodity="CRUDEOIL",
        ),
        features={
            "port_congestion_score": 0.8,
            "anchorage_buildup_score": 0.7,
            "route_disruption_score": 0.9,
            "chokepoint_stress_score": 0.85,
            "tanker_flow_momentum": -0.6,
            "shipping_momentum_score": 0.4,
        },
        quality_score=quality_score,
        confidence_score=quality_score,
        commodity_tags=["CRUDEOIL"],
        key_drivers=["Port congestion elevated", "Hormuz stress elevated"],
    )


def make_benchmark_macro_features(timestamp: datetime) -> list[MacroFeature]:
    return [
        MacroFeature(feature_name="bdi_level", timestamp=timestamp, value=1450.0),
        MacroFeature(feature_name="bdi_zscore", timestamp=timestamp, value=1.4),
        MacroFeature(feature_name="bdi_momentum_20d", timestamp=timestamp, value=0.08),
        MacroFeature(feature_name="bdi_shock_flag", timestamp=timestamp, value=1.0),
    ]


def make_tanker_benchmark_macro_features(timestamp: datetime) -> list[MacroFeature]:
    return [
        MacroFeature(feature_name="bdti_level", timestamp=timestamp, value=1125.0),
        MacroFeature(feature_name="bdti_zscore", timestamp=timestamp, value=1.1),
        MacroFeature(feature_name="bdti_momentum_10d", timestamp=timestamp, value=0.12),
        MacroFeature(feature_name="bdti_shock_flag", timestamp=timestamp, value=0.0),
        MacroFeature(feature_name="bcti_level", timestamp=timestamp, value=845.0),
        MacroFeature(feature_name="bcti_zscore", timestamp=timestamp, value=0.8),
        MacroFeature(feature_name="bcti_momentum_10d", timestamp=timestamp, value=0.07),
        MacroFeature(feature_name="bcti_shock_flag", timestamp=timestamp, value=0.0),
        MacroFeature(feature_name="tanker_value_level", timestamp=timestamp, value=55.0),
        MacroFeature(feature_name="tanker_value_zscore", timestamp=timestamp, value=0.9),
        MacroFeature(feature_name="tanker_value_momentum_20d", timestamp=timestamp, value=0.05),
        MacroFeature(feature_name="tanker_value_shock_flag", timestamp=timestamp, value=0.0),
    ]


def make_bulk_positions(timestamp: datetime) -> pd.DataFrame:
    base = pd.Timestamp(timestamp) - pd.Timedelta(days=8)
    rows = []
    for day in range(8):
        rows.extend(
            [
                {
                    "vessel_id": f"bulk-a-{day}",
                    "timestamp": base + pd.Timedelta(days=day),
                    "latitude": 31.0 + day * 0.05,
                    "longitude": 121.0 + day * 0.05,
                    "speed_knots": max(1.0, 10.0 - day * 0.8),
                    "cargo_class": "bulk_carrier",
                },
                {
                    "vessel_id": f"bulk-b-{day}",
                    "timestamp": base + pd.Timedelta(days=day, hours=6),
                    "latitude": 30.8 + day * 0.04,
                    "longitude": 120.8 + day * 0.04,
                    "speed_knots": max(1.0, 9.0 - day * 0.7),
                    "cargo_class": "bulk_carrier",
                },
            ]
        )
    return pd.DataFrame(rows)


def make_tanker_positions(timestamp: datetime) -> pd.DataFrame:
    base = pd.Timestamp(timestamp) - pd.Timedelta(days=8)
    rows = []
    for day in range(8):
        rows.extend(
            [
                {
                    "vessel_id": f"tanker-a-{day}",
                    "timestamp": base + pd.Timedelta(days=day),
                    "latitude": 25.2 + day * 0.03,
                    "longitude": 56.3 + day * 0.03,
                    "speed_knots": max(1.0, 11.0 - day * 0.9),
                    "cargo_class": "crude_tanker",
                },
                {
                    "vessel_id": f"tanker-b-{day}",
                    "timestamp": base + pd.Timedelta(days=day, hours=8),
                    "latitude": 25.5 + day * 0.02,
                    "longitude": 56.6 + day * 0.02,
                    "speed_knots": max(1.0, 10.0 - day * 0.7),
                    "cargo_class": "crude_tanker",
                },
            ]
        )
    return pd.DataFrame(rows)


def test_composite_engine_includes_shipping_overlay_when_feature_vector_is_present():
    price_frame = make_price_frame()
    as_of_timestamp = price_frame.index[-1].to_pydatetime()
    engine = CompositeDecisionEngine()

    package = engine.generate_signal_package(
        data=price_frame,
        commodity="CRUDEOIL",
        shipping_feature_vectors=[make_shipping_vector(as_of_timestamp)],
        as_of_timestamp=as_of_timestamp,
    )

    assert package.suggestion.shipping_summary in {"Supportive", "Elevated risk", "Mixed"}
    assert package.suggestion.shipping_alignment_score is not None
    assert package.suggestion.shipping_alignment_score > 0.0
    assert package.snapshot.shipping_alignment_score is not None
    assert "shipping" in package.snapshot.component_scores
    assert "shipping_features" in package.suggestion.diagnostics


def test_sparse_shipping_data_reduces_quality_without_breaking_signal_generation():
    price_frame = make_price_frame()
    as_of_timestamp = price_frame.index[-1].to_pydatetime()
    engine = CompositeDecisionEngine()

    package = engine.generate_signal_package(
        data=price_frame,
        commodity="CRUDEOIL",
        shipping_feature_vectors=[make_shipping_vector(as_of_timestamp, quality_score=0.20)],
        as_of_timestamp=as_of_timestamp,
    )

    assert package.suggestion.shipping_data_quality_score == 0.20
    assert package.suggestion.shipping_data_quality_penalty > 0.0
    assert package.suggestion.signal_id


def test_bdi_is_promoted_to_shipping_benchmark_for_base_metals():
    price_frame = make_price_frame()
    as_of_timestamp = price_frame.index[-1].to_pydatetime()
    engine = CompositeDecisionEngine()
    bulk_positions = make_bulk_positions(as_of_timestamp)
    benchmark_timestamp = pd.to_datetime(bulk_positions["timestamp"]).max().to_pydatetime()
    shipping_vectors = ShippingFeaturePipeline().run(
        commodity="COPPER",
        vessel_positions=bulk_positions,
        route_events=pd.DataFrame(),
        macro_features=make_benchmark_macro_features(benchmark_timestamp),
        as_of_timestamp=as_of_timestamp,
    )

    package = engine.generate_signal_package(
        data=price_frame,
        commodity="COPPER",
        macro_features=make_benchmark_macro_features(benchmark_timestamp),
        shipping_feature_vectors=shipping_vectors,
        as_of_timestamp=as_of_timestamp,
    )

    shipping_features = package.suggestion.diagnostics["shipping_features"]
    assert shipping_features["bdi_benchmark_active"] == 1.0
    assert shipping_features["bdi_benchmark_zscore"] == 1.4
    assert "bdi_shipping_divergence" in shipping_features
    assert shipping_features["shipping_market_benchmark_active"] == 1.0
    assert any("dry-bulk" in driver.lower() for driver in package.suggestion.key_shipping_drivers)


def test_tanker_benchmarks_are_promoted_for_crude_shipping_context():
    price_frame = make_price_frame()
    as_of_timestamp = price_frame.index[-1].to_pydatetime()
    engine = CompositeDecisionEngine()
    tanker_positions = make_tanker_positions(as_of_timestamp)
    benchmark_timestamp = pd.to_datetime(tanker_positions["timestamp"]).max().to_pydatetime()
    shipping_vectors = ShippingFeaturePipeline().run(
        commodity="CRUDEOIL",
        vessel_positions=tanker_positions,
        route_events=pd.DataFrame(),
        macro_features=make_tanker_benchmark_macro_features(benchmark_timestamp),
        as_of_timestamp=as_of_timestamp,
    )

    package = engine.generate_signal_package(
        data=price_frame,
        commodity="CRUDEOIL",
        macro_features=make_tanker_benchmark_macro_features(benchmark_timestamp),
        shipping_feature_vectors=shipping_vectors,
        as_of_timestamp=as_of_timestamp,
    )

    shipping_features = package.suggestion.diagnostics["shipping_features"]
    assert shipping_features["bdti_benchmark_active"] == 1.0
    assert shipping_features["shipping_market_benchmark_active"] == 1.0
    assert shipping_features["shipping_market_benchmark_zscore"] > 0.0
    assert any("tanker" in driver.lower() for driver in package.suggestion.key_shipping_drivers)
