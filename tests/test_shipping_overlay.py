from datetime import datetime

import pandas as pd

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
