from __future__ import annotations

import json
import importlib
from datetime import datetime, timezone
from pathlib import Path
import sys

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

settings = importlib.import_module("commodities_quant_engine.config.settings").settings
CompositeDecisionEngine = importlib.import_module(
    "commodities_quant_engine.signals.composite.composite_decision"
).CompositeDecisionEngine


def make_price_frame(periods: int = 260) -> pd.DataFrame:
    index = pd.date_range("2025-01-01", periods=periods, freq="B")
    close = pd.Series(range(periods), index=index, dtype=float) * 0.45 + 98.0
    return pd.DataFrame(
        {
            "open": close - 0.15,
            "high": close + 0.45,
            "low": close - 0.45,
            "close": close,
            "volume": 1400 + pd.Series(range(periods), index=index) * 4,
        },
        index=index,
    )


def main() -> None:
    settings.nlp_event.enabled = True
    settings.nlp_event.max_items_per_cycle = 20

    text_inputs = [
        {
            "source_id": "supply_1",
            "timestamp": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            "headline": "Major pipeline outage triggers severe supply disruption in crude exports",
            "body": "Officials confirm disruption expected for weeks",
            "source": "demo_wire",
        },
        {
            "source_id": "rates_1",
            "timestamp": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            "headline": "Central bank signals hawkish rate hike trajectory",
            "body": "Dollar strengthens as policy guidance tightens",
            "source": "demo_wire",
        },
        {
            "source_id": "inv_1",
            "timestamp": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            "headline": "Weekly inventory drawdown reported across key storage hubs",
            "body": "Stocks fell sharply versus expectations",
            "source": "demo_wire",
        },
        {
            "source_id": "weather_1",
            "timestamp": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            "headline": "Severe drought risk threatens crop outlook in major producing regions",
            "body": "Monsoon deficit raises yield uncertainty",
            "source": "demo_wire",
        },
        {
            "source_id": "opec_1",
            "timestamp": datetime.now(timezone.utc).replace(tzinfo=None).isoformat(),
            "headline": "OPEC commentary points to structural production discipline",
            "body": "Guidance suggests medium-term supply restraint",
            "source": "demo_wire",
        },
    ]

    engine = CompositeDecisionEngine()
    package_without_text = engine.generate_signal_package(make_price_frame(), "CRUDEOIL", raw_text_items=[])
    package_with_text = engine.generate_signal_package(make_price_frame(), "CRUDEOIL", raw_text_items=text_inputs)

    print("=== Event Intelligence Demo ===")
    print("Structured event diagnostics:")
    print(json.dumps(package_with_text.suggestion.diagnostics.get("event_intelligence_diagnostics", {}), indent=2, default=str))

    print("\nGenerated event feature vector:")
    print(json.dumps(package_with_text.suggestion.diagnostics.get("event_intelligence_features", {}), indent=2, default=str))

    print("\nDirectional/composite impact:")
    print(
        json.dumps(
            {
                "without_text": {
                    "composite_score": package_without_text.suggestion.composite_score,
                    "confidence": package_without_text.suggestion.confidence_score,
                    "directional_scores": package_without_text.suggestion.directional_scores,
                    "regime": package_without_text.suggestion.regime_label,
                },
                "with_text": {
                    "composite_score": package_with_text.suggestion.composite_score,
                    "confidence": package_with_text.suggestion.confidence_score,
                    "directional_scores": package_with_text.suggestion.directional_scores,
                    "regime": package_with_text.suggestion.regime_label,
                    "explanation": package_with_text.suggestion.explanation_summary,
                },
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
