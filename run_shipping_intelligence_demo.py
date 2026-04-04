from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sys

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from commodities_quant_engine.main import build_demo_price_data
from commodities_quant_engine.shipping import ShippingFeaturePipeline
from commodities_quant_engine.workflow import ResearchWorkflow


def build_demo_shipping_positions() -> pd.DataFrame:
    base = datetime(2026, 3, 1, 0, 0)
    rows = []
    for day in range(18):
        timestamp = base + timedelta(days=day)
        congestion_boost = 0.02 * max(0, day - 10)
        rows.extend(
            [
                {
                    "vessel_id": "tanker-a",
                    "timestamp": timestamp,
                    "latitude": 25.16 + congestion_boost,
                    "longitude": 56.31 + congestion_boost,
                    "speed_knots": max(0.3, 1.2 - congestion_boost * 10),
                    "cargo_class": "crude_tanker",
                },
                {
                    "vessel_id": "tanker-b",
                    "timestamp": timestamp + timedelta(hours=8),
                    "latitude": 26.10,
                    "longitude": 56.90,
                    "speed_knots": max(6.0, 12.0 - congestion_boost * 30),
                    "cargo_class": "crude_tanker",
                },
                {
                    "vessel_id": "tanker-c",
                    "timestamp": timestamp + timedelta(hours=16),
                    "latitude": 25.18 + congestion_boost,
                    "longitude": 56.33 + congestion_boost,
                    "speed_knots": max(0.3, 1.0 - congestion_boost * 8),
                    "cargo_class": "crude_tanker",
                },
            ]
        )
    return pd.DataFrame(rows)


def build_demo_route_events() -> pd.DataFrame:
    dates = pd.date_range("2026-03-10", periods=8, freq="D")
    return pd.DataFrame(
        {
            "route_id": ["GULF_TO_SINGAPORE"] * len(dates),
            "timestamp": dates,
            "event_type": ["rerouting"] * len(dates),
            "severity": [0.25, 0.25, 0.30, 0.45, 0.55, 0.65, 0.60, 0.50],
            "detour_distance_ratio": [0.0, 0.0, 0.05, 0.15, 0.25, 0.35, 0.30, 0.20],
        }
    )


def main() -> None:
    price_data = build_demo_price_data(periods=280)
    shipping_vectors = ShippingFeaturePipeline().run(
        commodity="CRUDEOIL",
        vessel_positions=build_demo_shipping_positions(),
        route_events=build_demo_route_events(),
        as_of_timestamp=price_data.index[-1].to_pydatetime(),
    )
    package = ResearchWorkflow().run_signal_cycle(
        commodity="CRUDEOIL",
        price_data=price_data,
        shipping_feature_vectors=shipping_vectors,
        as_of_timestamp=price_data.index[-1].to_pydatetime(),
        persist_snapshot=False,
        persist_report=False,
    )
    print(package.suggestion.to_markdown())


if __name__ == "__main__":
    main()
