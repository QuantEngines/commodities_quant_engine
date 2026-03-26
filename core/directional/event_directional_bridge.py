from __future__ import annotations

from typing import Dict


def event_directional_adjustment(event_features: Dict[str, float]) -> float:
    return float(
        max(
            -0.35,
            min(
                0.35,
                event_features.get("supply_shock_score", 0.0) * 0.20
                + event_features.get("demand_strength_score", 0.0) * 0.15
                - event_features.get("demand_weakness_score", 0.0) * 0.15
                + event_features.get("inventory_signal_score", 0.0) * 0.10
                + event_features.get("macro_tailwind_score", 0.0) * 0.08
                - event_features.get("macro_headwind_score", 0.0) * 0.08,
            ),
        )
    )
