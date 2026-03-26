from __future__ import annotations

from typing import Dict


def event_regime_adjustment(event_features: Dict[str, float]) -> float:
    return float(
        max(
            -0.25,
            min(
                0.25,
                event_features.get("persistent_trend_event_score", 0.0) * 0.15
                + event_features.get("regime_shift_probability_proxy", 0.0) * 0.20
                - event_features.get("macro_headwind_score", 0.0) * 0.10,
            ),
        )
    )
