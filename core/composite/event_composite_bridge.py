from __future__ import annotations

from typing import Dict


def event_risk_penalty(event_features: Dict[str, float]) -> float:
    return float(
        max(
            0.0,
            min(
                0.8,
                event_features.get("uncertainty_penalty", 0.0) * 0.30
                + event_features.get("event_volatility_risk_score", 0.0) * 0.25
                + event_features.get("geopolitics_risk_score", 0.0) * 0.20,
            ),
        )
    )
