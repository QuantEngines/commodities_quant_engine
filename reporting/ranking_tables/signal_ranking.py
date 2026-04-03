from __future__ import annotations

from typing import Dict

import pandas as pd


class SignalRankingTable:
    """Build sortable ranking tables for portfolio-cycle suggestions."""

    @staticmethod
    def build(suggestions: Dict[str, Dict[str, object]]) -> pd.DataFrame:
        rows = []
        for commodity, payload in suggestions.items():
            rows.append(
                {
                    "commodity": commodity,
                    "direction": payload.get("direction", "flat"),
                    "target_weight": float(payload.get("target_weight", 0.0) or 0.0),
                    "signal_score": float(payload.get("signal_score", 0.0) or 0.0),
                    "confidence_score": float(payload.get("confidence_score", 0.0) or 0.0),
                    "target_notional": float(payload.get("target_notional", 0.0) or 0.0),
                }
            )
        if not rows:
            return pd.DataFrame(
                columns=[
                    "commodity",
                    "direction",
                    "target_weight",
                    "signal_score",
                    "confidence_score",
                    "target_notional",
                ]
            )
        frame = pd.DataFrame(rows)
        return frame.sort_values(["signal_score", "confidence_score"], ascending=False).reset_index(drop=True)

    @staticmethod
    def to_markdown(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "No portfolio suggestions available."
        return frame.to_markdown(index=False)
