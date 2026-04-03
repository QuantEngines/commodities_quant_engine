from __future__ import annotations

from typing import Dict

import pandas as pd


class PortfolioDashboard:
    """Build dashboard-ready payload from portfolio cycle artifacts."""

    @staticmethod
    def build_payload(
        portfolio_budget: float,
        portfolio_weights: Dict[str, float],
        sector_exposures: Dict[str, float],
        suggestions_frame: pd.DataFrame,
    ) -> Dict[str, object]:
        gross_weight = float(sum(abs(v) for v in portfolio_weights.values()))
        active_positions = int((suggestions_frame["direction"] != "flat").sum()) if not suggestions_frame.empty else 0
        avg_conf = (
            float(suggestions_frame["confidence_score"].mean())
            if (not suggestions_frame.empty and "confidence_score" in suggestions_frame.columns)
            else 0.0
        )
        total_notional = (
            float(suggestions_frame["target_notional"].sum())
            if (not suggestions_frame.empty and "target_notional" in suggestions_frame.columns)
            else 0.0
        )

        return {
            "portfolio_budget": float(portfolio_budget),
            "gross_weight": gross_weight,
            "active_positions": active_positions,
            "avg_confidence": avg_conf,
            "suggested_total_notional": total_notional,
            "utilization_pct": (total_notional / float(portfolio_budget)) if portfolio_budget > 0 else 0.0,
            "sector_exposures": dict(sorted(sector_exposures.items(), key=lambda kv: kv[1], reverse=True)),
            "top_weights": dict(sorted(portfolio_weights.items(), key=lambda kv: kv[1], reverse=True)[:10]),
        }
