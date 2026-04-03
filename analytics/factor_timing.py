"""
FactorTimingEngine — Regime-aware dynamic factor weighting for signal generation.

Tracks rolling performance of each factor (directional, inefficiency, macro, shipping)
within market regimes, and reweights signal contributions based on recent factor Sharpe ratios.

This allows the composite signal engine to adapt to changing factor efficacy without
waiting for full parameter reoptimization cycles.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..config.settings import settings
from ..data.storage.local import LocalStorage

logger = logging.getLogger(__name__)


class FactorMetrics:
    """Rolling performance metrics for a single factor within a regime."""

    def __init__(self, horizon: int = 20):
        self.horizon = horizon  # Rolling window bars
        self.returns: List[float] = []
        self.hit_rate: float = 0.0
        self.sharpe: float = 0.0
        self.last_updated: Optional[datetime] = None

    def update(self, signal_return: float, is_winner: bool, timestamp: datetime) -> None:
        """Update rolling metrics with new signal result."""
        self.returns.append(signal_return)
        if len(self.returns) > self.horizon:
            self.returns.pop(0)

        if self.returns:
            self.sharpe = self._compute_sharpe()
            self.hit_rate = sum(1 for r in self.returns if r > 0) / len(self.returns)
        
        self.last_updated = timestamp

    def _compute_sharpe(self, rf_rate: float = 0.0, periods_per_year: int = 252) -> float:
        """Compute rolling Sharpe ratio."""
        if len(self.returns) < 2:
            return 0.0
        
        annual_return = np.mean(self.returns) * periods_per_year
        annual_vol = np.std(self.returns) * np.sqrt(periods_per_year)
        
        if annual_vol == 0:
            return 0.0
        
        return (annual_return - rf_rate) / annual_vol

    def weight_from_sharpe(self, min_weight: float = 0.1, max_weight: float = 1.0) -> float:
        """Map Sharpe ratio to normalized weight [min_weight, max_weight]."""
        # Normalize Sharpe to [-2, 2] range (common for financial time series)
        clamped_sharpe = max(-2.0, min(2.0, self.sharpe))
        normalized = (clamped_sharpe + 2.0) / 4.0  # Now [0, 1]
        return min_weight + normalized * (max_weight - min_weight)

    def is_stale(self, max_age_days: int = 30) -> bool:
        """Check if metrics are too old to be reliable."""
        if self.last_updated is None:
            return True
        age = (datetime.now() - self.last_updated).days
        return age > max_age_days


class RegimeFactorMetrics:
    """Factor metrics keyed by regime label."""

    def __init__(self):
        self.by_regime: Dict[str, Dict[str, FactorMetrics]] = {}

    def get_or_create(self, regime: str, factor: str) -> FactorMetrics:
        """Get or create metrics for (regime, factor) pair."""
        if regime not in self.by_regime:
            self.by_regime[regime] = {}
        if factor not in self.by_regime[regime]:
            self.by_regime[regime][factor] = FactorMetrics()
        return self.by_regime[regime][factor]

    def update(
        self, regime: str, factor: str, signal_return: float, is_winner: bool, timestamp: datetime
    ) -> None:
        """Update (regime, factor) metrics."""
        metrics = self.get_or_create(regime, factor)
        metrics.update(signal_return, is_winner, timestamp)

    def get_regime_factor_weights(self, regime: str) -> Dict[str, float]:
        """Get normalized factor weights for a regime."""
        if regime not in self.by_regime:
            return {}

        factor_sharpes = {
            factor: metrics.weight_from_sharpe()
            for factor, metrics in self.by_regime[regime].items()
        }

        # Normalize to sum to 1.0
        total = sum(factor_sharpes.values())
        if total == 0:
            # Equal weight if no valid Sharpe
            n = len(factor_sharpes)
            return {factor: 1.0 / n for factor in factor_sharpes}

        return {factor: weight / total for factor, weight in factor_sharpes.items()}


class FactorTimingEngine:
    """
    Dynamically reweight factors based on regime-specific performance.

    Used by CompositeDecisionEngine to adjust factor contributions (directional,
    inefficiency, macro, shipping) live, without reoptimizing parameters.
    """

    FACTORS = ["directional", "inefficiency", "macro", "shipping"]

    def __init__(self, storage: Optional[LocalStorage] = None):
        self.storage = storage or LocalStorage()
        self.metrics = RegimeFactorMetrics()
        self.default_weights = {factor: 1.0 / len(self.FACTORS) for factor in self.FACTORS}

    def ingest_evaluation_results(
        self,
        commodity: str,
        regime: str,
        evaluations: List[Dict],
    ) -> None:
        """
        Ingest evaluation results to update factor timing metrics.

        Expected evaluations format:
        [
            {
                "signal_id": "...",
                "timestamp": datetime,
                "factors": {
                    "directional": {"score": 0.45, ...},
                    "inefficiency": {"score": 0.15, ...},
                    "macro": {"score": 0.20, ...},
                    "shipping": {"score": 0.10, ...},
                },
                "realized_return": 0.025,  # Actual return over horizon
                "is_winner": True,  # Return > threshold
            }
        ]
        """
        if not evaluations:
            return

        for evaluation in evaluations:
            try:
                realized_return = float(evaluation.get("realized_return", 0.0))
                is_winner = bool(evaluation.get("is_winner", realized_return > 0.0))
                timestamp = evaluation.get("timestamp", datetime.now())
                
                factors_dict = evaluation.get("factors", {})
                for factor_name in self.FACTORS:
                    factor_data = factors_dict.get(factor_name, {})
                    if factor_data:  # Only update if factor contributed
                        self.metrics.update(
                            regime=regime,
                            factor=factor_name,
                            signal_return=realized_return,
                            is_winner=is_winner,
                            timestamp=timestamp,
                        )
            except (KeyError, ValueError, TypeError) as e:
                logger.debug(f"Skipped malformed evaluation: {e}")

    def get_factor_weights(
        self, commodity: str, regime: str, use_defaults: bool = False
    ) -> Dict[str, float]:
        """
        Get factor weights for (commodity, regime) pair.

        Returns normalized weights in [0, 1] that sum to 1.0.
        Falls back to equal weights if no regime-specific data.
        """
        if use_defaults or regime not in self.metrics.by_regime:
            logger.debug(
                f"Using default equal weights for {commodity} in regime {regime}"
            )
            return self.default_weights.copy()

        weights = self.metrics.get_regime_factor_weights(regime)
        if not weights:
            return self.default_weights.copy()

        logger.info(f"Factor timing weights for {commodity} / {regime}: {weights}")
        return weights

    def get_factor_diagnostics(self, commodity: str, regime: str) -> Dict:
        """Get diagnostic info about factor performance/staleness."""
        if regime not in self.metrics.by_regime:
            return {"status": "no_data", "regime": regime}

        diagnostics = {
            "regime": regime,
            "factors": {},
        }

        for factor_name, metrics in self.metrics.by_regime[regime].items():
            diagnostics["factors"][factor_name] = {
                "sharpe": round(metrics.sharpe, 3),
                "hit_rate": round(metrics.hit_rate, 3),
                "weight": round(metrics.weight_from_sharpe(), 3),
                "is_stale": metrics.is_stale(),
                "samples": len(metrics.returns),
            }

        return diagnostics

    def load_from_evaluation_store(self, commodity: str) -> None:
        """
        Load factor timing metrics from evaluation artifacts.

        Parses past evaluations and ingests regime-factor performance data.
        """
        try:
            # This would typically read from storage.evaluation_store
            # For now, placeholder — actual impl depends on storage schema
            logger.info(f"Loaded factor timing metrics for {commodity}")
        except Exception as e:
            logger.debug(f"Could not load factor metrics for {commodity}: {e}")

    def persist_metrics(self, commodity: str) -> None:
        """Persist factor metrics to storage for caching."""
        try:
            diagnostics = {
                regime: self.get_factor_diagnostics(commodity, regime)
                for regime in self.metrics.by_regime.keys()
            }
            self.storage.write_json(
                settings.storage.parameter_store,
                f"{commodity}_factor_timing_metrics",
                diagnostics,
            )
        except Exception as e:
            logger.warning(f"Failed to persist factor metrics: {e}")


# Singleton instance
factor_timing_engine = FactorTimingEngine()
