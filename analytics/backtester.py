"""
HistoricalBacktester — Backtests signal snapshots against realized outcomes.

Computes per-factor contribution to signal returns, accumulates regime-aware
Sharpe ratios, and feeds FactorTimingEngine to enable true factor timing.

Output: Factor metrics by regime for live dynamic reweighting.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..analytics.evaluation import SignalEvaluationEngine
from ..analytics.factor_timing import FactorTimingEngine
from ..config.settings import settings
from ..data.storage.local import LocalStorage

logger = logging.getLogger(__name__)


class BacktestEvaluation:
    """Result of backtesting a single signal snapshot."""

    def __init__(
        self,
        signal_id: str,
        commodity: str,
        regime: str,
        signal_timestamp: datetime,
        signal_scores: Dict[str, float],  # {'directional': 0.45, 'inefficiency': 0.15, ...}
        realized_return: float,
        horizon_days: int,
        is_winner: bool,
    ):
        self.signal_id = signal_id
        self.commodity = commodity
        self.regime = regime
        self.signal_timestamp = signal_timestamp
        self.signal_scores = signal_scores
        self.realized_return = realized_return
        self.horizon_days = horizon_days
        self.is_winner = is_winner

    def to_dict(self) -> Dict:
        """Serialize to dict for storage/analysis."""
        return {
            "signal_id": self.signal_id,
            "commodity": self.commodity,
            "regime": self.regime,
            "signal_timestamp": self.signal_timestamp.isoformat(),
            "factors": self.signal_scores,
            "realized_return": self.realized_return,
            "horizon_days": self.horizon_days,
            "is_winner": self.is_winner,
        }


class HistoricalBacktester:
    """
    Backtests signal snapshots against actual price movements.

    Workflow:
    1. Load signal snapshots from historical evaluation store
    2. For each snapshot, compute realized return over horizon days
    3. Decompose return attribution to each factor
    4. Accumulate returns by (regime, factor)
    5. Feed FactorTimingEngine for live reweighting
    """

    def __init__(self, storage: Optional[LocalStorage] = None):
        self.storage = storage or LocalStorage()
        self.factor_timing_engine = FactorTimingEngine(storage=storage)
        self._eval_engine = SignalEvaluationEngine(storage=self.storage)

    def backtest_commodity_historical(
        self,
        commodity: str,
        price_data: pd.DataFrame,
        signal_snapshots: Optional[List[Dict]] = None,
        horizons: Optional[List[int]] = None,
    ) -> Dict[str, List[BacktestEvaluation]]:
        """
        Backtest all available signals for a commodity.

        Args:
            commodity: Commodity symbol
            price_data: DataFrame with OHLCV indexed by timestamp
            signal_snapshots: List of signal snapshot dicts (loaded from storage if None)
            horizons: Evaluation horizons in days (default: [1, 3, 5, 10, 20])

        Returns:
            Dict[regime, list of BacktestEvaluation]
        """
        horizons = horizons or [1, 3, 5, 10, 20]

        if signal_snapshots is None:
            signal_snapshots = self._load_commodity_snapshots(commodity)
            if not signal_snapshots:
                logger.warning(f"No signal snapshots found for {commodity}")
                return {}

        if price_data.empty:
            logger.warning(f"Empty price data for {commodity}")
            return {}

        evaluations_by_regime: Dict[str, List[BacktestEvaluation]] = {}

        for snapshot in signal_snapshots:
            try:
                snapshot_ts = pd.Timestamp(snapshot.get("timestamp"))
                regime = snapshot.get("regime_label", "unknown")

                # Find snapshot index in price data
                try:
                    snapshot_idx = price_data.index.get_loc(snapshot_ts, method="nearest")
                except Exception:
                    continue

                signal_scores = self._extract_factor_scores(snapshot)

                # Evaluate across multiple horizons
                for horizon in horizons:
                    future_idx = snapshot_idx + horizon
                    if future_idx >= len(price_data):
                        continue

                    future_close = price_data.iloc[future_idx]["close"]
                    current_close = price_data.iloc[snapshot_idx]["close"]
                    realized_return = (future_close - current_close) / current_close

                    is_winner = realized_return > 0  # Simple threshold

                    evaluation = BacktestEvaluation(
                        signal_id=snapshot.get("signal_id", f"{commodity}_{snapshot_ts}"),
                        commodity=commodity,
                        regime=regime,
                        signal_timestamp=snapshot_ts.to_pydatetime(),
                        signal_scores=signal_scores,
                        realized_return=realized_return,
                        horizon_days=horizon,
                        is_winner=is_winner,
                    )

                    if regime not in evaluations_by_regime:
                        evaluations_by_regime[regime] = []
                    evaluations_by_regime[regime].append(evaluation)

            except Exception as e:
                logger.debug(f"Skipped snapshot: {e}")

        # Ingest into FactorTimingEngine
        for regime, evals in evaluations_by_regime.items():
            eval_dicts = [e.to_dict() for e in evals]
            self.factor_timing_engine.ingest_evaluation_results(
                commodity=commodity,
                regime=regime,
                evaluations=eval_dicts,
            )

        logger.info(
            f"Backtested {sum(len(e) for e in evaluations_by_regime.values())} "
            f"signals for {commodity} across {len(evaluations_by_regime)} regimes"
        )
        return evaluations_by_regime

    def backtest_multiple_commodities(
        self,
        commodities: List[str],
        price_data_by_commodity: Dict[str, pd.DataFrame],
    ) -> Dict[str, Dict[str, List[BacktestEvaluation]]]:
        """
        Backtest multiple commodities in parallel.

        Returns:
            Dict[commodity, Dict[regime, list of BacktestEvaluation]]
        """
        results = {}
        for commodity in commodities:
            if commodity not in price_data_by_commodity:
                logger.warning(f"No price data for {commodity}")
                continue

            results[commodity] = self.backtest_commodity_historical(
                commodity=commodity,
                price_data=price_data_by_commodity[commodity],
            )

        return results

    def _load_commodity_snapshots(self, commodity: str) -> List[Dict]:
        """Load signal snapshots from the evaluation store via SignalEvaluationEngine."""
        try:
            snapshots = self._eval_engine.load_signal_snapshots(commodity)
            return [
                {
                    "signal_id": s.signal_id,
                    "timestamp": s.timestamp,
                    "regime_label": s.regime_label,
                    "directional_scores": s.directional_scores,
                    "inefficiency_score": s.inefficiency_score,
                    "composite_score": s.composite_score,
                    "component_scores": s.component_scores,
                    "direction": s.direction,
                    "conviction": s.conviction,
                }
                for s in snapshots
            ]
        except Exception as e:
            logger.debug(f"Could not load snapshots for {commodity}: {e}")
            return []

    def _extract_factor_scores(self, snapshot: Dict) -> Dict[str, float]:
        """Extract factor contribution scores from snapshot."""
        # Snapshot includes directional_scores, inefficiency_score, macro adjustments, etc.
        # We need to extract/normalize them
        scores = {
            "directional": 0.0,
            "inefficiency": 0.0,
            "macro": 0.0,
            "shipping": 0.0,
        }

        # Parse snapshot structure (adjust based on your actual SignalSnapshot format)
        if isinstance(snapshot.get("directional_scores"), dict):
            # Average across horizons
            dir_scores = list(snapshot["directional_scores"].values())
            scores["directional"] = np.mean(dir_scores) if dir_scores else 0.0

        scores["inefficiency"] = snapshot.get("inefficiency_score", 0.0)

        # Macro and shipping from component contributions
        component_scores = snapshot.get("component_scores", {})
        scores["macro"] = component_scores.get("macro", 0.0)
        scores["shipping"] = component_scores.get("shipping", 0.0)

        # Normalize to [0, 1] or leave as is
        return scores

    def generate_backtest_report(
        self, evaluations_by_regime: Dict[str, List[BacktestEvaluation]], commodity: str
    ) -> pd.DataFrame:
        """
        Generate diagnostic report from backtest results.

        Returns:
            DataFrame with regime-level performance metrics
        """
        rows = []
        for regime, evals in evaluations_by_regime.items():
            if not evals:
                continue

            returns = [e.realized_return for e in evals]
            hit_rate = sum(1 for e in evals if e.is_winner) / len(evals)
            mean_return = np.mean(returns)
            sharpe = mean_return / (np.std(returns) + 1e-8) * np.sqrt(252)

            rows.append(
                {
                    "commodity": commodity,
                    "regime": regime,
                    "n_signals": len(evals),
                    "hit_rate": hit_rate,
                    "mean_return": mean_return,
                    "sharpe": sharpe,
                    "std_return": np.std(returns),
                }
            )

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def persist_backtest_results(
        self, commodity: str, evaluations_by_regime: Dict[str, List[BacktestEvaluation]]
    ) -> None:
        """Store backtest results for later analysis."""
        try:
            payload = {
                "commodity": commodity,
                "timestamp": datetime.now().isoformat(),
                "regimes": {
                    regime: [e.to_dict() for e in evals]
                    for regime, evals in evaluations_by_regime.items()
                },
            }
            self.storage.write_json(
                settings.storage.evaluation_store,
                f"{commodity}_backtest_results",
                payload,
            )
        except Exception as e:
            logger.warning(f"Could not persist backtest: {e}")


# Singleton instance
historical_backtester = HistoricalBacktester()
