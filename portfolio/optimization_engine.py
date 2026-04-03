"""
PortfolioOptimizationEngine — Joint commodity signal weighting and portfolio construction.

Converts individual commodity signals into portfolio-level allocations using
Markowitz-style optimization with:
- Commodity correlation awareness
- Portfolio-level exposure constraints
- Sector/family clustering (bullion, base metals, energy, agri)
- Dynamic risk budgeting

Output: Portfolio-weighted signal strengths per commodity.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ..config.commodity_universe import default_mcx_commodity_definitions
from ..config.settings import settings
from ..data.storage.local import LocalStorage

logger = logging.getLogger(__name__)


class CommodityCovarianceEstimator:
    """Estimates inter-commodity correlation matrix from price data."""

    def __init__(self, lookback_periods: int = 60):
        self.lookback_periods = lookback_periods

    def estimate_covariance(
        self, prices_by_commodity: Dict[str, np.ndarray]
    ) -> pd.DataFrame:
        """
        Estimate covariance matrix from price series.

        Args:
            prices_by_commodity: Dict[commodity_symbol, price_array]

        Returns:
            Covariance matrix as DataFrame
        """
        if not prices_by_commodity:
            return pd.DataFrame()

        # Compute returns
        returns_data = {}
        for commodity, prices in prices_by_commodity.items():
            if len(prices) < 2:
                continue
            returns = np.diff(np.log(prices[-self.lookback_periods :]))
            returns_data[commodity] = returns

        if not returns_data:
            return pd.DataFrame()

        df_returns = pd.DataFrame(returns_data)
        return df_returns.cov()

    def impute_missing_correlations(
        self, cov_matrix: pd.DataFrame, commodity_families: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Impute missing correlations using sector averages.

        For commodities without price history, infer correlation
        from sector-level average correlations.
        """
        if cov_matrix.empty:
            return cov_matrix

        imputed = cov_matrix.copy()

        # Compute sector-level average correlations
        family_correlations: Dict[str, float] = {}
        all_commodities = list(cov_matrix.index)

        for family in set(commodity_families.values()):
            family_commodities = [
                c for c in all_commodities
                if commodity_families.get(c) == family
            ]
            if len(family_commodities) > 1:
                subset = cov_matrix.loc[family_commodities, family_commodities]
                # Average off-diagonal correlation
                n = len(subset)
                off_diag = (
                    (subset.values.sum() - np.trace(subset)) / (n * (n - 1))
                    if n > 1
                    else 0.5
                )
                family_correlations[family] = max(0.1, min(0.9, off_diag))

        # Default: cross-family correlation lower than within-family
        default_cross_family_cor = 0.3
        default_within_family_cor = family_correlations.get("energy", 0.5)

        # Impute missing rows/cols
        for i, comm_i in enumerate(all_commodities):
            for j, comm_j in enumerate(all_commodities):
                if pd.isna(imputed.iloc[i, j]) or imputed.iloc[i, j] == 0:
                    if i == j:
                        imputed.iloc[i, j] = 1.0
                    else:
                        family_i = commodity_families.get(comm_i, "unknown")
                        family_j = commodity_families.get(comm_j, "unknown")
                        cor = (
                            default_within_family_cor
                            if family_i == family_j
                            else default_cross_family_cor
                        )
                        imputed.iloc[i, j] = cor
                        imputed.iloc[j, i] = cor

        return imputed


class PortfolioOptimizer:
    """Solves Markowitz portfolio optimization with constraints."""

    def __init__(self):
        self.min_weight = 0.0  # Allow zero weight
        self.max_weight = 0.5  # No single commodity >50%
        self.max_sector_weight = 0.6  # No sector >60%
        self.target_leverage = 1.0  # Fully invested

    def optimize_weights(
        self,
        expected_returns: Dict[str, float],
        cov_matrix: pd.DataFrame,
        commodity_families: Dict[str, str],
        constraints: Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        Optimize portfolio weights using Markowitz mean-variance framework.

        Args:
            expected_returns: Dict[commodity, expected_signal_score]
            cov_matrix: Covariance matrix of returns
            commodity_families: Dict[commodity, family_name]
            constraints: Optional override constraints

        Returns:
            Dict[commodity, optimized_weight]
        """
        commodities = list(expected_returns.keys())
        if not commodities or cov_matrix.empty:
            # Equal weight fallback
            return {c: 1.0 / len(commodities) for c in commodities}

        n = len(commodities)
        expected_ret_array = np.array([expected_returns.get(c, 0.0) for c in commodities])

        # Align covariance matrix to commodities
        cov_aligned = cov_matrix.loc[commodities, commodities].fillna(0)
        cov_array = cov_aligned.values + np.eye(n) * 1e-6  # Add regularization

        def portfolio_return(w: np.ndarray) -> float:
            return -np.dot(w, expected_ret_array)  # Negative because we minimize

        def portfolio_variance(w: np.ndarray) -> float:
            return np.sqrt(np.dot(w, cov_array @ w))

        def portfolio_sharpe(w: np.ndarray) -> float:
            """Minimize negative Sharpe ratio."""
            ret = np.dot(w, expected_ret_array)
            vol = np.sqrt(np.dot(w, cov_array @ w)) + 1e-8
            return -ret / vol if ret > 0 else vol  # Return vol if negative return

        # Constraints
        constraints_list = [
            {"type": "eq", "fun": lambda w: np.sum(w) - self.target_leverage}
        ]

        # Sector constraints
        for family in set(commodity_families.values()):
            family_indices = [
                i for i, c in enumerate(commodities)
                if commodity_families.get(c) == family
            ]

            def sector_constraint(w: np.ndarray, indices=family_indices) -> float:
                return self.max_sector_weight - np.sum(w[indices])

            constraints_list.append({"type": "ineq", "fun": sector_constraint})

        # Bounds: [min_weight, max_weight] per commodity
        bounds = [(self.min_weight, self.max_weight) for _ in range(n)]

        # Initial guess: equal weight
        x0 = np.array([1.0 / n] * n)

        try:
            result = minimize(
                portfolio_sharpe,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints_list,
                options={"ftol": 1e-9, "maxiter": 1000},
            )

            if result.success and np.abs(np.sum(result.x) - self.target_leverage) < 1e-3:
                weights = result.x
            else:
                logger.warning(f"Optimization failed: {result.message}; using equal weights")
                weights = x0
        except Exception as e:
            logger.error(f"Optimization error: {e}; using equal weights")
            weights = x0

        return {c: float(w) for c, w in zip(commodities, weights)}


class PortfolioOptimizationEngine:
    """
    Converts commodity signals into portfolio-level allocations.

    Acts as a portfolio-layer wrapper around individual signal generation,
    enabling cross-commodity capital allocation and risk budgeting.
    """

    def __init__(self, storage: Optional[LocalStorage] = None):
        self.storage = storage or LocalStorage()
        self.cov_estimator = CommodityCovarianceEstimator(lookback_periods=60)
        self.optimizer = PortfolioOptimizer()
        self.commodity_definitions = default_mcx_commodity_definitions()
        self.commodity_families = self._build_commodity_families()

    def _build_commodity_families(self) -> Dict[str, str]:
        """Map commodity -> sector family (bullion, base_metals, energy, agri)."""
        families = {}
        for commodity, defn in self.commodity_definitions.items():
            # commodity_universe.py stores the field as "segment", not "family"
            major_family = defn.get("segment", defn.get("family", "other"))
            families[commodity] = major_family
        return families

    def optimize_commodity_weights(
        self,
        commodity_signals: Dict[str, float],
        price_history: Optional[Dict[str, np.ndarray]] = None,
    ) -> Dict[str, float]:
        """
        Optimize cross-commodity portfolio weights.

        Args:
            commodity_signals: Dict[commodity, signal_score] normalized to [0, 1]
            price_history: Optional Dict[commodity, price_array] for covariance estimation

        Returns:
            Dict[commodity, portfolio_weight] normalized to sum to 1.0
        """
        if not commodity_signals:
            return {}

        # Estimate covariance
        if price_history:
            cov_matrix = self.cov_estimator.estimate_covariance(price_history)
        else:
            cov_matrix = pd.DataFrame()

        # Impute missing correlations
        cov_matrix = self.cov_estimator.impute_missing_correlations(
            cov_matrix, self.commodity_families
        )

        # Map signal scores to expected returns (normalize)
        max_signal = max(commodity_signals.values()) if commodity_signals.values() else 1.0
        expected_returns = {
            c: (s / max_signal) if max_signal > 0 else 0.0
            for c, s in commodity_signals.items()
        }

        # Run optimization
        optimized_weights = self.optimizer.optimize_weights(
            expected_returns=expected_returns,
            cov_matrix=cov_matrix,
            commodity_families=self.commodity_families,
        )

        logger.info(f"Portfolio optimization: {optimized_weights}")
        return optimized_weights

    def get_portfolio_signal_allocation(
        self,
        signals_by_commodity: Dict[str, Dict],
    ) -> Dict[str, Dict]:
        """
        Convert individual commodity signals into portfolio-weighted outputs.

        Args:
            signals_by_commodity: Dict[commodity, signal_dict]
                where signal_dict includes 'score', 'confidence', etc.

        Returns:
            Dict[commodity, portfolio_adjusted_signal]
                with 'weight' and 'portfolio_adjusted_score' added
        """
        commodity_scores = {
            c: float(sig.get("score", 0.0))
            for c, sig in signals_by_commodity.items()
        }

        # Optimize weights
        weights = self.optimize_commodity_weights(commodity_scores)

        # Apply portfolio weights to signals
        result = {}
        for commodity, signal in signals_by_commodity.items():
            weight = weights.get(commodity, 0.0)
            adjusted_signal = {
                **signal,
                "portfolio_weight": weight,
                "portfolio_adjusted_score": float(signal.get("score", 0.0)) * weight,
            }
            result[commodity] = adjusted_signal

        return result

    def get_sector_exposures(
        self, portfolio_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Aggregate portfolio weights by family (sector).

        Returns:
            Dict[family_name, total_exposure]
        """
        sector_exposures = {}
        for commodity, weight in portfolio_weights.items():
            family = self.commodity_families.get(commodity, "other")
            sector_exposures[family] = sector_exposures.get(family, 0.0) + weight

        return sector_exposures


# Singleton instance
portfolio_optimization_engine = PortfolioOptimizationEngine()
