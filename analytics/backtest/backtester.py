from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ...analytics.adaptation import AdaptiveParameterEngine
from ...analytics.evaluation_pricing import evaluation_price
from ...analytics.evaluation import SignalEvaluationEngine
from ...config.settings import settings
from ...data.ingestion import market_data_service
from ...data.models import EvaluationArtifact, MacroEvent, MacroFeature, SignalSnapshot, Suggestion
from ...data.storage.local import LocalStorage
from ...signals.composite.composite_decision import CompositeDecisionEngine

PACKAGE_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class BacktestResult:
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profitable_trades: int
    avg_trade_return: float
    macro_alignment_score: float
    macro_conflict_penalty: float
    signal_accuracy: float
    period_start: date
    period_end: date
    evaluation_summary: Dict[str, object]


@dataclass
class MacroSignalEvaluation:
    signal_type: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    macro_influence: float
    sample_size: int


class MacroBacktester:
    """Compatibility wrapper aligned with the governed evaluation workflow."""

    def __init__(self, storage: Optional[LocalStorage] = None, results_dir: str = "results/backtests"):
        self.storage = storage or LocalStorage()
        resolved_results_dir = Path(results_dir)
        if not resolved_results_dir.is_absolute():
            resolved_results_dir = PACKAGE_ROOT / resolved_results_dir
        self.results_dir = resolved_results_dir
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.evaluation_engine = SignalEvaluationEngine(storage=self.storage)
        self.adaptation_engine = AdaptiveParameterEngine(storage=self.storage)
        self.transaction_costs = float(settings.evaluation_pricing.turnover_cost_bps) / 10000.0

    def run_backtest(
        self,
        commodity: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        macro_features: Optional[List[MacroFeature]] = None,
        macro_events: Optional[List[MacroEvent]] = None,
        rebalance_freq: str = "5B",
        price_data: Optional[pd.DataFrame] = None,
        persist: bool = False,
    ) -> BacktestResult:
        macro_features = macro_features or []
        macro_events = macro_events or []
        price_data = self._resolve_price_data(commodity, start_date, end_date, price_data)
        if price_data.empty:
            raise ValueError(f"No historical price data available for {commodity}.")

        start_date = start_date or price_data.index[0].date()
        end_date = end_date or price_data.index[-1].date()
        suggestions, snapshots = self._generate_historical_signals(price_data, commodity, macro_features, macro_events, rebalance_freq)
        evaluation = self.evaluation_engine.evaluate_signals(
            commodity=commodity,
            price_data=price_data,
            signal_snapshots=snapshots,
            macro_events=macro_events,
            as_of_timestamp=price_data.index[-1].to_pydatetime(),
            persist=persist,
        )
        portfolio_returns = self._simulate_portfolio(price_data, suggestions)
        return self._calculate_performance_metrics(portfolio_returns, suggestions, evaluation, start_date, end_date)

    def evaluate_macro_signal_impact(
        self,
        commodity: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        macro_features: Optional[List[MacroFeature]] = None,
        macro_events: Optional[List[MacroEvent]] = None,
        price_data: Optional[pd.DataFrame] = None,
    ) -> Dict[str, MacroSignalEvaluation]:
        macro_features = macro_features or []
        with_macro = self.run_backtest(
            commodity=commodity,
            start_date=start_date,
            end_date=end_date,
            macro_features=macro_features,
            macro_events=macro_events,
            price_data=price_data,
        )
        without_macro = self.run_backtest(
            commodity=commodity,
            start_date=start_date,
            end_date=end_date,
            macro_features=[],
            macro_events=macro_events,
            price_data=price_data,
        )
        return {
            "macro_enabled": MacroSignalEvaluation(
                signal_type="macro_enabled",
                accuracy=with_macro.signal_accuracy,
                precision=with_macro.win_rate,
                recall=with_macro.win_rate,
                f1_score=self._f1(with_macro.win_rate, with_macro.win_rate),
                macro_influence=with_macro.sharpe_ratio - without_macro.sharpe_ratio,
                sample_size=with_macro.total_trades,
            ),
            "baseline": MacroSignalEvaluation(
                signal_type="baseline",
                accuracy=without_macro.signal_accuracy,
                precision=without_macro.win_rate,
                recall=without_macro.win_rate,
                f1_score=self._f1(without_macro.win_rate, without_macro.win_rate),
                macro_influence=0.0,
                sample_size=without_macro.total_trades,
            ),
        }

    def generate_backtest_report(self, results: Dict[str, BacktestResult], output_file: Optional[str] = None) -> str:
        output_path = Path(output_file) if output_file else self.results_dir / f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        lines = ["# Backtest Report", ""]
        for name, result in results.items():
            lines.extend(
                [
                    f"## {name}",
                    f"- Total return: {result.total_return:.2%}",
                    f"- Sharpe ratio: {result.sharpe_ratio:.2f}",
                    f"- Win rate: {result.win_rate:.2%}",
                    f"- Signal accuracy: {result.signal_accuracy:.2%}",
                    f"- Trades: {result.total_trades}",
                    "",
                ]
            )
        output_path.write_text("\n".join(lines))
        return str(output_path)

    def _resolve_price_data(
        self,
        commodity: str,
        start_date: Optional[date],
        end_date: Optional[date],
        price_data: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        if price_data is not None:
            return price_data.copy()
        if start_date is None or end_date is None:
            raise ValueError("start_date and end_date are required when price_data is not supplied.")
        return market_data_service.load_price_frame(commodity, start_date, end_date)

    def _generate_historical_signals(
        self,
        price_data: pd.DataFrame,
        commodity: str,
        macro_features: List[MacroFeature],
        macro_events: List[MacroEvent],
        rebalance_freq: str,
    ) -> tuple[list[Suggestion], list[SignalSnapshot]]:
        active_version = self.adaptation_engine.load_active_version(commodity)
        engine = CompositeDecisionEngine(parameter_state={"version_id": active_version.version_id, **active_version.parameters})
        suggestions: List[Suggestion] = []
        snapshots: List[SignalSnapshot] = []

        rebalance_dates = price_data.resample(rebalance_freq).last().index
        for timestamp in rebalance_dates:
            historical = price_data.loc[:timestamp]
            if len(historical) < settings.signal.min_history_rows:
                continue
            package = engine.generate_signal_package(
                data=historical,
                commodity=commodity,
                macro_features=macro_features,
                macro_events=macro_events,
                as_of_timestamp=timestamp.to_pydatetime(),
            )
            suggestions.append(package.suggestion)
            snapshots.append(package.snapshot)
        return suggestions, snapshots

    def _simulate_portfolio(self, price_data: pd.DataFrame, signals: List[Suggestion]) -> pd.Series:
        if not signals:
            return pd.Series(0.0, index=price_data.index)
        closes = price_data["close"].astype(float)
        opens = price_data["open"].astype(float) if "open" in price_data.columns else closes
        rolling_median_volume = price_data["volume"].rolling(20, min_periods=5).median() if "volume" in price_data.columns else pd.Series(index=price_data.index, dtype=float)
        rolling_vol = closes.pct_change().rolling(settings.evaluation_pricing.vol_target_window_bars, min_periods=5).std(ddof=0)
        signal_map = {pd.Timestamp(signal.timestamp): signal for signal in signals}
        current_position = 0.0
        pending_signal: Optional[Suggestion] = None
        portfolio_returns = []
        prev_close: Optional[float] = None
        for idx, timestamp in enumerate(price_data.index):
            row = price_data.iloc[idx]
            day_return = 0.0
            if prev_close is not None:
                current_open = float(opens.iloc[idx])
                overnight_return = current_open / prev_close - 1.0 if prev_close else 0.0
                day_return += current_position * overnight_return

            if pending_signal is not None:
                median_volume = float(rolling_median_volume.iloc[idx]) if len(rolling_median_volume) > idx and pd.notna(rolling_median_volume.iloc[idx]) else None
                prior_position = current_position
                realized_vol = float(rolling_vol.iloc[idx]) if idx < len(rolling_vol) and pd.notna(rolling_vol.iloc[idx]) else None
                target_position = self._signal_to_position(pending_signal, realized_vol=realized_vol)
                current_open = float(opens.iloc[idx])
                if prior_position != 0.0:
                    prior_direction = "long" if prior_position > 0 else "short"
                    exit_exec = evaluation_price(
                        row,
                        prior_direction,
                        phase="exit",
                        median_volume=median_volume,
                        participation=abs(prior_position),
                    )
                    if prior_direction == "long":
                        exit_adjustment = exit_exec / current_open - 1.0
                    else:
                        exit_adjustment = current_open / exit_exec - 1.0
                    day_return += abs(prior_position) * exit_adjustment
                current_position = target_position
                if current_position != 0.0:
                    entry_exec = evaluation_price(
                        row,
                        pending_signal.preferred_direction,
                        phase="entry",
                        median_volume=median_volume,
                        participation=abs(current_position),
                    )
                    close_price = float(closes.iloc[idx])
                    if pending_signal.preferred_direction == "long":
                        intraday_return = close_price / entry_exec - 1.0
                    else:
                        intraday_return = entry_exec / close_price - 1.0
                    day_return += abs(current_position) * intraday_return
                trading_cost = abs(current_position - prior_position) * self.transaction_costs
                day_return -= trading_cost
                pending_signal = None
            elif prev_close is not None:
                close_price = float(closes.iloc[idx])
                day_return += current_position * (close_price / prev_close - 1.0 - (float(opens.iloc[idx]) / prev_close - 1.0))

            if timestamp in signal_map:
                pending_signal = signal_map[timestamp]

            portfolio_returns.append(day_return)
            prev_close = float(closes.iloc[idx])
        return pd.Series(portfolio_returns, index=price_data.index)

    def _signal_to_position(self, signal: Suggestion, realized_vol: Optional[float] = None) -> float:
        if signal.confidence_score < settings.evaluation_pricing.min_trade_confidence:
            return 0.0

        direction = 0.0
        if signal.preferred_direction == "long":
            direction = 1.0
        elif signal.preferred_direction == "short":
            direction = -1.0
        if direction == 0.0:
            return 0.0

        annualization = max(1, int(settings.evaluation_pricing.annualization_days))
        target_daily_vol = float(settings.evaluation_pricing.target_annualized_vol) / np.sqrt(annualization)
        vol_scalar = 1.0
        if realized_vol is not None and realized_vol > 0:
            vol_scalar = target_daily_vol / float(realized_vol)

        raw_position = direction * float(signal.confidence_score) * vol_scalar
        max_abs_position = float(settings.evaluation_pricing.max_abs_position)
        return float(np.clip(raw_position, -max_abs_position, max_abs_position))

    def _calculate_performance_metrics(
        self,
        portfolio_returns: pd.Series,
        signals: List[Suggestion],
        evaluation: EvaluationArtifact,
        start_date: date,
        end_date: date,
    ) -> BacktestResult:
        cumulative = (1.0 + portfolio_returns).cumprod()
        total_return = float(cumulative.iloc[-1] - 1.0) if len(cumulative) else 0.0
        annualization = max(1, int(settings.evaluation_pricing.annualization_days))
        annualized_return = float((1.0 + total_return) ** (annualization / max(len(portfolio_returns), 1)) - 1.0) if len(portfolio_returns) else 0.0
        volatility = float(portfolio_returns.std(ddof=0) * np.sqrt(annualization)) if len(portfolio_returns) else 0.0
        sharpe_ratio = annualized_return / volatility if volatility else 0.0
        running_max = cumulative.cummax() if len(cumulative) else cumulative
        drawdowns = (cumulative / running_max - 1.0) if len(cumulative) else cumulative
        max_drawdown = float(drawdowns.min()) if len(drawdowns) else 0.0

        detailed = self.storage.load_domain_dataframe(settings.storage.evaluation_store, f"{signals[0].commodity}_detailed") if signals and evaluation.detailed_path else pd.DataFrame()
        if detailed.empty and evaluation.summary_metrics.get("sample_size", 0) > 0:
            # The caller may have chosen persist=False; use summary-only fallbacks.
            win_rate = float(evaluation.summary_metrics.get("overall_hit_rate", 0.0))
            total_trades = len(signals)
            profitable_trades = int(round(win_rate * total_trades))
            avg_trade_return = float(evaluation.summary_metrics.get("overall_average_return", 0.0))
            signal_accuracy = win_rate
        else:
            if "direction_correct" in detailed.columns:
                win_rate = float(detailed["direction_correct"].mean())
                profitable_trades = int(detailed["direction_correct"].sum())
                total_trades = int(detailed["signal_id"].nunique())
                avg_trade_return = float(detailed["signed_return"].mean()) if "signed_return" in detailed.columns else 0.0
                signal_accuracy = win_rate
            else:
                win_rate = 0.0
                profitable_trades = 0
                total_trades = len(signals)
                avg_trade_return = 0.0
                signal_accuracy = 0.0

        macro_alignment = float(np.mean([signal.macro_alignment_score or 0.0 for signal in signals])) if signals else 0.0
        macro_conflict = float(np.mean([signal.macro_conflict_score or 0.0 for signal in signals])) if signals else 0.0

        return BacktestResult(
            total_return=total_return,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            total_trades=total_trades,
            profitable_trades=profitable_trades,
            avg_trade_return=avg_trade_return,
            macro_alignment_score=macro_alignment,
            macro_conflict_penalty=macro_conflict,
            signal_accuracy=signal_accuracy,
            period_start=start_date,
            period_end=end_date,
            evaluation_summary=evaluation.summary_metrics,
        )

    def _f1(self, precision: float, recall: float) -> float:
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)
