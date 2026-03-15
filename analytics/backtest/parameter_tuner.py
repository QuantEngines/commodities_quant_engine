from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from ...analytics.adaptation import AdaptiveParameterEngine
from ...config.settings import settings
from ...data.models import MacroEvent, MacroFeature
from ...data.storage.local import LocalStorage
from .backtester import BacktestResult, MacroBacktester

PACKAGE_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class ParameterSet:
    name: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]


@dataclass
class OptimizationResult:
    incumbent_parameters: ParameterSet
    candidate_parameters: Optional[ParameterSet]
    promoted: bool
    optimization_method: str
    evidence: Dict[str, Any]
    validation_results: Dict[str, BacktestResult]


class MacroParameterTuner:
    """Compatibility wrapper that now delegates to the governed adaptation engine."""

    def __init__(self, backtester: Optional[MacroBacktester] = None, storage: Optional[LocalStorage] = None):
        self.storage = storage or LocalStorage()
        self.backtester = backtester or MacroBacktester(storage=self.storage)
        self.adaptation_engine = AdaptiveParameterEngine(storage=self.storage)
        self.results_dir = PACKAGE_ROOT / "results/optimization"
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def optimize_parameters(
        self,
        commodity: str,
        train_start: Optional[date] = None,
        train_end: Optional[date] = None,
        validation_start: Optional[date] = None,
        validation_end: Optional[date] = None,
        macro_features: Optional[List[MacroFeature]] = None,
        macro_events: Optional[List[MacroEvent]] = None,
        optimization_method: str = "governed_recalibration",
        price_data: Optional[pd.DataFrame] = None,
        approve: bool = False,
        dry_run: bool = True,
    ) -> OptimizationResult:
        if price_data is None and (train_start is None or train_end is None):
            raise ValueError("Provide price_data or both train_start and train_end.")

        validation_results: Dict[str, BacktestResult] = {}
        backtest = self.backtester.run_backtest(
            commodity=commodity,
            start_date=train_start,
            end_date=train_end,
            macro_features=macro_features,
            macro_events=macro_events,
            price_data=price_data,
            persist=True,
        )
        validation_results["train"] = backtest

        decision = self.adaptation_engine.recommend_update(
            commodity=commodity,
            dry_run=dry_run,
            approve=approve,
        )

        incumbent = self.adaptation_engine.load_active_version(commodity)
        incumbent_set = ParameterSet(
            name=incumbent.version_id,
            parameters=incumbent.parameters,
            metrics=incumbent.metrics,
        )

        candidate_set = None
        if decision.candidate_version_id:
            candidate_payload = self.storage.read_json(f"{settings.storage.parameter_store}/{commodity}", decision.candidate_version_id)
            candidate_set = ParameterSet(
                name=str(candidate_payload.get("version_id", decision.candidate_version_id)),
                parameters=candidate_payload.get("parameters", {}),
                metrics=candidate_payload.get("metrics", {}),
            )

        if validation_start and validation_end:
            validation_results["validation"] = self.backtester.run_backtest(
                commodity=commodity,
                start_date=validation_start,
                end_date=validation_end,
                macro_features=macro_features,
                macro_events=macro_events,
                price_data=price_data.loc[str(validation_start) : str(validation_end)] if price_data is not None else None,
                persist=False,
            )

        result = OptimizationResult(
            incumbent_parameters=incumbent_set,
            candidate_parameters=candidate_set,
            promoted=decision.promoted,
            optimization_method=optimization_method,
            evidence=decision.evidence,
            validation_results=validation_results,
        )
        self._save_optimization_results(result, commodity)
        return result

    def _save_optimization_results(self, result: OptimizationResult, commodity: str):
        payload = {
            "commodity": commodity,
            "created_at": datetime.now().isoformat(),
            "optimization_method": result.optimization_method,
            "promoted": result.promoted,
            "incumbent_parameters": result.incumbent_parameters.parameters,
            "candidate_parameters": result.candidate_parameters.parameters if result.candidate_parameters else None,
            "evidence": result.evidence,
            "validation_results": {
                name: {
                    "total_return": backtest.total_return,
                    "sharpe_ratio": backtest.sharpe_ratio,
                    "win_rate": backtest.win_rate,
                    "signal_accuracy": backtest.signal_accuracy,
                }
                for name, backtest in result.validation_results.items()
            },
        }
        output_path = self.results_dir / f"optimization_{commodity}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_path.write_text(json.dumps(payload, indent=2, default=str))
