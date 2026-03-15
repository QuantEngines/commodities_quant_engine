from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd

from ..analytics.adaptation import AdaptiveParameterEngine
from ..analytics.evaluation import SignalEvaluationEngine
from ..config.settings import settings
from ..data.models import AdaptationDecision, EvaluationArtifact, MacroEvent, MacroFeature, SignalPackage
from ..data.storage.local import LocalStorage
from ..signals.composite.composite_decision import CompositeDecisionEngine


class ResearchWorkflow:
    """Explicit end-to-end research lifecycle for signals, evaluation, and adaptation."""

    def __init__(self, storage: Optional[LocalStorage] = None):
        self.storage = storage or LocalStorage()
        self.adaptation_engine = AdaptiveParameterEngine(storage=self.storage)
        self.evaluation_engine = SignalEvaluationEngine(storage=self.storage)

    def run_signal_cycle(
        self,
        commodity: str,
        price_data: pd.DataFrame,
        macro_features: Optional[List[MacroFeature]] = None,
        macro_events: Optional[List[MacroEvent]] = None,
        as_of_timestamp: Optional[datetime] = None,
        persist_snapshot: bool = True,
        persist_report: bool = True,
    ) -> SignalPackage:
        active_version = self.adaptation_engine.load_active_version(commodity)
        composite_engine = CompositeDecisionEngine(parameter_state={"version_id": active_version.version_id, **active_version.parameters})
        package = composite_engine.generate_signal_package(
            data=price_data,
            commodity=commodity,
            macro_features=macro_features,
            macro_events=macro_events,
            as_of_timestamp=as_of_timestamp,
        )
        if persist_snapshot:
            self.evaluation_engine.persist_signal_snapshots([package.snapshot], commodity=commodity)
        if persist_report:
            report_payload = {
                "signal_id": package.snapshot.signal_id,
                "timestamp": package.snapshot.timestamp,
                "commodity": package.snapshot.commodity,
                "suggestion_markdown": package.suggestion.to_markdown(),
                "quality_report": asdict(package.quality_report),
            }
            self.storage.write_json(settings.storage.report_store, f"{commodity}_{package.snapshot.signal_id}", report_payload)
        return package

    def run_evaluation_cycle(
        self,
        commodity: str,
        price_data: pd.DataFrame,
        macro_events: Optional[List[MacroEvent]] = None,
        as_of_timestamp: Optional[datetime] = None,
    ) -> EvaluationArtifact:
        return self.evaluation_engine.evaluate_signals(
            commodity=commodity,
            price_data=price_data,
            macro_events=macro_events,
            as_of_timestamp=as_of_timestamp,
        )

    def run_adaptation_cycle(
        self,
        commodity: str,
        dry_run: bool = True,
        approve: bool = False,
        auto_promote: Optional[bool] = None,
    ) -> AdaptationDecision:
        return self.adaptation_engine.recommend_update(
            commodity=commodity,
            dry_run=dry_run,
            approve=approve,
            auto_promote=auto_promote,
        )
