from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Mapping, Optional, Union

import pandas as pd

from ..analytics.adaptation import AdaptiveParameterEngine
from ..analytics.evaluation import SignalEvaluationEngine
from ..config.settings import settings
from ..data.models import AdaptationDecision, EvaluationArtifact, MacroEvent, MacroFeature, SignalPackage
from ..data.storage.local import LocalStorage
from ..nlp.macro_event_engine.cluster_report import ClusterReportGenerator
from ..shipping.models import ShippingFeatureVector
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
        shipping_feature_vectors: Optional[List[ShippingFeatureVector]] = None,
        raw_text_items: Optional[List[Union[str, Mapping[str, object]]]] = None,
        llm_json_by_source_id: Optional[Dict[str, str]] = None,
        as_of_timestamp: Optional[datetime] = None,
        persist_snapshot: bool = True,
        persist_report: bool = True,
    ) -> SignalPackage:
        active_version = self.adaptation_engine.load_active_version(commodity)
        calibration_payload = self.storage.read_json(settings.storage.evaluation_store, f"{commodity}_calibration")
        parameter_state: Dict[str, object] = {"version_id": active_version.version_id, **active_version.parameters}
        if calibration_payload:
            parameter_state["confidence_calibration"] = calibration_payload.get("confidence_calibration", {})
            parameter_state["regime_calibration"] = calibration_payload.get("regime_calibration", {})
        composite_engine = CompositeDecisionEngine(parameter_state=parameter_state)
        package = composite_engine.generate_signal_package(
            data=price_data,
            commodity=commodity,
            macro_features=macro_features,
            macro_events=macro_events,
            shipping_feature_vectors=shipping_feature_vectors,
            raw_text_items=raw_text_items,
            llm_json_by_source_id=llm_json_by_source_id,
            as_of_timestamp=as_of_timestamp,
        )
        if persist_snapshot:
            self.evaluation_engine.persist_signal_snapshots([package.snapshot], commodity=commodity)
            self._persist_structured_events(package=package, commodity=commodity)
        if persist_report:
            self._generate_cluster_report(package=package, commodity=commodity)
            report_payload = {
                "signal_id": package.snapshot.signal_id,
                "timestamp": package.snapshot.timestamp,
                "commodity": package.snapshot.commodity,
                "suggestion_markdown": package.suggestion.to_markdown(),
                "quality_report": asdict(package.quality_report),
            }
            self.storage.write_json(settings.storage.report_store, f"{commodity}_{package.snapshot.signal_id}", report_payload)
        return package

    def _generate_cluster_report(self, package: SignalPackage, commodity: str) -> None:
        cluster_manifest = package.suggestion.diagnostics.get("event_cluster_manifest", [])
        if not isinstance(cluster_manifest, list) or not cluster_manifest:
            return
        diagnostics = package.suggestion.diagnostics.get("event_intelligence_diagnostics", {})
        ClusterReportGenerator().generate(
            cluster_manifest=cluster_manifest,
            diagnostics=diagnostics,
            commodity=commodity,
            signal_id=package.snapshot.signal_id,
            as_of_timestamp=package.snapshot.timestamp,
            storage=self.storage,
        )

    def _persist_structured_events(self, package: SignalPackage, commodity: str) -> None:
        event_rows = package.suggestion.diagnostics.get("event_intelligence_events", [])
        if not isinstance(event_rows, list) or not event_rows:
            return

        normalized_rows: List[Dict[str, object]] = []
        for idx, row in enumerate(event_rows):
            if not isinstance(row, dict):
                continue
            normalized_rows.append(
                {
                    "signal_id": package.snapshot.signal_id,
                    "commodity": commodity,
                    "snapshot_timestamp": package.snapshot.timestamp.isoformat(),
                    "event_row_id": f"{package.snapshot.signal_id}_{idx}",
                    **row,
                }
            )
        if not normalized_rows:
            return

        self.storage.append_jsonl(
            settings.storage.signal_store,
            f"{commodity}_structured_events",
            normalized_rows,
            compress=True,
        )
        event_df = pd.DataFrame(normalized_rows)
        self.storage.append_dataframe(
            event_df,
            settings.storage.signal_store,
            f"{commodity}_structured_events",
            dedupe_on=["event_row_id"],
        )

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
