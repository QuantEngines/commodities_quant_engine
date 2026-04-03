from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Mapping, Optional, Union

import pandas as pd

from ..analytics.adaptation import AdaptiveParameterEngine
from ..analytics.attribution import AttributionEngine
from ..analytics.diagnostics import DiagnosticsEngine
from ..analytics.evaluation import SignalEvaluationEngine
from ..config.settings import settings
from ..data.models import AdaptationDecision, EvaluationArtifact, MacroEvent, MacroFeature, SignalPackage
from ..data.storage.local import LocalStorage
from ..nlp.macro_event_engine.cluster_report import ClusterReportGenerator
from ..portfolio.optimization_engine import PortfolioOptimizationEngine
from ..portfolio.position_suggestion.recommendation_engine import PositionSuggestionEngine
from ..reporting.dashboards import PortfolioDashboard
from ..reporting.ranking_tables import SignalRankingTable
from ..shipping.context_builder import shipping_context_builder
from ..shipping.models import ShippingFeatureVector
from ..signals.composite.composite_decision import CompositeDecisionEngine
from ..signals.intraday.intraday_engine import IntradayFactorRotationEngine


class ResearchWorkflow:
    """Explicit end-to-end research lifecycle for signals, evaluation, and adaptation."""

    def __init__(self, storage: Optional[LocalStorage] = None):
        self.storage = storage or LocalStorage()
        self.adaptation_engine = AdaptiveParameterEngine(storage=self.storage)
        self.evaluation_engine = SignalEvaluationEngine(storage=self.storage)
        self.attribution_engine = AttributionEngine()
        self.diagnostics_engine = DiagnosticsEngine()
        self.portfolio_optimization_engine = PortfolioOptimizationEngine(storage=self.storage)
        self.position_suggestion_engine = PositionSuggestionEngine(storage=self.storage)
        self.intraday_engine = IntradayFactorRotationEngine()

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
        if as_of_timestamp is None:
            as_of_timestamp = price_data.index[-1].to_pydatetime() if not price_data.empty else datetime.now()
        
        # Auto-generate shipping context if not provided (Sprint 1: Always-on shipping)
        if shipping_feature_vectors is None:
            shipping_feature_vectors = shipping_context_builder.build(
                commodity=commodity,
                as_of_timestamp=as_of_timestamp,
            )
        
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
        artifact = self.evaluation_engine.evaluate_signals(
            commodity=commodity,
            price_data=price_data,
            macro_events=macro_events,
            as_of_timestamp=as_of_timestamp,
        )
        snapshots = self.evaluation_engine.load_signal_snapshots(commodity)
        attribution_payload = self.attribution_engine.summarize_component_attribution(snapshots)
        diagnostics_payload = self.diagnostics_engine.summarize_evaluation_health(artifact)
        attribution_path = self.storage.write_json(
            settings.storage.report_store,
            f"{commodity}_attribution_summary",
            attribution_payload,
        )
        diagnostics_path = self.storage.write_json(
            settings.storage.report_store,
            f"{commodity}_diagnostics_summary",
            diagnostics_payload,
        )
        artifact.scorecards["attribution_summary_path"] = str(attribution_path)
        artifact.scorecards["diagnostics_summary_path"] = str(diagnostics_path)
        artifact.scorecards["diagnostics_summary"] = diagnostics_payload
        artifact.scorecards["attribution_summary"] = attribution_payload
        return artifact

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

    def run_portfolio_cycle(
        self,
        price_data_by_commodity: Dict[str, pd.DataFrame],
        portfolio_budget: float,
        current_positions: Optional[Dict[str, int]] = None,
        as_of_timestamp: Optional[datetime] = None,
        persist_snapshots: bool = True,
        persist_signal_reports: bool = True,
        persist_portfolio_report: bool = True,
    ) -> Dict[str, object]:
        """Run a suggestion-only portfolio cycle across commodities.

        This method never places orders and only returns analytical suggestions.
        """
        if not price_data_by_commodity:
            raise ValueError("price_data_by_commodity cannot be empty")

        signal_packages: Dict[str, SignalPackage] = {}
        signal_data: Dict[str, Dict[str, object]] = {}
        latest_prices: Dict[str, float] = {}
        realized_volatility: Dict[str, float] = {}
        price_history: Dict[str, object] = {}

        for commodity, frame in price_data_by_commodity.items():
            if frame is None or frame.empty:
                continue
            commodity_as_of = as_of_timestamp or frame.index[-1].to_pydatetime()
            package = self.run_signal_cycle(
                commodity=commodity,
                price_data=frame,
                as_of_timestamp=commodity_as_of,
                persist_snapshot=persist_snapshots,
                persist_report=persist_signal_reports,
            )
            signal_packages[commodity] = package
            signal_data[commodity] = {
                "composite_score": float(package.suggestion.composite_score),
                "confidence_score": float(package.suggestion.confidence_score),
                "regime_label": package.suggestion.regime_label,
                "signal_id": package.suggestion.signal_id,
            }
            latest_prices[commodity] = float(frame["close"].iloc[-1])
            vol = float(frame["close"].pct_change().dropna().std(ddof=0)) if len(frame) > 2 else 0.0
            realized_volatility[commodity] = max(0.0, vol)
            price_history[commodity] = frame["close"].astype(float).to_numpy()

        if not signal_packages:
            raise ValueError("No valid commodity frames were available for portfolio cycle")

        commodity_signals = {
            commodity: max(0.0, abs(float(payload.get("composite_score", 0.0))))
            for commodity, payload in signal_data.items()
        }
        portfolio_weights = self.portfolio_optimization_engine.optimize_commodity_weights(
            commodity_signals=commodity_signals,
            price_history=price_history,
        )
        sector_exposures = self.portfolio_optimization_engine.get_sector_exposures(portfolio_weights)

        suggestions = self.position_suggestion_engine.generate_portfolio_suggestions(
            portfolio_weights=portfolio_weights,
            signal_data=signal_data,
            price_data=latest_prices,
            portfolio_budget=portfolio_budget,
            market_volatility=realized_volatility,
            current_positions=current_positions,
        )
        suggestion_dicts = {commodity: suggestion.to_dict() for commodity, suggestion in suggestions.items()}
        ranking_frame = SignalRankingTable.build(suggestion_dicts)
        ranking_markdown = SignalRankingTable.to_markdown(ranking_frame)
        dashboard_payload = PortfolioDashboard.build_payload(
            portfolio_budget=portfolio_budget,
            portfolio_weights=portfolio_weights,
            sector_exposures=sector_exposures,
            suggestions_frame=ranking_frame,
        )

        report_name = f"portfolio_cycle_{(as_of_timestamp or datetime.now()).strftime('%Y%m%d_%H%M%S')}"
        report_payload = {
            "generated_at": (as_of_timestamp or datetime.now()).isoformat(),
            "portfolio_budget": float(portfolio_budget),
            "suggestion_only": True,
            "note": "No broker orders are generated or transmitted by this workflow.",
            "commodities": sorted(signal_packages.keys()),
            "portfolio_weights": portfolio_weights,
            "sector_exposures": sector_exposures,
            "suggestions": suggestion_dicts,
            "ranking_table": ranking_frame.to_dict(orient="records"),
            "ranking_markdown": ranking_markdown,
            "dashboard": dashboard_payload,
            "signal_ids": {commodity: package.snapshot.signal_id for commodity, package in signal_packages.items()},
        }

        suggestions_name = f"{report_name}_suggestions"
        self.position_suggestion_engine.persist_suggestions(suggestions, name=suggestions_name)

        report_path = None
        if persist_portfolio_report:
            report_path = self.storage.write_json(
                settings.storage.report_store,
                report_name,
                report_payload,
            )

        return {
            "signal_packages": signal_packages,
            "portfolio_weights": portfolio_weights,
            "sector_exposures": sector_exposures,
            "suggestions": suggestions,
            "portfolio_report_name": report_name,
            "portfolio_report_path": str(report_path) if report_path else None,
        }

    def run_intraday_signal_cycle(
        self,
        commodity: str,
        daily_price_data: pd.DataFrame,
        intraday_price_data: pd.DataFrame,
        interval: str = "1H",
        as_of_timestamp: Optional[datetime] = None,
        persist_snapshot: bool = True,
        persist_report: bool = True,
    ) -> Dict[str, object]:
        """Run daily signal context plus intraday tactical signal cycle."""
        daily_package = self.run_signal_cycle(
            commodity=commodity,
            price_data=daily_price_data,
            as_of_timestamp=as_of_timestamp,
            persist_snapshot=persist_snapshot,
            persist_report=persist_report,
        )
        daily_signal = {
            "regime_label": daily_package.suggestion.regime_label,
            "composite_score": float(daily_package.suggestion.composite_score),
            "confidence_score": float(daily_package.suggestion.confidence_score),
            "factor_weights": {},
        }
        intraday_payload = self.intraday_engine.generate_intraday_signal_package(
            commodity=commodity,
            daily_signal=daily_signal,
            intraday_price_data=intraday_price_data,
            interval=interval,
        )
        return {
            "daily_package": daily_package,
            "intraday_signal": intraday_payload,
        }
