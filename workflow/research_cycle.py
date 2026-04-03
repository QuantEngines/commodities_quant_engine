from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Union

import json

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
                macro_features=macro_features,
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
        shipping_artifacts = self._persist_shipping_vectors(
            commodity=commodity,
            signal_id=package.snapshot.signal_id,
            as_of_timestamp=as_of_timestamp,
            shipping_feature_vectors=shipping_feature_vectors,
        )
        if shipping_artifacts:
            package.suggestion.diagnostics["shipping_artifacts"] = shipping_artifacts
            package.snapshot.metadata["shipping_artifacts"] = shipping_artifacts
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
                "shipping_artifacts": shipping_artifacts,
            }
            self.storage.write_json(settings.storage.report_store, f"{commodity}_{package.snapshot.signal_id}", report_payload)
        return package

    def _persist_shipping_vectors(
        self,
        commodity: str,
        signal_id: str,
        as_of_timestamp: datetime,
        shipping_feature_vectors: Optional[List[ShippingFeatureVector]],
    ) -> Dict[str, Any]:
        vectors = shipping_feature_vectors or []
        if not vectors:
            return {}

        serializable_rows: List[Dict[str, Any]] = []
        flattened_rows: List[Dict[str, Any]] = []
        for index, vector in enumerate(vectors):
            vector_payload = vector.to_dict()
            row_id = f"{signal_id}_{pd.Timestamp(vector.timestamp).isoformat()}_{index}"
            serializable_rows.append(
                {
                    "row_id": row_id,
                    "signal_id": signal_id,
                    "commodity": commodity,
                    "as_of_timestamp": as_of_timestamp.isoformat(),
                    **vector_payload,
                }
            )
            flattened_rows.append(
                {
                    "row_id": row_id,
                    "signal_id": signal_id,
                    "commodity": commodity,
                    "as_of_timestamp": as_of_timestamp.isoformat(),
                    "vector_timestamp": pd.Timestamp(vector.timestamp).isoformat(),
                    "source": vector.source,
                    "quality_score": float(vector.quality_score),
                    "confidence_score": float(vector.confidence_score),
                    "observation_start": pd.Timestamp(vector.observation_window.start_time).isoformat(),
                    "observation_end": pd.Timestamp(vector.observation_window.end_time).isoformat(),
                    "bdi_benchmark_active": float(vector.features.get("bdi_benchmark_active", 0.0)),
                    "bdi_benchmark_level": float(vector.features.get("bdi_benchmark_level", 0.0)),
                    "bdi_benchmark_zscore": float(vector.features.get("bdi_benchmark_zscore", 0.0)),
                    "bdi_benchmark_momentum": float(vector.features.get("bdi_benchmark_momentum", 0.0)),
                    "bdi_benchmark_support": float(vector.features.get("bdi_benchmark_support", 0.0)),
                    "bdi_shipping_stress_score": float(vector.features.get("bdi_shipping_stress_score", 0.0)),
                    "bdi_shipping_divergence": float(vector.features.get("bdi_shipping_divergence", 0.0)),
                    "shipping_market_benchmark_active": float(vector.features.get("shipping_market_benchmark_active", 0.0)),
                    "shipping_market_benchmark_zscore": float(vector.features.get("shipping_market_benchmark_zscore", 0.0)),
                    "shipping_market_benchmark_momentum": float(vector.features.get("shipping_market_benchmark_momentum", 0.0)),
                    "shipping_market_benchmark_support": float(vector.features.get("shipping_market_benchmark_support", 0.0)),
                    "shipping_market_stress_score": float(vector.features.get("shipping_market_stress_score", 0.0)),
                    "shipping_market_divergence": float(vector.features.get("shipping_market_divergence", 0.0)),
                    "shipping_features": json.dumps(vector.features, default=str),
                    "key_drivers": json.dumps(vector.key_drivers, default=str),
                    "notes": json.dumps(vector.notes, default=str),
                }
            )

        history_path = self.storage.append_dataframe(
            pd.DataFrame(flattened_rows),
            settings.storage.shipping_store,
            f"{commodity}_benchmark_vectors",
            dedupe_on=["row_id"],
        )
        raw_path = self.storage.append_jsonl(
            settings.storage.shipping_store,
            f"{commodity}_benchmark_vectors",
            serializable_rows,
            compress=True,
        )

        latest_row = flattened_rows[-1]
        divergence_values = [float(row["bdi_shipping_divergence"]) for row in flattened_rows]
        active_rows = [row for row in flattened_rows if float(row["bdi_benchmark_active"]) > 0.5]
        summary_payload = {
            "signal_id": signal_id,
            "commodity": commodity,
            "as_of_timestamp": as_of_timestamp.isoformat(),
            "vector_count": len(flattened_rows),
            "benchmark_active_count": len(active_rows),
            "latest_vector_timestamp": latest_row["vector_timestamp"],
            "latest_bdi_benchmark_zscore": latest_row["bdi_benchmark_zscore"],
            "latest_bdi_benchmark_support": latest_row["bdi_benchmark_support"],
            "latest_bdi_shipping_stress_score": latest_row["bdi_shipping_stress_score"],
            "latest_bdi_shipping_divergence": latest_row["bdi_shipping_divergence"],
            "latest_shipping_market_benchmark_zscore": latest_row["shipping_market_benchmark_zscore"],
            "latest_shipping_market_benchmark_support": latest_row["shipping_market_benchmark_support"],
            "latest_shipping_market_stress_score": latest_row["shipping_market_stress_score"],
            "latest_shipping_market_divergence": latest_row["shipping_market_divergence"],
            "max_abs_bdi_shipping_divergence": max((abs(value) for value in divergence_values), default=0.0),
            "mean_abs_bdi_shipping_divergence": float(sum(abs(value) for value in divergence_values) / len(divergence_values)),
            "history_path": str(history_path),
            "raw_path": str(raw_path),
        }
        summary_path = self.storage.write_json(
            settings.storage.shipping_store,
            f"{commodity}_{signal_id}_benchmark_summary",
            summary_payload,
        )
        return {
            "history_path": str(history_path),
            "raw_path": str(raw_path),
            "summary_path": str(summary_path),
            "latest_bdi_shipping_divergence": summary_payload["latest_bdi_shipping_divergence"],
            "latest_shipping_market_divergence": summary_payload["latest_shipping_market_divergence"],
            "benchmark_active_count": summary_payload["benchmark_active_count"],
        }

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
