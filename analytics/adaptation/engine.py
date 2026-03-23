from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from ...config.settings import settings
from ...data.models import AdaptationDecision, ParameterVersion, SignalSnapshot
from ...data.storage.local import LocalStorage
from ..evaluation.engine import SignalEvaluationEngine


class AdaptiveParameterEngine:
    """Governed adaptive recalibration using holdout-validated statistical updates."""

    def __init__(self, storage: Optional[LocalStorage] = None):
        self.storage = storage or LocalStorage()
        self.evaluation_engine = SignalEvaluationEngine(storage=self.storage)

    def load_active_version(self, commodity: str) -> ParameterVersion:
        domain = f"{settings.storage.parameter_store}/{commodity}"
        active_payload = self.storage.read_json(domain, "active")
        if active_payload:
            return self._version_from_payload(active_payload)
        default = ParameterVersion(
            version_id="default",
            commodity=commodity,
            created_at=datetime.now(),
            parent_version_id=None,
            parameters={
                "directional_feature_weights": deepcopy(settings.signal.directional_feature_weights),
                "directional_intercepts": deepcopy(settings.signal.directional_intercepts),
            },
            evidence={"source": "settings_defaults"},
            metrics={},
            mode="incumbent",
            approved=True,
            active=True,
            reason="Default parameter state from configuration.",
        )
        self._persist_version(default)
        self._set_active_version(default)
        return default

    def recommend_update(
        self,
        commodity: str,
        dry_run: bool = True,
        approve: bool = False,
        auto_promote: Optional[bool] = None,
    ) -> AdaptationDecision:
        auto_promote = settings.adaptation.auto_promote if auto_promote is None else auto_promote
        incumbent = self.load_active_version(commodity)
        snapshots = self.evaluation_engine.load_signal_snapshots(commodity)
        evaluation_df = self.storage.load_domain_dataframe(settings.storage.evaluation_store, f"{commodity}_detailed")
        if evaluation_df.empty:
            return self._decision(
                commodity=commodity,
                incumbent=incumbent,
                candidate=None,
                promoted=False,
                approved=False,
                reason="No evaluation records are available yet.",
                evidence={},
                safety_checks={"has_evaluations": False},
                mode="recommend",
            )

        snapshot_df = pd.DataFrame([snapshot.to_dict() for snapshot in snapshots]).rename(columns={"timestamp": "signal_timestamp"})
        merged = evaluation_df.merge(
            snapshot_df[["signal_id", "feature_vector", "directional_scores", "signal_timestamp"]],
            on="signal_id",
            how="inner",
        )
        if len(merged) < settings.adaptation.min_sample_size:
            return self._decision(
                commodity=commodity,
                incumbent=incumbent,
                candidate=None,
                promoted=False,
                approved=False,
                reason="Insufficient evaluated signal sample for recalibration.",
                evidence={"sample_size": int(len(merged))},
                safety_checks={"min_sample_size": False},
                mode="recommend",
            )

        candidate_parameters, evidence, safety_checks = self._fit_candidate_parameters(merged, incumbent)
        if not candidate_parameters:
            return self._decision(
                commodity=commodity,
                incumbent=incumbent,
                candidate=None,
                promoted=False,
                approved=False,
                reason="Candidate update did not clear improvement or stability thresholds.",
                evidence=evidence,
                safety_checks=safety_checks,
                mode="recommend",
            )

        candidate = ParameterVersion(
            version_id=datetime.now().strftime("v%Y%m%d%H%M%S"),
            commodity=commodity,
            created_at=datetime.now(),
            parent_version_id=incumbent.version_id,
            parameters=candidate_parameters,
            evidence=evidence,
            metrics=evidence.get("candidate_metrics", {}),
            mode="candidate",
            approved=approve,
            active=False,
            reason="Holdout-validated statistical recalibration of directional feature weights.",
        )
        self._persist_version(candidate)

        should_promote = bool(
            candidate
            and not dry_run
            and (
                (auto_promote and not settings.adaptation.manual_approval_required)
                or approve
            )
        )
        if should_promote:
            incumbent.active = False
            self._persist_version(incumbent)
            candidate.active = True
            self._persist_version(candidate)
            self._set_active_version(candidate)

        return self._decision(
            commodity=commodity,
            incumbent=incumbent,
            candidate=candidate,
            promoted=should_promote,
            approved=approve,
            reason="Candidate parameter version created." if not should_promote else "Candidate parameter version promoted to active.",
            evidence=evidence,
            safety_checks=safety_checks,
            mode="promote" if should_promote else "recommend",
        )

    def _fit_candidate_parameters(self, merged: pd.DataFrame, incumbent: ParameterVersion):
        candidate_weights = deepcopy(incumbent.parameters["directional_feature_weights"])
        candidate_intercepts = deepcopy(incumbent.parameters.get("directional_intercepts", {}))
        horizon_results: Dict[str, Dict[str, float]] = {}
        improvement_flags: List[bool] = []
        drift_flags: List[bool] = []
        safety_checks = {"min_sample_size": True, "holdout_improvement": False, "feature_drift_ok": True}

        for horizon, horizon_df in merged.groupby("horizon"):
            if len(horizon_df) < settings.adaptation.min_sample_size:
                continue

            dataset = self._expand_feature_vectors(horizon_df)
            feature_names = settings.signal.directional_feature_names
            X = dataset[feature_names].fillna(0.0)
            y = self._winsorize_series(dataset["realized_return"].astype(float))
            splits = self._purged_walkforward_splits(len(dataset), int(horizon))
            if not splits:
                continue

            candidate_fold_preds: List[pd.Series] = []
            incumbent_fold_preds: List[pd.Series] = []
            holdout_targets: List[pd.Series] = []
            for train_idx, holdout_idx in splits:
                train_X = X.iloc[train_idx]
                train_y = y.iloc[train_idx]
                holdout_X = X.iloc[holdout_idx]
                holdout_y = y.iloc[holdout_idx]
                if holdout_X.empty or train_X.empty:
                    continue

                fold_model = Ridge(alpha=settings.adaptation.ridge_alpha)
                sample_weights = self._sample_weights(len(train_X))
                fold_model.fit(train_X, train_y, sample_weight=sample_weights)
                candidate_fold_preds.append(pd.Series(fold_model.predict(holdout_X), index=holdout_X.index))
                incumbent_fold_preds.append(
                    dataset.iloc[holdout_idx]["directional_scores"].map(
                        lambda item: float(item.get(str(horizon), item.get(horizon, 0.0)) if isinstance(item, dict) else 0.0)
                    )
                )
                holdout_targets.append(holdout_y)

            if not candidate_fold_preds:
                continue

            candidate_pred = pd.concat(candidate_fold_preds).sort_index()
            incumbent_pred = pd.concat(incumbent_fold_preds).sort_index()
            holdout_y = pd.concat(holdout_targets).sort_index()

            candidate_metrics = self._prediction_metrics(candidate_pred, holdout_y)
            incumbent_metrics = self._prediction_metrics(incumbent_pred, holdout_y)

            final_model = Ridge(alpha=settings.adaptation.ridge_alpha)
            final_weights = self._sample_weights(len(X))
            final_model.fit(X, y, sample_weight=final_weights)

            improvement = (
                candidate_metrics["hit_rate"] >= incumbent_metrics["hit_rate"] + settings.adaptation.min_hit_rate_improvement
                and candidate_metrics["rank_ic"] >= incumbent_metrics["rank_ic"] + settings.adaptation.min_rank_ic_improvement
            )
            drift_ok = self._feature_drift_ok(candidate_weights.get(str(horizon), {}), feature_names, final_model.coef_)
            improvement_flags.append(improvement)
            drift_flags.append(drift_ok)

            horizon_results[str(horizon)] = {
                "sample_size": int(len(dataset)),
                "fold_count": int(len(candidate_fold_preds)),
                "holdout_records": int(len(holdout_y)),
                "candidate_hit_rate": candidate_metrics["hit_rate"],
                "incumbent_hit_rate": incumbent_metrics["hit_rate"],
                "candidate_rank_ic": candidate_metrics["rank_ic"],
                "incumbent_rank_ic": incumbent_metrics["rank_ic"],
                "improved": bool(improvement),
                "drift_ok": bool(drift_ok),
            }
            if improvement and drift_ok:
                candidate_weights[str(horizon)] = {
                    feature_name: float(coef)
                    for feature_name, coef in zip(feature_names, final_model.coef_)
                }
                candidate_intercepts[str(horizon)] = float(final_model.intercept_)

        safety_checks["holdout_improvement"] = bool(improvement_flags) and any(improvement_flags)
        safety_checks["feature_drift_ok"] = bool(drift_flags) and all(drift_flags)

        evidence = {
            "evaluated_horizons": horizon_results,
            "candidate_metrics": {
                "improved_horizons": int(sum(result["improved"] for result in horizon_results.values())),
                "tested_horizons": int(len(horizon_results)),
            },
        }
        if not horizon_results or not any(improvement_flags):
            return None, evidence, safety_checks
        if evidence["candidate_metrics"]["improved_horizons"] < settings.adaptation.min_improved_horizons:
            return None, evidence, safety_checks
        if not all(safety_checks.values()):
            return None, evidence, safety_checks

        return {
            "directional_feature_weights": candidate_weights,
            "directional_intercepts": candidate_intercepts,
        }, evidence, safety_checks

    def _purged_walkforward_splits(self, length: int, horizon: int) -> List[Tuple[np.ndarray, np.ndarray]]:
        if length <= 0:
            return []

        min_sample = int(settings.adaptation.min_sample_size)
        purge = max(int(settings.adaptation.purge_overlap_bars), int(horizon))
        embargo = max(int(settings.adaptation.embargo_bars), int(horizon))
        min_train = max(min_sample, purge + embargo + 1)

        min_holdout = max(5, int(horizon))
        desired_holdout = max(min_holdout, int(length * float(settings.adaptation.holdout_fraction)))
        available_holdout = length - (min_train + purge + embargo)
        holdout_size = min(desired_holdout, available_holdout)
        if holdout_size < min_holdout:
            return []

        earliest_holdout_start = min_train + purge + embargo
        latest_holdout_start = length - holdout_size
        if earliest_holdout_start > latest_holdout_start:
            return []

        folds = max(1, int(settings.adaptation.walk_forward_folds))
        if folds == 1:
            holdout_starts = [latest_holdout_start]
        else:
            holdout_starts = sorted(
                set(int(value) for value in np.linspace(earliest_holdout_start, latest_holdout_start, num=folds))
            )

        splits: List[Tuple[np.ndarray, np.ndarray]] = []
        for holdout_start in holdout_starts:
            train_end_exclusive = holdout_start - purge - embargo
            if train_end_exclusive < min_train:
                continue
            holdout_end_exclusive = min(length, holdout_start + holdout_size)
            if holdout_start >= holdout_end_exclusive:
                continue
            train_idx = np.arange(0, train_end_exclusive)
            holdout_idx = np.arange(holdout_start, holdout_end_exclusive)
            splits.append((train_idx, holdout_idx))
        return splits

    def _expand_feature_vectors(self, dataset: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for row in dataset.to_dict(orient="records"):
            feature_vector = row.get("feature_vector", {})
            directional_scores = row.get("directional_scores", {})
            expanded = dict(row)
            expanded["directional_scores"] = directional_scores
            for feature_name in settings.signal.directional_feature_names:
                expanded[feature_name] = float(feature_vector.get(feature_name, 0.0)) if isinstance(feature_vector, dict) else 0.0
            rows.append(expanded)
        expanded_df = pd.DataFrame(rows)
        sort_column = "signal_timestamp" if "signal_timestamp" in expanded_df.columns else "timestamp"
        return expanded_df.sort_values(sort_column)

    def _prediction_metrics(self, predicted: pd.Series, actual: pd.Series) -> Dict[str, float]:
        if len(predicted) == 0:
            return {"hit_rate": 0.0, "rank_ic": 0.0}
        predicted_direction = np.sign(predicted)
        actual_direction = np.sign(actual)
        hit_rate = float((predicted_direction == actual_direction).mean())
        if predicted.nunique() <= 1 or actual.nunique() <= 1:
            rank_ic = 0.0
        else:
            rank_ic_value = predicted.corr(actual, method="spearman")
            rank_ic = float(rank_ic_value) if pd.notna(rank_ic_value) else 0.0
        return {"hit_rate": hit_rate, "rank_ic": rank_ic}

    def _sample_weights(self, length: int) -> np.ndarray:
        if length <= 0:
            return np.array([])
        halflife = max(1, int(settings.adaptation.recency_halflife_signals))
        age = np.arange(length - 1, -1, -1)
        weights = 0.5 ** (age / halflife)
        return weights / weights.mean()

    def _winsorize_series(self, series: pd.Series) -> pd.Series:
        quantile = float(settings.adaptation.target_winsor_quantile)
        if series.empty or quantile <= 0.0:
            return series
        lower = float(series.quantile(quantile))
        upper = float(series.quantile(1.0 - quantile))
        return series.clip(lower=lower, upper=upper)

    def _feature_drift_ok(self, incumbent_weights: Dict[str, float], feature_names: List[str], candidate_coef: np.ndarray) -> bool:
        drift = 0.0
        for feature_name, coef in zip(feature_names, candidate_coef):
            drift = max(drift, abs(float(coef) - float(incumbent_weights.get(feature_name, 0.0))))
        return drift <= settings.adaptation.max_feature_drift

    def _persist_version(self, version: ParameterVersion):
        domain = f"{settings.storage.parameter_store}/{version.commodity}"
        self.storage.write_json(domain, version.version_id, version.to_dict())

    def _set_active_version(self, version: ParameterVersion):
        domain = f"{settings.storage.parameter_store}/{version.commodity}"
        self.storage.write_json(domain, "active", version.to_dict())

    def _version_from_payload(self, payload: Dict[str, object]) -> ParameterVersion:
        return ParameterVersion(
            version_id=str(payload["version_id"]),
            commodity=str(payload["commodity"]),
            created_at=pd.Timestamp(payload["created_at"]).to_pydatetime(),
            parent_version_id=payload.get("parent_version_id"),
            parameters=payload.get("parameters", {}),
            evidence=payload.get("evidence", {}),
            metrics=payload.get("metrics", {}),
            mode=str(payload.get("mode", "candidate")),
            approved=bool(payload.get("approved", False)),
            active=bool(payload.get("active", False)),
            reason=str(payload.get("reason", "")),
        )

    def _decision(
        self,
        commodity: str,
        incumbent: ParameterVersion,
        candidate: Optional[ParameterVersion],
        promoted: bool,
        approved: bool,
        reason: str,
        evidence: Dict[str, object],
        safety_checks: Dict[str, bool],
        mode: str,
    ) -> AdaptationDecision:
        decision = AdaptationDecision(
            commodity=commodity,
            created_at=datetime.now(),
            incumbent_version_id=incumbent.version_id if incumbent else None,
            candidate_version_id=candidate.version_id if candidate else None,
            promoted=promoted,
            approved=approved,
            reason=reason,
            evidence=evidence,
            safety_checks=safety_checks,
            mode=mode,
        )
        domain = f"{settings.storage.parameter_store}/{commodity}"
        self.storage.write_json(domain, f"decision_{decision.created_at.strftime('%Y%m%d%H%M%S')}", decision.to_dict())
        if candidate and promoted:
            self._set_active_version(candidate)
        return decision
