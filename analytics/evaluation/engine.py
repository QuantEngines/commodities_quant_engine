from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from ...analytics.execution import direction_to_sign, execution_price, slippage_rate
from ...config.settings import settings
from ...data.models import EvaluationArtifact, MacroEvent, SignalEvaluationRecord, SignalSnapshot
from ...data.storage.local import LocalStorage


class SignalEvaluationEngine:
    """Evaluate historical signals against realized market outcomes."""

    def __init__(self, storage: Optional[LocalStorage] = None):
        self.storage = storage or LocalStorage()

    def evaluate_signals(
        self,
        commodity: str,
        price_data: pd.DataFrame,
        signal_snapshots: Optional[Iterable[SignalSnapshot]] = None,
        macro_events: Optional[List[MacroEvent]] = None,
        as_of_timestamp: Optional[datetime] = None,
        horizons: Optional[List[int]] = None,
        persist: bool = True,
    ) -> EvaluationArtifact:
        horizons = horizons or settings.evaluation.horizons
        macro_events = macro_events or []
        as_of_timestamp = as_of_timestamp or price_data.index[-1].to_pydatetime()
        snapshots = list(signal_snapshots) if signal_snapshots is not None else self.load_signal_snapshots(commodity)
        detailed_records = self._build_evaluation_records(price_data, snapshots, horizons, macro_events, as_of_timestamp)
        detailed_df = pd.DataFrame([self._serialize_evaluation_record(record) for record in detailed_records])
        summary_metrics, scorecards, degradation_alerts = self._summarize_evaluations(detailed_df)

        artifact = EvaluationArtifact(
            commodity=commodity,
            created_at=as_of_timestamp,
            horizons=horizons,
            summary_metrics=summary_metrics,
            degradation_alerts=degradation_alerts,
            scorecards=scorecards,
        )

        if persist:
            detailed_path = self.storage.append_dataframe(
                detailed_df,
                settings.storage.evaluation_store,
                f"{commodity}_detailed",
                dedupe_on=["signal_id", "horizon"],
            )
            summary_payload = asdict(artifact)
            summary_payload["detailed_path"] = str(detailed_path)
            summary_path = self.storage.write_json(settings.storage.evaluation_store, f"{commodity}_summary", summary_payload)
            artifact.detailed_path = str(detailed_path)
            artifact.summary_path = str(summary_path)

        return artifact

    def persist_signal_snapshots(self, signal_snapshots: Iterable[SignalSnapshot], commodity: Optional[str] = None) -> str:
        serialized = [self._serialize_snapshot(snapshot) for snapshot in signal_snapshots]
        snapshot_df = pd.DataFrame(serialized)
        if snapshot_df.empty:
            raise ValueError("No signal snapshots were provided for persistence.")
        commodity = commodity or str(snapshot_df["commodity"].iloc[0])
        path = self.storage.append_jsonl(settings.storage.signal_store, commodity, serialized, compress=True)
        return str(path)

    def load_signal_snapshots(self, commodity: str) -> List[SignalSnapshot]:
        rows = self.storage.load_jsonl(settings.storage.signal_store, commodity)
        if not rows:
            df = self.storage.load_domain_dataframe(settings.storage.signal_store, commodity)
            rows = df.to_dict(orient="records") if not df.empty else []
        if not rows:
            return []
        snapshots = []
        seen_signal_ids = set()
        for row in rows:
            signal_id = row["signal_id"]
            if signal_id in seen_signal_ids:
                continue
            seen_signal_ids.add(signal_id)
            snapshots.append(
                SignalSnapshot(
                    signal_id=signal_id,
                    timestamp=pd.Timestamp(row["timestamp"]).to_pydatetime(),
                    commodity=row["commodity"],
                    contract=row["contract"],
                    exchange=row["exchange"],
                    signal_category=row["signal_category"],
                    direction=row["direction"],
                    conviction=float(row["conviction"]),
                    regime_label=row["regime_label"],
                    regime_probability=float(row["regime_probability"]),
                    inefficiency_score=float(row["inefficiency_score"]),
                    composite_score=float(row["composite_score"]),
                    suggested_horizon=int(row["suggested_horizon"]),
                    directional_scores=self._deserialize_payload(row.get("directional_scores", {})),
                    key_drivers=self._deserialize_payload(row.get("key_drivers", [])),
                    key_risks=self._deserialize_payload(row.get("key_risks", [])),
                    component_scores=self._deserialize_payload(row.get("component_scores", {})),
                    feature_vector=self._deserialize_payload(row.get("feature_vector", {})),
                    model_version=row.get("model_version", "default"),
                    config_version=row.get("config_version", settings.config_version),
                    data_quality_flag=row.get("data_quality_flag", "unknown"),
                    macro_alignment_score=row.get("macro_alignment_score"),
                    macro_conflict_score=row.get("macro_conflict_score"),
                    metadata=self._deserialize_payload(row.get("metadata", {})),
                )
            )
        return snapshots

    def _build_evaluation_records(
        self,
        price_data: pd.DataFrame,
        snapshots: List[SignalSnapshot],
        horizons: List[int],
        macro_events: List[MacroEvent],
        as_of_timestamp: datetime,
    ) -> List[SignalEvaluationRecord]:
        closes = price_data["close"].astype(float)
        returns = closes.pct_change().fillna(0.0)
        rolling_median_volume = price_data["volume"].rolling(20, min_periods=5).median() if "volume" in price_data.columns else pd.Series(index=price_data.index, dtype=float)
        records: List[SignalEvaluationRecord] = []
        entry_lag_bars = max(0, int(settings.evaluation.entry_lag_bars))
        for snapshot in sorted(snapshots, key=lambda item: item.timestamp):
            entry_idx = self._entry_index(price_data.index, snapshot.timestamp, entry_lag_bars)
            if entry_idx is None:
                continue
            for horizon in horizons:
                exit_idx = entry_idx + horizon
                if exit_idx >= len(price_data.index):
                    continue
                exit_ts = price_data.index[exit_idx].to_pydatetime()
                if exit_ts > as_of_timestamp:
                    continue
                raw_return, signed_return, path_returns, execution_metadata = self._forward_return_path(price_data, entry_idx, exit_idx, snapshot.direction, rolling_median_volume)
                actual_direction = self._direction_bucket(raw_return)
                direction_correct = actual_direction == snapshot.direction
                realized_regime = self._realized_regime(raw_return, returns.iloc[entry_idx + 1 : exit_idx + 1])
                record = SignalEvaluationRecord(
                    signal_id=snapshot.signal_id,
                    timestamp=snapshot.timestamp,
                    commodity=snapshot.commodity,
                    horizon=horizon,
                    direction=snapshot.direction,
                    confidence=float(snapshot.conviction),
                    composite_score=float(snapshot.composite_score),
                    realized_return=float(raw_return),
                    signed_return=float(signed_return),
                    direction_correct=bool(direction_correct),
                    excess_return=float(signed_return),
                    volatility_adjusted_return=float(self._vol_adjusted_return(signed_return, path_returns)),
                    max_favorable_excursion=float(path_returns.max()) if len(path_returns) else 0.0,
                    max_adverse_excursion=float(path_returns.min()) if len(path_returns) else 0.0,
                    follow_through_ratio=float((path_returns > 0).mean()) if len(path_returns) else 0.0,
                    reversal_probability=float(self._reversal_probability(path_returns)),
                    event_window_flag=self._event_window_flag(snapshot.timestamp, horizon, macro_events),
                    regime_label=snapshot.regime_label,
                    realized_regime_label=realized_regime,
                    regime_alignment=realized_regime in snapshot.regime_label,
                    confidence_bucket=self._confidence_bucket(snapshot.conviction),
                    metadata={
                        "suggested_horizon": snapshot.suggested_horizon,
                        "signal_category": snapshot.signal_category,
                        "model_version": snapshot.model_version,
                        **execution_metadata,
                    },
                )
                records.append(record)
        return records

    def _summarize_evaluations(self, detailed_df: pd.DataFrame) -> Tuple[Dict[str, object], Dict[str, object], List[str]]:
        if detailed_df.empty:
            return {"sample_size": 0}, {"by_horizon": {}, "confidence_buckets": {}}, ["No evaluable signals were available."]

        if "metadata" in detailed_df.columns:
            detailed_df = detailed_df.copy()
            detailed_df["metadata"] = detailed_df["metadata"].map(self._deserialize_payload)

        by_horizon = {}
        for horizon, group in detailed_df.groupby("horizon"):
            if group["composite_score"].nunique() <= 1 or group["realized_return"].nunique() <= 1:
                rank_ic = 0.0
            else:
                rank_ic_value = group["composite_score"].corr(group["realized_return"], method="spearman")
                rank_ic = float(rank_ic_value) if pd.notna(rank_ic_value) else 0.0
            by_horizon[str(horizon)] = {
                "sample_size": int(len(group)),
                "hit_rate": float(group["direction_correct"].mean()),
                "average_return": float(group["signed_return"].mean()),
                "average_vol_adjusted_return": float(group["volatility_adjusted_return"].mean()),
                "rank_ic": rank_ic,
                "brier_score": float(((group["confidence"] - group["direction_correct"].astype(float)) ** 2).mean()),
                "mean_mfe": float(group["max_favorable_excursion"].mean()),
                "mean_mae": float(group["max_adverse_excursion"].mean()),
            }

        confidence_buckets = (
            detailed_df.groupby("confidence_bucket")
            .agg(sample_size=("signal_id", "count"), hit_rate=("direction_correct", "mean"), average_return=("signed_return", "mean"))
            .reset_index()
            .to_dict(orient="records")
        )
        by_signal_class = (
            detailed_df.groupby(detailed_df["metadata"].map(lambda item: item.get("signal_category", "unknown")))
            .agg(sample_size=("signal_id", "count"), hit_rate=("direction_correct", "mean"), average_return=("signed_return", "mean"))
            .reset_index()
            .rename(columns={"metadata": "signal_category"})
            .to_dict(orient="records")
        )
        by_regime = (
            detailed_df.groupby("regime_label")
            .agg(sample_size=("signal_id", "count"), hit_rate=("direction_correct", "mean"), average_return=("signed_return", "mean"))
            .reset_index()
            .to_dict(orient="records")
        )
        confusion = self._confusion_matrix(detailed_df)
        precision_recall = self._precision_recall(detailed_df)
        degradation_alerts = self._degradation_alerts(detailed_df)

        if detailed_df["composite_score"].nunique() <= 1 or detailed_df["realized_return"].nunique() <= 1:
            overall_rank_ic = 0.0
        else:
            overall_rank_ic_value = detailed_df["composite_score"].corr(detailed_df["realized_return"], method="spearman")
            overall_rank_ic = float(overall_rank_ic_value) if pd.notna(overall_rank_ic_value) else 0.0
        summary_metrics = {
            "sample_size": int(len(detailed_df)),
            "overall_hit_rate": float(detailed_df["direction_correct"].mean()),
            "overall_average_return": float(detailed_df["signed_return"].mean()),
            "overall_rank_ic": overall_rank_ic,
            "overall_brier_score": float(((detailed_df["confidence"] - detailed_df["direction_correct"].astype(float)) ** 2).mean()),
            "event_window_hit_rate": float(detailed_df.loc[detailed_df["event_window_flag"], "direction_correct"].mean())
            if detailed_df["event_window_flag"].any()
            else None,
            "non_event_hit_rate": float(detailed_df.loc[~detailed_df["event_window_flag"], "direction_correct"].mean())
            if (~detailed_df["event_window_flag"]).any()
            else None,
        }
        scorecards = {
            "by_horizon": by_horizon,
            "confidence_buckets": confidence_buckets,
            "by_signal_class": by_signal_class,
            "by_regime": by_regime,
            "confusion_matrix": confusion,
            "precision_recall": precision_recall,
        }
        return summary_metrics, scorecards, degradation_alerts

    def _entry_index(self, index: pd.DatetimeIndex, timestamp: datetime, lag_bars: int = 0) -> Optional[int]:
        position = index.searchsorted(pd.Timestamp(timestamp), side="left")
        if position >= len(index):
            return None
        if index[position] == pd.Timestamp(timestamp):
            resolved = int(position)
        elif position > 0:
            resolved = int(position - 1)
        else:
            return None
        entry_position = resolved + max(0, lag_bars)
        if entry_position >= len(index):
            return None
        return int(entry_position)

    def _forward_return_path(
        self,
        price_data: pd.DataFrame,
        entry_idx: int,
        exit_idx: int,
        direction: str,
        rolling_median_volume: pd.Series,
    ) -> Tuple[float, float, pd.Series, Dict[str, object]]:
        closes = price_data["close"].astype(float)
        entry_row = price_data.iloc[entry_idx]
        exit_row = price_data.iloc[exit_idx]
        entry_median_volume = float(rolling_median_volume.iloc[entry_idx]) if len(rolling_median_volume) > entry_idx and pd.notna(rolling_median_volume.iloc[entry_idx]) else None
        exit_median_volume = float(rolling_median_volume.iloc[exit_idx]) if len(rolling_median_volume) > exit_idx and pd.notna(rolling_median_volume.iloc[exit_idx]) else None
        entry_price = execution_price(entry_row, direction=direction, phase="entry", median_volume=entry_median_volume)
        exit_price = execution_price(exit_row, direction=direction, phase="exit", median_volume=exit_median_volume)
        raw_return = float(exit_price / entry_price - 1.0)
        direction_sign = direction_to_sign(direction)
        signed_return = raw_return * direction_sign
        path = closes.iloc[entry_idx + 1 : exit_idx + 1] / entry_price - 1.0
        metadata = {
            "entry_timestamp": price_data.index[entry_idx].isoformat(),
            "exit_timestamp": price_data.index[exit_idx].isoformat(),
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "entry_price_field": settings.execution.entry_price_field,
            "exit_price_field": settings.execution.exit_price_field,
            "entry_slippage_bps_realized": float(slippage_rate(entry_row, phase="entry", median_volume=entry_median_volume) * 10000.0),
            "exit_slippage_bps_realized": float(slippage_rate(exit_row, phase="exit", median_volume=exit_median_volume) * 10000.0),
        }
        return raw_return, signed_return, path * direction_sign, metadata

    def _vol_adjusted_return(self, signed_return: float, path_returns: pd.Series) -> float:
        realized_vol = float(path_returns.std(ddof=0)) if len(path_returns) else 0.0
        if realized_vol == 0.0:
            return float(signed_return)
        return float(signed_return / realized_vol)

    def _reversal_probability(self, path_returns: pd.Series) -> float:
        if len(path_returns) == 0:
            return 0.0
        favorable = path_returns.max() > 0
        finished_negative = path_returns.iloc[-1] <= 0
        return 1.0 if favorable and finished_negative else 0.0

    def _event_window_flag(self, timestamp: datetime, horizon: int, macro_events: List[MacroEvent]) -> bool:
        end_ts = pd.Timestamp(timestamp) + pd.Timedelta(days=horizon)
        return any(pd.Timestamp(timestamp) <= event.timestamp <= end_ts for event in macro_events)

    def _realized_regime(self, raw_return: float, forward_returns: pd.Series) -> str:
        realized_vol = float(forward_returns.std(ddof=0)) if len(forward_returns) else 0.0
        if raw_return > 0.01 and realized_vol < 0.02:
            return "trend_following_bullish"
        if raw_return < -0.01 and realized_vol < 0.02:
            return "trend_following_bearish"
        if realized_vol > 0.03:
            return "volatile_reversal"
        return "mean_reverting_rangebound"

    def _confidence_bucket(self, confidence: float) -> str:
        bucket_count = settings.evaluation.confidence_buckets
        index = min(bucket_count - 1, int(confidence * bucket_count))
        lower = index / bucket_count
        upper = (index + 1) / bucket_count
        return f"{lower:.2f}-{upper:.2f}"

    def _direction_bucket(self, raw_return: float) -> str:
        if raw_return > 0.001:
            return "long"
        if raw_return < -0.001:
            return "short"
        return "neutral"

    def _confusion_matrix(self, detailed_df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
        actual = detailed_df["realized_return"].map(self._direction_bucket)
        predicted = detailed_df["direction"]
        matrix: Dict[str, Dict[str, int]] = {}
        for pred_label in ("long", "short", "neutral"):
            matrix[pred_label] = {}
            for actual_label in ("long", "short", "neutral"):
                matrix[pred_label][actual_label] = int(((predicted == pred_label) & (actual == actual_label)).sum())
        return matrix

    def _precision_recall(self, detailed_df: pd.DataFrame) -> Dict[str, float]:
        actual = detailed_df["realized_return"].map(self._direction_bucket)
        predicted = detailed_df["direction"]
        long_tp = int(((predicted == "long") & (actual == "long")).sum())
        long_fp = int(((predicted == "long") & (actual != "long")).sum())
        long_fn = int(((predicted != "long") & (actual == "long")).sum())
        precision = long_tp / (long_tp + long_fp) if (long_tp + long_fp) else 0.0
        recall = long_tp / (long_tp + long_fn) if (long_tp + long_fn) else 0.0
        return {"long_precision": precision, "long_recall": recall}

    def _degradation_alerts(self, detailed_df: pd.DataFrame) -> List[str]:
        ordered = detailed_df.sort_values("timestamp")
        window = settings.evaluation.degradation_window_signals
        if len(ordered) < window:
            return []
        trailing = ordered.tail(window)
        alerts = []
        if trailing["direction_correct"].mean() + 0.10 < ordered["direction_correct"].mean():
            alerts.append("Recent hit rate is materially below the full-history average.")
        if trailing["signed_return"].mean() < 0 and ordered["signed_return"].mean() > 0:
            alerts.append("Recent realized signed returns turned negative after a positive longer-run average.")
        return alerts

    def _serialize_snapshot(self, snapshot: SignalSnapshot) -> Dict[str, object]:
        payload = snapshot.to_dict()
        for field_name in ("directional_scores", "key_drivers", "key_risks", "component_scores", "feature_vector", "metadata"):
            payload[field_name] = json.dumps(payload[field_name], default=str)
        return payload

    def _serialize_evaluation_record(self, record: SignalEvaluationRecord) -> Dict[str, object]:
        payload = record.to_dict()
        payload["metadata"] = json.dumps(payload["metadata"], default=str)
        return payload

    def _deserialize_payload(self, payload):
        if isinstance(payload, str):
            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                return payload
        return payload
