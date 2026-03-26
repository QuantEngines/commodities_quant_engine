from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from ...config.settings import settings


@dataclass
class EventOverlayCalibration:
    directional_weight: float
    regime_weight: float
    risk_weight: float


def _commodity_family(symbol: str) -> str:
    commodity_cfg = settings.commodities.get(symbol)
    if commodity_cfg is None:
        return "default"
    segment = str(commodity_cfg.segment).lower()
    if segment in {"energy", "base_metals", "bullion", "agri"}:
        return segment
    return "default"


def _family_priors() -> Dict[str, EventOverlayCalibration]:
    return {
        "energy": EventOverlayCalibration(0.24, 0.26, 0.34),
        "base_metals": EventOverlayCalibration(0.18, 0.22, 0.27),
        "bullion": EventOverlayCalibration(0.14, 0.20, 0.24),
        "agri": EventOverlayCalibration(0.22, 0.18, 0.30),
        "default": EventOverlayCalibration(0.20, 0.20, 0.30),
    }


def _summary_files() -> Iterable[Path]:
    evaluations_dir = Path(settings.storage.base_dir) / settings.storage.evaluation_store
    if not evaluations_dir.exists():
        return []
    return sorted(evaluations_dir.glob("*_summary.json"))


def _read_json(path: Path) -> Dict[str, object]:
    import json

    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def calibrate_overlay_weights_for_commodity(commodity: str) -> EventOverlayCalibration:
    family = _commodity_family(commodity)
    priors = _family_priors()
    prior = priors.get(family, priors["default"])

    hit_rates: List[float] = []
    event_edges: List[float] = []
    for path in _summary_files():
        symbol = path.name.replace("_summary.json", "")
        if _commodity_family(symbol) != family:
            continue
        payload = _read_json(path)
        summary = payload.get("summary_metrics", {}) if isinstance(payload, dict) else {}
        if not isinstance(summary, dict):
            continue
        overall_hit = summary.get("overall_hit_rate")
        event_hit = summary.get("event_window_hit_rate")
        non_event_hit = summary.get("non_event_hit_rate")
        if isinstance(overall_hit, (int, float)):
            hit_rates.append(float(overall_hit))
        if isinstance(event_hit, (int, float)) and isinstance(non_event_hit, (int, float)):
            event_edges.append(float(event_hit) - float(non_event_hit))

    if not hit_rates:
        return prior

    mean_hit = float(np.mean(hit_rates))
    mean_edge = float(np.mean(event_edges)) if event_edges else 0.0

    directional_scale = np.clip(0.85 + (mean_hit - 0.5) * 1.2 + mean_edge * 0.8, 0.65, 1.35)
    regime_scale = np.clip(0.90 + (mean_hit - 0.5) * 0.8 + mean_edge * 0.6, 0.70, 1.30)
    risk_scale = np.clip(1.05 - mean_edge * 0.9, 0.70, 1.40)

    return EventOverlayCalibration(
        directional_weight=float(np.clip(prior.directional_weight * directional_scale, 0.08, 0.40)),
        regime_weight=float(np.clip(prior.regime_weight * regime_scale, 0.08, 0.40)),
        risk_weight=float(np.clip(prior.risk_weight * risk_scale, 0.10, 0.55)),
    )
