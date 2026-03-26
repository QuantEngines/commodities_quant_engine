from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Sequence

from ...nlp.schemas import CommodityEvent


def normalize_event_dict(payload: Dict[str, Any], commodity_scope: Sequence[str], source_id: str, raw_text: str) -> CommodityEvent:
    normalized = dict(payload)
    normalized.setdefault("commodity_scope", list(commodity_scope))
    normalized.setdefault("source_id", source_id)
    normalized.setdefault("timestamp", datetime.utcnow())
    normalized.setdefault("raw_text", raw_text)
    normalized.setdefault("summary", "normalized_llm_event")
    normalized.setdefault("entities_keywords", [])
    return CommodityEvent.model_validate(normalized)
