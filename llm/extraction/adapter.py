from __future__ import annotations

import json
from typing import Any, Dict, Optional, Sequence

from ...llm.validators import normalize_event_dict
from ...nlp.schemas import CommodityEvent


class LLMExtractionAdapter:
    """Optional adapter for plugging external LLM JSON output into validated event schema."""

    def parse_or_none(
        self,
        llm_json: Optional[str],
        commodity_scope: Sequence[str],
        source_id: str,
        raw_text: str,
    ) -> Optional[CommodityEvent]:
        if not llm_json:
            return None
        try:
            payload: Dict[str, Any] = json.loads(llm_json)
        except json.JSONDecodeError:
            return None
        try:
            return normalize_event_dict(payload, commodity_scope=commodity_scope, source_id=source_id, raw_text=raw_text)
        except Exception:
            return None
