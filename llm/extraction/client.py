from __future__ import annotations

import json
import os
from typing import Optional, Sequence

import requests

from ..prompts.event_extraction_prompt import (
    EVENT_EXTRACTION_SYSTEM_PROMPT,
    build_event_extraction_user_prompt,
)


class LLMInferenceClient:
    """Optional OpenAI-compatible LLM client used for runtime event extraction."""

    def __init__(self):
        self.provider = os.getenv("LLM_PROVIDER", "openai_compatible").strip().lower()
        self.api_key = os.getenv("LLM_API_KEY", "").strip()
        self.base_url = os.getenv("LLM_BASE_URL", "https://api.openai.com/v1").strip().rstrip("/")
        self.model = os.getenv("LLM_MODEL", "gpt-4o-mini").strip()
        self.timeout_seconds = int(os.getenv("LLM_TIMEOUT_SECONDS", "20"))

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def generate_event_json(
        self,
        raw_text: str,
        commodity_scope: Sequence[str],
        source_id: str,
    ) -> Optional[str]:
        if not self.enabled:
            return None
        if self.provider != "openai_compatible":
            return None

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": EVENT_EXTRACTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_event_extraction_user_prompt(
                        raw_text=raw_text,
                        commodity_scope=list(commodity_scope),
                        source_id=source_id,
                    ),
                },
            ],
        }
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=self.timeout_seconds)
            response.raise_for_status()
            body = response.json()
            choices = body.get("choices", [])
            if not choices:
                return None
            message = choices[0].get("message", {})
            content = message.get("content")
            return content if isinstance(content, str) and content.strip() else None
        except Exception:
            return None
