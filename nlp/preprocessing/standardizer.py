from __future__ import annotations

import re


_WHITESPACE_RE = re.compile(r"\s+")


def standardize_text(text: str) -> str:
    cleaned = text.replace("\n", " ").replace("\t", " ").strip().lower()
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)
    return cleaned
