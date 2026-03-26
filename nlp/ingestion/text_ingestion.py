from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Mapping, Union


@dataclass
class TextRecord:
    source_id: str
    timestamp: datetime
    headline: str
    body: str
    source: str


def _from_mapping(item: Mapping[str, object], idx: int) -> TextRecord:
    ts = item.get("timestamp")
    if isinstance(ts, datetime):
        timestamp = ts
    elif isinstance(ts, str):
        timestamp = datetime.fromisoformat(ts)
    else:
        timestamp = datetime.utcnow()
    return TextRecord(
        source_id=str(item.get("source_id", f"record_{idx}")),
        timestamp=timestamp,
        headline=str(item.get("headline", "")).strip(),
        body=str(item.get("body", "")).strip(),
        source=str(item.get("source", "unknown")),
    )


def normalize_text_records(items: Iterable[Union[str, Mapping[str, object], TextRecord]]) -> List[TextRecord]:
    records: List[TextRecord] = []
    for idx, item in enumerate(items):
        if isinstance(item, TextRecord):
            records.append(item)
            continue
        if isinstance(item, str):
            records.append(
                TextRecord(
                    source_id=f"record_{idx}",
                    timestamp=datetime.utcnow(),
                    headline=item.strip(),
                    body="",
                    source="raw_text",
                )
            )
            continue
        if isinstance(item, Mapping):
            records.append(_from_mapping(item, idx))
    return [record for record in records if record.headline]
