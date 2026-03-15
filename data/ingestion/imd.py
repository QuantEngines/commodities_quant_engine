from __future__ import annotations

from typing import Any, Dict

from .local_first import LocalFirstDataSource


class IMDDataSource(LocalFirstDataSource):
    """IMD weather adapter using free local/public tabular data."""

    exchange_code = "IMD"
    default_segment = "weather"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
