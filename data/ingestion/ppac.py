from __future__ import annotations

from typing import Any, Dict

from .local_first import LocalFirstDataSource


class PPACDataSource(LocalFirstDataSource):
    """PPAC energy-price adapter using free local/public tabular data."""

    exchange_code = "PPAC"
    default_segment = "energy_reference"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
