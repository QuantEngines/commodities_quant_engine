from __future__ import annotations

from typing import Any, Dict

from .local_first import LocalFirstDataSource


class MOSPIDataSource(LocalFirstDataSource):
    """MOSPI adapter for free official data files or configured public URLs."""

    exchange_code = "MOSPI"
    default_segment = "macro"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
