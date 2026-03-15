from __future__ import annotations

from typing import Any, Dict

from .local_first import LocalFirstDataSource


class NSEDataSource(LocalFirstDataSource):
    """NSE commodity derivatives adapter with free/local-first support."""

    exchange_code = "NSE"
    default_segment = "commodity_derivatives"

    def __init__(self, config: Dict[str, Any]):
        defaults = {
            "free_tickers": {
                "GOLD": "GC=F",
                "SILVER": "SI=F",
                "CRUDEOIL": "CL=F",
                "COPPER": "HG=F",
            }
        }
        merged = {**defaults, **config}
        merged["free_tickers"] = {**defaults["free_tickers"], **config.get("free_tickers", {})}
        super().__init__(merged)
