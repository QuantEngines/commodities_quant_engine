from __future__ import annotations

from typing import Any, Dict

from .local_first import LocalFirstDataSource


class NCDEXDataSource(LocalFirstDataSource):
    """NCDEX adapter using local files first and free public proxies when configured."""

    exchange_code = "NCDEX"
    default_segment = "agri"

    def __init__(self, config: Dict[str, Any]):
        defaults = {
            "lot_sizes": {"COTTON": 25, "SOYBEAN": 10, "WHEAT": 10},
            "tick_sizes": {"COTTON": 0.20, "SOYBEAN": 1.0, "WHEAT": 1.0},
            "multipliers": {"COTTON": 25, "SOYBEAN": 10, "WHEAT": 10},
            "free_tickers": {
                "COTTON": "CT=F",
                "SOYBEAN": "ZS=F",
                "WHEAT": "ZW=F",
                "INR/USD": "INR=X",
            },
        }
        merged = {**defaults, **config}
        merged["free_tickers"] = {**defaults["free_tickers"], **config.get("free_tickers", {})}
        super().__init__(merged)
