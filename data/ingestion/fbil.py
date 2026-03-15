from __future__ import annotations

from typing import Any, Dict

from .local_first import LocalFirstDataSource


class FBILDataSource(LocalFirstDataSource):
    """FBIL/RBI reference-rate adapter using free local/public inputs."""

    exchange_code = "FBIL"
    default_segment = "reference_rates"

    def __init__(self, config: Dict[str, Any]):
        defaults = {
            "free_tickers": {
                "INR/USD": "INR=X",
                "USD/INR": "INR=X",
                "FX_USD_INR": "INR=X",
            }
        }
        merged = {**defaults, **config}
        merged["free_tickers"] = {**defaults["free_tickers"], **config.get("free_tickers", {})}
        super().__init__(merged)
