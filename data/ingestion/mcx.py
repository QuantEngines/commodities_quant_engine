from __future__ import annotations

from typing import Dict, Any

from ...config.commodity_universe import default_mcx_commodity_definitions
from .local_first import LocalFirstDataSource


class MCXDataSource(LocalFirstDataSource):
    """MCX adapter with free/local-first loading and optional public ticker fallback."""

    exchange_code = "MCX"
    default_segment = "commodities"

    def __init__(self, config: Dict[str, Any]):
        default_definitions = default_mcx_commodity_definitions()
        defaults = {
            "lot_sizes": {symbol: int(meta["contract_multiplier"]) for symbol, meta in default_definitions.items()},
            "tick_sizes": {symbol: float(meta["tick_size"]) for symbol, meta in default_definitions.items()},
            "multipliers": {symbol: int(meta["contract_multiplier"]) for symbol, meta in default_definitions.items()},
            "free_tickers": {
                "GOLD": "GC=F",
                "GOLDM": "GC=F",
                "GOLDGUINEA": "GC=F",
                "GOLDPETAL": "GC=F",
                "GOLDTEN": "GC=F",
                "SILVER": "SI=F",
                "SILVERM": "SI=F",
                "SILVERMIC": "SI=F",
                "SILVER1000": "SI=F",
                "COPPER": "HG=F",
                "COPPERM": "HG=F",
                "CRUDEOIL": "CL=F",
                "CRUDEOILM": "CL=F",
                "BRCRUDEOIL": "BZ=F",
                "NATURALGAS": "NG=F",
                "NATGASMINI": "NG=F",
                "COTTON": "CT=F",
                "COTTONCNDY": "CT=F",
                "WHEAT": "ZW=F",
                "SOYABEAN": "ZS=F",
                "INR/USD": "INR=X",
            },
        }
        merged = {**defaults, **config}
        merged["free_tickers"] = {**defaults["free_tickers"], **config.get("free_tickers", {})}
        super().__init__(merged)
