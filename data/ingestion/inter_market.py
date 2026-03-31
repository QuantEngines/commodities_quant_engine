"""
Inter-market data sources — COMEX and LBME metals exchanges.

Enables cross-exchange triangulation and basis/arbitrage trading:
- COMEX (NYMEX) — Crude oil, Gold, Silver, Copper futures (US markets)
- LBME (London Bullion) — Gold, Silver, Platinum (London cash market)

Supports basis trades: long MCX, short COMEX (or vice versa)
Supports arbitrage: Monitor cross-exchange price discrepancies
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from .base import DataSource
from .local_first import LocalFirstDataSource
from ..data.models import Contract, OHLCV


class COMEXDataSource(LocalFirstDataSource):
    """
    COMEX (NYMEX) futures contracts — US commodity exchange.

    Supports: Crude Oil (CL), Gold (GC), Silver (SI), Copper (HG), Natural Gas (NG)
    Provides: Liquid financial futures for basis trading vs physical markets
    """

    exchange_code = "COMEX"
    default_segment = "futures"

    def __init__(self, config: Dict[str, Any]):
        # Default COMEX contract multipliers
        defaults = {
            "lot_sizes": {
                "CL": 1000,   # Crude Oil: 1000 bbl
                "GC": 100,    # Gold: 100 troy oz
                "SI": 5000,   # Silver: 5000 troy oz
                "HG": 25000,  # Copper: 25000 lbs
                "NG": 10000,  # Natural Gas: 10000 MMBtu
            },
            "tick_sizes": {
                "CL": 0.01,
                "GC": 0.10,
                "SI": 0.005,
                "HG": 0.0005,
                "NG": 0.001,
            },
            "multipliers": {
                "CL": 1000,
                "GC": 100,
                "SI": 5000,
                "HG": 25000,
                "NG": 10000,
            },
            # Commodity ticker mappings (e.g., for Yahoo Finance proxy)
            "free_tickers": {
                "CL": "CL=F",      # Crude Oil
                "GC": "GC=F",      # Gold
                "SI": "SI=F",      # Silver
                "HG": "HG=F",      # Copper
                "NG": "NG=F",      # Natural Gas
            },
        }
        merged = {**defaults, **config}
        merged["free_tickers"] = {
            **defaults["free_tickers"],
            **config.get("free_tickers", {}),
        }
        super().__init__(merged)

    def fetch_contract_master(
        self, commodity: str, as_of_date: Optional[date] = None
    ) -> List[Contract]:
        """Fetch active COMEX contracts for commodity."""
        as_of_date = as_of_date or date.today()

        # COMEX contracts typically expire on 3rd business day of contract month
        # For simplicity, approximate with typical expiry patterns
        contracts = []
        commodity_upper = (commodity or "").upper()

        # Map MCX commodities to COMEX contracts
        commodity_map = {
            "GOLD": "GC",
            "GOLDM": "GC",
            "SILVER": "SI",
            "SILVERM": "SI",
            "COPPER": "HG",
            "COPPERM": "HG",
            "CRUDEOIL": "CL",
            "NATURALGAS": "NG",
        }

        comex_symbol = commodity_map.get(commodity_upper)
        if not comex_symbol:
            return []

        # Generate 3 front contracts
        for offset in range(3):
            anchor = self._month_anchor(as_of_date, offset)
            month_code = self._month_code(anchor)
            symbol = f"{comex_symbol}{month_code}"

            contracts.append(
                Contract(
                    commodity=comex_symbol,
                    symbol=symbol,
                    expiry_date=anchor + timedelta(
                        days=3
                    ),  # Approximate 3rd business day
                    lot_size=int(self.config.get("lot_sizes", {}).get(comex_symbol, 1)),
                    tick_size=float(
                        self.config.get("tick_sizes", {}).get(comex_symbol, 0.01)
                    ),
                    multiplier=int(
                        self.config.get("multipliers", {}).get(comex_symbol, 1)
                    ),
                    exchange=self.exchange_code,
                    segment=self.default_segment,
                    source="comex",
                )
            )

        return contracts

    def _month_code(self, dt: date) -> str:
        """COMEX month code: F=Jan, G=Feb, ..., Z=Dec"""
        codes = "FGHJKMNQUVXZ"
        return codes[dt.month - 1]

    def _month_anchor(self, base_date: date, offset: int = 0) -> date:
        """Calculate contract month anchor (simplified)."""
        month = (base_date.month - 1 + offset) % 12 + 1
        year = base_date.year + (base_date.month - 1 + offset) // 12
        return date(year, month, 1)


class LBMEDataSource(LocalFirstDataSource):
    """
    LBME (London Bullion Market Exchange) — Physical metals spot prices.

    Provides: London Fix prices for Gold, Silver, Platinum (cash market)
    Use for: Spot-futures basis trading, arbitrage signals
    """

    exchange_code = "LBME"
    default_segment = "spot"

    def __init__(self, config: Dict[str, Any]):
        defaults = {
            # LBME spot market (no futures), so lot_size = 1 (spot quote)
            "lot_sizes": {
                "GOLD": 1,      # USD per troy oz
                "SILVER": 1,    # USD per troy oz
                "PLATINUM": 1,  # USD per troy oz
            },
            "tick_sizes": {
                "GOLD": 0.01,
                "SILVER": 0.001,
                "PLATINUM": 0.01,
            },
            "multipliers": {
                "GOLD": 1,
                "SILVER": 1,
                "PLATINUM": 1,
            },
            "free_tickers": {
                "GOLD": "GC=F",       # Yahoo Finance Gold proxy (COMEX futures)
                "SILVER": "SI=F",     # Yahoo Finance Silver proxy
                "PLATINUM": "PL=F",   # Yahoo Finance Platinum proxy
            },
        }
        merged = {**defaults, **config}
        super().__init__(merged)

    def fetch_contract_master(
        self, commodity: str, as_of_date: Optional[date] = None
    ) -> List[Contract]:
        """LBME is spot market (no expiring contracts)."""
        as_of_date = as_of_date or date.today()

        # Single "contract" representing the spot fix
        if commodity.upper() not in ["GOLD", "SILVER", "PLATINUM"]:
            return []

        return [
            Contract(
                commodity=commodity,
                symbol=f"{commodity}_LBME_SPOT",
                expiry_date=as_of_date + timedelta(days=365),  # Pseudo-expiry
                lot_size=1,
                tick_size=0.01,
                multiplier=1,
                exchange=self.exchange_code,
                segment=self.default_segment,
                source="lbme_spot",
            )
        ]

    def fetch_ohlcv(
        self, contract: str, start_date: date, end_date: date
    ) -> List[OHLCV]:
        """Fetch LBME spot price history."""
        # LBME provides once-daily fix (typically 3pm London time)
        # Load from local source if available, else return empty
        return super().fetch_ohlcv(contract, start_date, end_date)

    def fetch_spot(
        self, commodity: str, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """Fetch London spot price history."""
        # Direct spot data (not futures)
        return super().fetch_spot(commodity, start_date, end_date)


class InterMarketBasisCalculator:
    """
    Computes basis (spread) between markets for arbitrage signals.

    Basis = Spot Price - Futures Price (typically)
    Used to identify rich/cheap futures, delivery disruptions, etc.
    """

    def __init__(self):
        self.mcx_source: Optional[DataSource] = None
        self.comex_source: Optional[DataSource] = None
        self.lbme_source: Optional[DataSource] = None

    def calculate_commodity_basis(
        self,
        commodity: str,
        mcx_price: float,
        comex_price: float,
        fx_rate: float = 83.0,  # INR/USD default
    ) -> Dict[str, float]:
        """
        Calculate basis between MCX and COMEX.

        Args:
            commodity: Commodity symbol (e.g., "GOLD")
            mcx_price: MCX price in local currency
            comex_price: COMEX price in USD
            fx_rate: INR/USD exchange rate for conversion

        Returns:
            Dict with basis metrics: {basis_points, basis_pct, status}
        """
        # Normalize prices to same currency (USD)
        mcx_price_usd = mcx_price / fx_rate
        basis = mcx_price_usd - comex_price
        basis_pct = (basis / comex_price) * 100 if comex_price != 0 else 0

        return {
            "commodity": commodity,
            "mcx_price_usd": mcx_price_usd,
            "comex_price": comex_price,
            "basis": basis,
            "basis_bps": basis * 10000,  # Basis points
            "basis_pct": basis_pct,
            "is_mcx_rich": basis > 0,  # MCX trading at premium
            "is_mcx_cheap": basis < 0,  # MCX trading at discount
        }

    def get_arbitrage_opportunity(
        self, basis_history: pd.DataFrame, commodity: str
    ) -> Dict:
        """
        Identify potential arbitrage opportunities.

        Returns signal if basis significantly deviates from historical mean.
        """
        if basis_history.empty:
            return {"status": "no_data"}

        basis = basis_history["basis"].iloc[-1]
        mean_basis = basis_history["basis"].mean()
        std_basis = basis_history["basis"].std()

        z_score = (basis - mean_basis) / (std_basis + 1e-8)

        arbitrage_signal = "none"
        if z_score > 2.0:
            arbitrage_signal = "short_mcx"  # MCX rich vs COMEX
        elif z_score < -2.0:
            arbitrage_signal = "long_mcx"  # MCX cheap vs COMEX

        return {
            "commodity": commodity,
            "basis": basis,
            "mean_basis": mean_basis,
            "std_basis": std_basis,
            "z_score": z_score,
            "arbitrage_signal": arbitrage_signal,
            "timestamp": basis_history.index[-1],
        }


# Registry entries for data source manager
comex_data_source = COMEXDataSource({})
lbme_data_source = LBMEDataSource({})
inter_market_basis_calculator = InterMarketBasisCalculator()
