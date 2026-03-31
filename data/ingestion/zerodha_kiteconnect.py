"""
Zerodha KiteConnect API adapter for live/historical commodity market data.

Prerequisites:
- pip install zerodha-kiteconnect
- Zerodha KiteConnect subscription (https://kite.zerodha.com/docs/connect/)
- API key, API secret, and access token configured
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd

from ..models import Contract, OHLCV
from .base import DataSource

logger = logging.getLogger(__name__)


class ZerodhaKiteConnectDataSource(DataSource):
    """
    Zerodha KiteConnect API adapter for MCX commodities.
    
    Provides live and historical OHLCV, contract master, and spot data.
    Falls back gracefully if credentials are missing or API is unavailable.
    """

    exchange_code = "MCX"
    default_segment = "commodities"

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.api_key = config.get("api_key", "")
        self.api_secret = config.get("api_secret", "")
        self.access_token = config.get("access_token", "")
        self.fallback_enabled = config.get("fallback_enabled", True)
        self.kite = None
        self.enabled = bool(self.api_key and self.api_secret and self.access_token)
        
        if self.enabled:
            self._init_kite()
        else:
            logger.warning(
                "Zerodha KiteConnect disabled: missing api_key, api_secret, or access_token. "
                "Set these in config or environment variables."
            )

    def _init_kite(self) -> None:
        """Initialize KiteConnect client safely."""
        try:
            from kiteconnect import KiteConnect
            
            self.kite = KiteConnect(api_key=self.api_key)
            self.kite.set_access_token(self.access_token)
            logger.info("Zerodha KiteConnect initialized successfully")
        except ImportError:
            logger.error(
                "zerodha-kiteconnect not installed. "
                "Install with: pip install zerodha-kiteconnect"
            )
            self.enabled = False
        except Exception as e:
            logger.error(f"Failed to initialize KiteConnect: {e}")
            self.enabled = False

    def fetch_contract_master(
        self, commodity: str, as_of_date: Optional[date] = None
    ) -> List[Contract]:
        """
        Fetch active contracts for a commodity from KiteConnect.
        
        Returns the 3 nearest contract expiries (current month + 2 next).
        """
        if not self.enabled or not self.kite:
            return []

        try:
            as_of_date = as_of_date or date.today()
            contracts = []
            
            # Fetch available instruments from KiteConnect
            instruments = self.kite.instruments("MCX")
            
            # Filter for this commodity and extract active contracts
            for instrument in instruments:
                instrument_name = instrument.get("tradingsymbol", "")
                
                # Simple heuristic: MCX commodity contracts contain "-" separator
                # e.g., "GOLDGUINENOV2025", "SILVER-NOV2025"
                # Filter by commodity symbol
                if commodity in instrument_name and "MCX" in instrument.get("exchange", ""):
                    expiry_str = instrument.get("expiry")
                    if not expiry_str:
                        continue
                    
                    try:
                        expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d").date()
                        # Include only future contracts (not already expired)
                        if expiry_date >= as_of_date:
                            contracts.append(
                                Contract(
                                    commodity=commodity,
                                    symbol=instrument_name,
                                    expiry_date=expiry_date,
                                    lot_size=int(instrument.get("lot_size", 1)),
                                    tick_size=float(instrument.get("tick_size", 0.01)),
                                    multiplier=int(instrument.get("multiplier", 1)) or 1,
                                    exchange=self.exchange_code,
                                    segment=self.default_segment,
                                    quote_currency="INR",
                                    source="zerodha_kiteconnect",
                                    metadata={
                                        "instrument_token": instrument.get("instrument_token"),
                                        "name": instrument.get("name"),
                                    },
                                )
                            )
                    except (ValueError, TypeError) as e:
                        logger.debug(f"Skipping malformed contract {instrument_name}: {e}")
            
            # Sort by expiry and return top 3
            contracts.sort(key=lambda c: c.expiry_date)
            return contracts[:3]
        except Exception as e:
            logger.error(f"Error fetching contract master: {e}")
            return []

    def fetch_ohlcv(
        self, contract: str, start_date: date, end_date: date
    ) -> List[OHLCV]:
        """
        Fetch historical OHLCV data from KiteConnect.
        
        Args:
            contract: Contract symbol (e.g., "GOLDGUINENOV2025")
            start_date: Start date (inclusive)
            end_date: End date (inclusive)
        
        Returns:
            List of OHLCV data points
        """
        if not self.enabled or not self.kite:
            return []

        try:
            # Get instrument token for the contract
            instruments = self.kite.instruments("MCX")
            instrument_token = None
            for instrument in instruments:
                if instrument.get("tradingsymbol") == contract:
                    instrument_token = instrument.get("instrument_token")
                    break
            
            if not instrument_token:
                logger.warning(f"Contract {contract} not found in MCX instruments")
                return []
            
            # Fetch historical data (interval="day" for daily candles)
            candles = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=datetime.combine(start_date, datetime.min.time()),
                to_date=datetime.combine(end_date, datetime.max.time()),
                interval="day",
            )
            
            ohlcv_list = []
            for candle in candles:
                try:
                    ohlcv_list.append(
                        OHLCV(
                            timestamp=datetime.fromisoformat(
                                candle["date"].isoformat()
                            ),
                            open=float(candle["open"]),
                            high=float(candle["high"]),
                            low=float(candle["low"]),
                            close=float(candle["close"]),
                            volume=int(candle.get("volume", 0)),
                            open_interest=int(candle.get("oi"))
                            if "oi" in candle and candle["oi"]
                            else None,
                            contract=contract,
                        )
                    )
                except (ValueError, KeyError, TypeError) as e:
                    logger.debug(f"Skipping malformed candle: {e}")
            
            return ohlcv_list
        except Exception as e:
            logger.error(f"Error fetching OHLCV for {contract}: {e}")
            return []

    def fetch_open_interest(
        self, contract: str, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """
        Fetch open interest data from KiteConnect.
        
        Returns a DataFrame with open_interest indexed by timestamp.
        """
        if not self.enabled or not self.kite:
            return pd.DataFrame()

        try:
            # Fetch historical data to extract OI
            ohlcv_list = self.fetch_ohlcv(contract, start_date, end_date)
            if not ohlcv_list:
                return pd.DataFrame()
            
            df = pd.DataFrame(
                [
                    {
                        "timestamp": ohlcv.timestamp,
                        "open_interest": ohlcv.open_interest,
                    }
                    for ohlcv in ohlcv_list
                    if ohlcv.open_interest is not None
                ]
            )
            
            if df.empty:
                return df
            
            df.set_index("timestamp", inplace=True)
            return df[["open_interest"]]
        except Exception as e:
            logger.error(f"Error fetching open interest for {contract}: {e}")
            return pd.DataFrame()

    def fetch_spot(
        self, commodity: str, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """
        Fetch spot price data from KiteConnect.
        
        Falls back to empty DataFrame if not available.
        """
        if not self.enabled or not self.kite:
            return pd.DataFrame()

        try:
            # KiteConnect doesn't directly provide spot prices separately
            # Use the nearest contract for a spot-proxy
            contracts = self.fetch_contract_master(commodity, start_date)
            if not contracts:
                return pd.DataFrame()
            
            nearest_contract = contracts[0]
            ohlcv_list = self.fetch_ohlcv(nearest_contract.symbol, start_date, end_date)
            
            if not ohlcv_list:
                return pd.DataFrame()
            
            df = pd.DataFrame(
                [
                    {
                        "timestamp": ohlcv.timestamp,
                        "price": ohlcv.close,
                        "volume": ohlcv.volume,
                    }
                    for ohlcv in ohlcv_list
                ]
            )
            
            df.set_index("timestamp", inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error fetching spot for {commodity}: {e}")
            return pd.DataFrame()

    def fetch_reference_rates(
        self, currency_pair: str, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """
        Fetch FX reference rates from KiteConnect.
        
        Supports pairs like "INR/USD", "INR/EUR", etc.
        """
        if not self.enabled or not self.kite:
            return pd.DataFrame()

        try:
            # Map common currency pairs to KiteConnect symbols
            fx_symbol_map = {
                "INR/USD": "USDINR",
                "USD/INR": "USDINR",
                "INR/EUR": "EURINR",
                "EUR/INR": "EURINR",
                "INR/GBP": "GBPINR",
                "GBP/INR": "GBPINR",
                "INR/JPY": "JPYINR",
                "JPY/INR": "JPYINR",
            }
            
            symbol = fx_symbol_map.get(currency_pair.upper())
            if not symbol:
                logger.warning(f"FX pair {currency_pair} not supported")
                return pd.DataFrame()
            
            # Fetch as FOREX instrument from KiteConnect
            instruments = self.kite.instruments("FOREX")
            instrument_token = None
            for instrument in instruments:
                if symbol in instrument.get("tradingsymbol", ""):
                    instrument_token = instrument.get("instrument_token")
                    break
            
            if not instrument_token:
                logger.debug(f"FX symbol {symbol} not found in FOREX instruments")
                return pd.DataFrame()
            
            candles = self.kite.historical_data(
                instrument_token=instrument_token,
                from_date=datetime.combine(start_date, datetime.min.time()),
                to_date=datetime.combine(end_date, datetime.max.time()),
                interval="day",
            )
            
            df = pd.DataFrame(
                [
                    {
                        "timestamp": datetime.fromisoformat(candle["date"].isoformat()),
                        "rate": candle["close"],
                    }
                    for candle in candles
                ]
            )
            
            if df.empty:
                return df
            
            df.set_index("timestamp", inplace=True)
            return df
        except Exception as e:
            logger.error(f"Error fetching FX rates for {currency_pair}: {e}")
            return pd.DataFrame()

    def fetch_macro_series(
        self, series_name: str, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """
        KiteConnect does not provide macro series data.
        
        This is a fallback-only method. Use LocalFirstDataSource for macros.
        """
        logger.debug(
            f"Macro series {series_name} not available via KiteConnect; "
            "use LocalFirstDataSource instead"
        )
        return pd.DataFrame()

    def fetch_weather(
        self, location: str, start_date: date, end_date: date
    ) -> pd.DataFrame:
        """
        KiteConnect does not provide weather data.
        
        This is a fallback-only method. Use external weather APIs.
        """
        logger.debug(
            f"Weather data for {location} not available via KiteConnect; "
            "use external weather provider instead"
        )
        return pd.DataFrame()

    def fetch_calendar(self, exchange: str, year: int) -> List[date]:
        """
        Fetch MCX trading calendar for the given year.
        
        Falls back to business days if not available.
        """
        if not self.enabled or not self.kite:
            # Fallback: business days excluding weekends
            business_days = pd.date_range(
                start=f"{year}-01-01", end=f"{year}-12-31", freq="B"
            )
            return [ts.date() for ts in business_days]

        try:
            # KiteConnect doesn't expose exchange calendar directly
            # Approximate with business days minus known MCX holidays
            mcx_holidays_2026 = [
                date(2026, 1, 26),  # Republic Day
                date(2026, 3, 25),  # Holi
                date(2026, 4, 2),   # Good Friday
                date(2026, 5, 1),   # May Day
                date(2026, 8, 15),  # Independence Day
                date(2026, 10, 2),  # Gandhi Jayanti
                date(2026, 10, 25), # Diwali (approx)
                date(2026, 12, 25), # Christmas
            ]
            
            business_days = pd.date_range(
                start=f"{year}-01-01", end=f"{year}-12-31", freq="B"
            )
            return [
                ts.date() for ts in business_days
                if ts.date() not in mcx_holidays_2026
            ]
        except Exception as e:
            logger.error(f"Error fetching calendar: {e}")
            return []
