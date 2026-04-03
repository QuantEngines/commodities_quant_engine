from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Any
from datetime import date
import pandas as pd

from ...data.models import Contract, OHLCV

class DataSource(ABC):
    """Abstract base class for data sources."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.name = config.get('name', self.__class__.__name__)
    
    @abstractmethod
    def fetch_contract_master(self, commodity: str, as_of_date: Optional[date] = None) -> List[Contract]:
        """Fetch contract master data for a commodity."""
        pass
    
    @abstractmethod
    def fetch_ohlcv(self, contract: str, start_date: date, end_date: date) -> List[OHLCV]:
        """Fetch OHLCV data for a contract."""
        pass
    
    @abstractmethod
    def fetch_open_interest(self, contract: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Fetch open interest data."""
        pass
    
    @abstractmethod
    def fetch_spot(self, commodity: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Fetch spot price data."""
        pass
    
    @abstractmethod
    def fetch_reference_rates(self, currency_pair: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Fetch reference exchange rates (e.g., INR/USD)."""
        pass
    
    @abstractmethod
    def fetch_macro_series(self, series_name: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Fetch macro-economic series."""
        pass
    
    @abstractmethod
    def fetch_weather(self, location: str, start_date: date, end_date: date) -> pd.DataFrame:
        """Fetch weather data."""
        pass
    
    @abstractmethod
    def fetch_calendar(self, exchange: str, year: int) -> List[date]:
        """Fetch trading calendar for exchange."""
        pass
    
    def _retry_request(self, func, *args, **kwargs):
        """Helper for retrying requests."""
        import time
        for attempt in range(self.config.get('retry_attempts', 3)):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == self.config.get('retry_attempts', 3) - 1:
                    raise e
                time.sleep(2 ** attempt)  # exponential backoff