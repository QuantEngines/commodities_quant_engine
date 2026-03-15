import requests
import pandas as pd
from typing import List, Dict, Optional, Any
from datetime import datetime, date, timedelta
import time
import logging
from functools import lru_cache
import json

from .base import MacroDataSource
from ...models import MacroSeries, MacroEvent, NewsItem

logger = logging.getLogger(__name__)

class ReutersMacroAdapter(MacroDataSource):
    """
    Reuters API adapter for macro data.

    This implementation uses Alpha Vantage and other free APIs as proxies
    for Reuters data, since actual Reuters API access requires expensive licenses.
    In production, replace with actual Reuters API calls.

    For actual Reuters integration, you would need:
    - Reuters API license and credentials
    - Thomson Reuters DataScope or Eikon API access
    - Reuters News API for news sentiment
    - Proper authentication and rate limiting
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.alpha_vantage_key = config.get('alpha_vantage_key', '')
        self.use_mock = config.get('use_mock', True)
        self.cache_ttl = config.get('cache_ttl_hours', 24)

        # Reuters/Alpha Vantage mappings
        self.alpha_vantage_mappings = {
            'US_CPI_YOY': 'CPI',  # Consumer Price Index
            'US_FED_RATE': 'FEDERAL_FUNDS_RATE',  # Federal Funds Rate
            'US_GDP_YOY': 'REAL_GDP',  # Real GDP
            'US_UNEMPLOYMENT': 'UNEMPLOYMENT',  # Unemployment Rate
            'IN_CPI_YOY': 'CPI',  # India CPI (limited availability)
            'CN_PMI': 'PMI',  # China PMI
            'EU_HICP_YOY': 'INFLATION',  # Euro Area Inflation
            'GB_GDP_YOY': 'REAL_GDP',  # UK GDP
            'JP_CPI_YOY': 'CPI',  # Japan CPI
        }

        # Rate limiting
        self.last_request_time = 0
        self.request_delay = 15.0  # Alpha Vantage free tier: 5 calls/minute

    @lru_cache(maxsize=50)
    def _get_alpha_vantage_data(self, function: str, symbol: str = None) -> Dict:
        """Fetch data from Alpha Vantage API with caching."""
        if not self.alpha_vantage_key:
            logger.warning("Alpha Vantage API key not provided, using mock data")
            return {}

        try:
            # Rate limiting
            elapsed = time.time() - self.last_request_time
            if elapsed < self.request_delay:
                time.sleep(self.request_delay - elapsed)
            self.last_request_time = time.time()

            url = "https://www.alphavantage.co/query"
            params = {
                'function': function,
                'apikey': self.alpha_vantage_key,
            }

            if symbol:
                params['symbol'] = symbol

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Check for API limit errors
            if 'Note' in data or 'Error Message' in data:
                logger.warning(f"Alpha Vantage API limit reached: {data}")
                return {}

            logger.info(f"Successfully fetched {function} data")
            return data

        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage data: {e}")
            return {}

    def fetch_macro_series(self, series_name: str, start_date: date, end_date: date) -> List[MacroSeries]:
        """Fetch macro series from Reuters (via Alpha Vantage proxy)."""
        if self.use_mock:
            return self._mock_macro_series(series_name, start_date, end_date)

        av_function = self.alpha_vantage_mappings.get(series_name)
        if not av_function:
            logger.warning(f"No Alpha Vantage mapping found for {series_name}, using mock data")
            return self._mock_macro_series(series_name, start_date, end_date)

        try:
            data = self._get_alpha_vantage_data(av_function)

            if not data:
                return self._mock_macro_series(series_name, start_date, end_date)

            # Parse Alpha Vantage response
            macro_series = []
            data_key = self._get_data_key(av_function)

            if data_key in data:
                for date_str, values in data[data_key].items():
                    dt = datetime.strptime(date_str, '%Y-%m-%d').date()

                    if start_date <= dt <= end_date:
                        value = self._extract_value(values, av_function)
                        if value is not None:
                            macro_series.append(MacroSeries(
                                series_name=series_name,
                                timestamp=datetime.combine(dt, datetime.min.time()),
                                value=float(value),
                                unit=self._get_series_unit(series_name),
                                frequency=self._get_series_frequency(series_name),
                                source='REUTERS_AV',
                                is_revised=False,
                                metadata={
                                    'alpha_vantage_function': av_function,
                                    'original_data': values
                                }
                            ))

            if macro_series:
                logger.info(f"Successfully fetched {len(macro_series)} data points for {series_name}")
                return macro_series
            else:
                logger.warning(f"No valid data found for {series_name}, using mock data")
                return self._mock_macro_series(series_name, start_date, end_date)

        except Exception as e:
            logger.error(f"Error fetching macro series {series_name}: {e}")
            return self._mock_macro_series(series_name, start_date, end_date)

    def _get_data_key(self, function: str) -> str:
        """Get the data key for Alpha Vantage response."""
        key_map = {
            'CPI': 'data',
            'FEDERAL_FUNDS_RATE': 'data',
            'REAL_GDP': 'data',
            'UNEMPLOYMENT': 'data',
            'INFLATION': 'data',
            'PMI': 'data'
        }
        return key_map.get(function, 'data')

    def _extract_value(self, values: Dict, function: str) -> Optional[float]:
        """Extract the relevant value from Alpha Vantage response."""
        try:
            if function == 'CPI':
                return float(values.get('value', 0))
            elif function == 'FEDERAL_FUNDS_RATE':
                return float(values.get('value', 0))
            elif function == 'REAL_GDP':
                return float(values.get('value', 0))
            elif function == 'UNEMPLOYMENT':
                return float(values.get('value', 0))
            elif function == 'INFLATION':
                return float(values.get('value', 0))
            elif function == 'PMI':
                return float(values.get('value', 0))
            else:
                return None
        except (ValueError, TypeError):
            return None

    def _get_series_unit(self, series_name: str) -> str:
        """Get the unit for a series."""
        if 'CPI' in series_name or 'INFLATION' in series_name:
            return 'index'
        elif 'RATE' in series_name:
            return 'percent'
        elif 'GDP' in series_name:
            return 'billions_usd'
        elif 'PMI' in series_name:
            return 'index'
        elif 'UNEMPLOYMENT' in series_name:
            return 'percent'
        else:
            return 'index'

    def _get_series_frequency(self, series_name: str) -> str:
        """Get the frequency for a series."""
        if 'GDP' in series_name:
            return 'quarterly'
        elif 'CPI' in series_name or 'PMI' in series_name or 'UNEMPLOYMENT' in series_name:
            return 'monthly'
        else:
            return 'daily'

    def _mock_macro_series(self, series_name: str, start_date: date, end_date: date) -> List[MacroSeries]:
        """Mock macro series data for testing."""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        mock_data = []

        for dt in dates:
            mock_data.append({
                'timestamp': dt,
                'value': 4.5 + 0.05 * (dt - dates[0]).days / 30,
                'is_revised': False,
                'metadata': {'source': 'REUTERS_MOCK'}
            })

        df = pd.DataFrame(mock_data)
        return self.standardize_macro_series(df, series_name, 'percent', 'daily', 'REUTERS')

    def fetch_macro_release_calendar(self, country: str, start_date: date, end_date: date) -> List[MacroEvent]:
        """Fetch macro event calendar from Reuters."""
        if self.use_mock:
            return self._mock_macro_events(country, start_date, end_date)

        # Reuters Economic Calendar API would be used here
        # This requires Thomson Reuters Eikon or DataScope subscription

        logger.warning("Reuters Economic Calendar API not implemented, using mock data")
        return self._mock_macro_events(country, start_date, end_date)

    def _mock_macro_events(self, country: str, start_date: date, end_date: date) -> List[MacroEvent]:
        """Mock macro events for testing."""
        mock_events = []

        # Generate monthly economic releases
        current_date = start_date
        while current_date <= end_date:
            if current_date.day == 20:  # End of month releases
                mock_events.append(MacroEvent(
                    event_id=f"{country}_GDP_{current_date.strftime('%Y%m')}",
                    event_type="gdp_release",
                    country=country,
                    timestamp=datetime.combine(current_date, datetime.min.time()),
                    title=f"{country} GDP Release",
                    description="Gross Domestic Product data release",
                    expected_impact="high",
                    source="REUTERS",
                    actual_impact=None,
                    commodities_affected=self._get_country_commodities(country)
                ))

            if current_date.day == 10:  # Mid-month releases
                mock_events.append(MacroEvent(
                    event_id=f"{country}_PMI_{current_date.strftime('%Y%m')}",
                    event_type="pmi_release",
                    country=country,
                    timestamp=datetime.combine(current_date, datetime.min.time()),
                    title=f"{country} PMI Release",
                    description="Purchasing Managers Index data release",
                    expected_impact="medium",
                    source="REUTERS",
                    actual_impact=None,
                    commodities_affected=self._get_country_commodities(country)
                ))

            current_date += timedelta(days=1)

        return mock_events

    def _get_country_commodities(self, country: str) -> List[str]:
        """Get commodities affected by a country's economic data."""
        country_commodities = {
            'US': ['GOLD', 'SILVER', 'COPPER', 'CRUDEOIL'],
            'IN': ['GOLD', 'SILVER', 'COPPER', 'CRUDEOIL', 'NATURALGAS'],
            'CN': ['COPPER', 'IRON', 'COAL', 'SOYBEAN'],
            'EU': ['GOLD', 'COPPER'],
            'GB': ['GOLD', 'COPPER'],
            'JP': ['GOLD', 'COPPER']
        }
        return country_commodities.get(country, ['GOLD'])

    def fetch_news_sentiment(self, keywords: List[str], start_date: date, end_date: date) -> List[NewsItem]:
        """Fetch news sentiment data from Reuters."""
        if self.use_mock:
            return self._mock_news_sentiment(keywords, start_date, end_date)

        # Reuters News API would be used here
        logger.warning("Reuters News API not implemented, using mock data")
        return self._mock_news_sentiment(keywords, start_date, end_date)

    def _mock_news_sentiment(self, keywords: List[str], start_date: date, end_date: date) -> List[NewsItem]:
        """Mock news sentiment for testing."""
        mock_news = []

        # Generate some sample news items
        dates = pd.date_range(start=start_date, end=end_date, freq='3D')

        for dt in dates[:4]:  # Limit to 4 items
            sentiment = np.random.choice([-1, 0, 1], p=[0.25, 0.5, 0.25])
            mock_news.append(NewsItem(
                news_id=f"REUTERS_NEWS_{dt.strftime('%Y%m%d')}",
                timestamp=dt.to_pydatetime(),
                headline=f"Economic Update: {'Bullish' if sentiment > 0 else 'Bearish' if sentiment < 0 else 'Mixed'} signals for {', '.join(keywords)}",
                summary=f"Reuters analysis of market conditions affecting {', '.join(keywords)} sector.",
                source="REUTERS",
                sentiment_score=float(sentiment),
                relevance_score=0.7,
                keywords=keywords,
                url=f"https://reuters.com/article/{dt.strftime('%Y%m%d')}"
            ))

        return mock_news
        return self.fetch_macro_series(series_name, start_date, end_date)

    def fetch_yields(self, country: str, tenor: str, start_date: date, end_date: date) -> List[MacroSeries]:
        """Fetch bond yields from Reuters."""
        series_name = f"{country}_YIELD_{tenor.upper()}"
        return self.fetch_macro_series(series_name, start_date, end_date)

    def fetch_fx_reference(self, currency_pair: str, start_date: date, end_date: date) -> List[MacroSeries]:
        """Fetch FX rates from Reuters."""
        series_name = f"FX_{currency_pair.replace('/', '_')}"
        return self.fetch_macro_series(series_name, start_date, end_date)

    def fetch_news_headlines(self, topics: List[str], start_date: date, end_date: date) -> List[NewsItem]:
        """Fetch news headlines from Reuters."""
        if self.mock_mode:
            return self._mock_news_headlines(topics, start_date, end_date)

        # TODO: Implement Reuters News API
        raise NotImplementedError("Reuters News API not implemented")

    def _mock_news_headlines(self, topics: List[str], start_date: date, end_date: date) -> List[NewsItem]:
        """Mock news headlines for testing."""
        mock_headlines = [
            NewsItem(
                news_id=f"REUTERS_NEWS_{start_date.strftime('%Y%m%d')}_001",
                timestamp=datetime.combine(start_date, datetime.min.time()),
                headline="Central Bank Signals Potential Rate Cuts Amid Economic Slowdown",
                source="REUTERS",
                relevance_score=0.9,
                sentiment_score=0.1,
                topics=["rates", "growth"],
                commodity_tags=["GOLD", "SILVER"],
                metadata={"mock": True}
            )
        ]
        return mock_headlines

    def fetch_news_articles(self, news_ids: List[str]) -> List[NewsItem]:
        """Fetch full news articles from Reuters."""
        # TODO: Implement article fetching
        return []

    def fetch_macro_events(self, country: str, event_types: List[str], start_date: date, end_date: date) -> List[MacroEvent]:
        """Fetch macro events from Reuters."""
        all_events = self.fetch_macro_release_calendar(country, start_date, end_date)
        return [e for e in all_events if e.event_type in event_types]

    # Placeholder implementations for inherited abstract methods
    def fetch_contract_master(self, commodity: str, as_of_date: Optional[date] = None):
        raise NotImplementedError("ReutersMacroAdapter does not support contract master")

    def fetch_ohlcv(self, contract: str, start_date: date, end_date: date):
        raise NotImplementedError("ReutersMacroAdapter does not support OHLCV")

    def fetch_open_interest(self, contract: str, start_date: date, end_date: date):
        raise NotImplementedError("ReutersMacroAdapter does not support open interest")

    def fetch_spot(self, commodity: str, start_date: date, end_date: date):
        raise NotImplementedError("ReutersMacroAdapter does not support spot prices")

    def fetch_reference_rates(self, currency_pair: str, start_date: date, end_date: date):
        raise NotImplementedError("ReutersMacroAdapter does not support reference rates")

    def fetch_weather(self, location: str, start_date: date, end_date: date):
        raise NotImplementedError("ReutersMacroAdapter does not support weather")

    def fetch_calendar(self, exchange: str, year: int):
        raise NotImplementedError("ReutersMacroAdapter does not support trading calendar")