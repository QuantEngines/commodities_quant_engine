from datetime import date

import pandas as pd

from ..data.ingestion.commodities_api import CommoditiesAPIDataSource
from ..data.ingestion.macro.providers.generic_news_adapter import GenericNewsAdapter
from ..data.ingestion.mcx import MCXDataSource


def test_mcx_local_first_ohlcv_loading(tmp_path):
    path = tmp_path / "gold.csv"
    pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=3, freq="B"),
            "open": [100, 101, 102],
            "high": [101, 102, 103],
            "low": [99, 100, 101],
            "close": [100.5, 101.5, 102.5],
            "volume": [1000, 1100, 1200],
        }
    ).to_csv(path, index=False)

    source = MCXDataSource({"local_catalog": {"GOLD": str(path)}})
    rows = source.fetch_ohlcv("GOLDFEB25", date(2025, 1, 1), date(2025, 1, 10))

    assert len(rows) == 3
    assert rows[0].contract == "GOLDFEB25"
    assert rows[-1].close == 102.5


def test_generic_news_adapter_supports_free_csv_only(tmp_path):
    path = tmp_path / "news.csv"
    pd.DataFrame(
        {
            "timestamp": ["2025-01-02 09:00:00", "2025-01-03 09:00:00"],
            "headline": ["Gold prices rise on softer real rates", "Unrelated sports headline"],
            "content": ["Bullion gains as rates ease", "Match report"],
            "source": ["LOCAL", "LOCAL"],
        }
    ).to_csv(path, index=False)

    adapter = GenericNewsAdapter({"csv_paths": {"news_csv": str(path)}, "rss_feeds": {}, "allow_paid_sources": False})
    items = adapter.fetch_news_headlines(["gold", "rates"], date(2025, 1, 1), date(2025, 1, 5))

    assert len(items) == 1
    assert items[0].source == "LOCAL"
    assert items[0].relevance_score > 0


def test_generic_news_adapter_tags_expanded_mcx_universe():
    adapter = GenericNewsAdapter({"rss_feeds": {}, "allow_paid_sources": False})
    tags = adapter._extract_commodity_tags("Natural gas and aluminium prices rise after supply disruption")

    assert "NATURALGAS" in tags
    assert "ALUMINIUM" in tags


def test_commodities_api_adapter_parses_timeseries_payload(monkeypatch):
    adapter = CommoditiesAPIDataSource(
        {
            "api_key": "demo-key",
            "symbol_map": {"GOLD": "XAU"},
            "base_url": "https://api.commodities-api.com/api",
        }
    )

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "success": True,
                "data": {
                    "rates": {
                        "2025-01-02": {"XAU": 0.0005},
                        "2025-01-03": {"XAU": 0.0004},
                    }
                },
            }

    monkeypatch.setattr(adapter.session, "get", lambda *args, **kwargs: FakeResponse())
    rows = adapter.fetch_ohlcv("GOLDFEB25", date(2025, 1, 1), date(2025, 1, 5))

    assert len(rows) == 2
    assert rows[0].close == 2000.0
    assert rows[1].close == 2500.0
