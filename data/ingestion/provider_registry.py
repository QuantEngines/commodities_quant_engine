from __future__ import annotations

from typing import Dict, Optional

from ...config.settings import settings
from .base import DataSource
from .commodities_api import CommoditiesAPIDataSource
from .fbil import FBILDataSource
from .icici_breeze import ICICIBreezeDataSource
from .imd import IMDDataSource
from .mcx import MCXDataSource
from .mospi import MOSPIDataSource
from .ncdex import NCDEXDataSource
from .nse import NSEDataSource
from .ppac import PPACDataSource
from .zerodha_kiteconnect import ZerodhaKiteConnectDataSource


class ProviderRegistry:
    """Instantiate free/local-first providers with optional overrides."""

    def __init__(self):
        self.providers: Dict[str, DataSource] = {
            "ZERODHA": ZerodhaKiteConnectDataSource(self._provider_config("ZERODHA")),
            "ICICI_BREEZE": ICICIBreezeDataSource(self._provider_config("ICICI_BREEZE")),
            "MCX": MCXDataSource(self._provider_config("MCX")),
            "COMMODITIES_API": CommoditiesAPIDataSource(self._provider_config("COMMODITIES_API")),
            "NCDEX": NCDEXDataSource(self._provider_config("NCDEX")),
            "NSE": NSEDataSource(self._provider_config("NSE")),
            "FBIL": FBILDataSource(self._provider_config("FBIL")),
            "MOSPI": MOSPIDataSource(self._provider_config("MOSPI")),
            "IMD": IMDDataSource(self._provider_config("IMD")),
            "PPAC": PPACDataSource(self._provider_config("PPAC")),
        }

    def get(self, name: str) -> Optional[DataSource]:
        return self.providers.get(name.upper())

    def _provider_config(self, name: str) -> Dict[str, object]:
        source_config = settings.data_sources.get(name.upper())
        if source_config is None:
            return {"name": name}
        payload = source_config.model_dump()
        nested = payload.pop("config", {}) or {}
        return {**payload, **nested}


provider_registry = ProviderRegistry()
