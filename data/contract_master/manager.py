from __future__ import annotations

from pathlib import Path
from datetime import date, timedelta
from typing import Dict, List, Optional

import pandas as pd

from ...config.settings import settings
from ...data.models import Commodity, Contract

PACKAGE_ROOT = Path(__file__).resolve().parents[2]


class ContractMaster:
    """Commodity and contract metadata manager with deterministic fallbacks."""

    def __init__(self):
        self.commodities: Dict[str, Commodity] = {}
        self.contracts: Dict[str, List[Contract]] = {}
        self.reload()

    def reload(self):
        self.commodities = {}
        self.contracts = {}
        self._load_commodities()
        self._load_curated_contracts()

    def _load_commodities(self):
        for symbol, config in settings.commodities.items():
            self.commodities[symbol] = Commodity(
                symbol=config.symbol,
                name=config.name or config.symbol,
                exchange=config.exchange,
                segment=config.segment,
                base_currency=config.base_currency,
                contract_multiplier=config.contract_multiplier,
                tick_size=config.tick_size,
                expiry_rule=config.expiry_rule,
                roll_days_before_expiry=config.roll_days_before_expiry,
                liquidity_threshold=config.liquidity_threshold,
                seasonality_class=config.seasonality_class,
                macro_sensitivity=config.macro_sensitivity,
            )

    def _load_curated_contracts(self):
        catalog_path = settings.contract_master.contract_catalog_path
        if not catalog_path:
            return
        file_path = Path(catalog_path)
        if not file_path.is_absolute():
            file_path = PACKAGE_ROOT / file_path
        if not file_path.exists():
            return

        frame = self._read_contract_catalog(file_path)
        if frame.empty:
            return
        frame = frame.copy()
        frame.columns = [str(column).strip().lower() for column in frame.columns]
        required_columns = {"commodity", "symbol", "expiry_date"}
        if not required_columns.issubset(frame.columns):
            return

        for row in frame.to_dict(orient="records"):
            commodity = str(row.get("commodity", "")).upper()
            if commodity not in self.commodities:
                continue
            contract = self._contract_from_row(row)
            if contract is not None:
                self.contracts.setdefault(commodity, []).append(contract)

        for commodity, contracts in self.contracts.items():
            deduped = {contract.symbol: contract for contract in contracts}
            self.contracts[commodity] = sorted(deduped.values(), key=lambda contract: contract.active_until)

    def _read_contract_catalog(self, file_path: Path) -> pd.DataFrame:
        if file_path.suffix == ".csv":
            return pd.read_csv(file_path)
        if file_path.suffix == ".parquet":
            return pd.read_parquet(file_path)
        if file_path.suffix == ".json":
            payload = pd.read_json(file_path)
            return payload if isinstance(payload, pd.DataFrame) else pd.DataFrame(payload)
        return pd.DataFrame()

    def _contract_from_row(self, row: Dict[str, object]) -> Optional[Contract]:
        try:
            expiry_date = pd.Timestamp(row["expiry_date"]).date()
        except Exception:
            return None
        first_notice = self._optional_date(row.get("first_notice_date"))
        last_trading = self._optional_date(row.get("last_trading_date"))
        source = str(row.get("source", "curated_local"))
        return Contract(
            commodity=str(row["commodity"]).upper(),
            symbol=str(row["symbol"]).upper(),
            expiry_date=expiry_date,
            lot_size=int(row.get("lot_size", row.get("multiplier", 1)) or 1),
            tick_size=float(row.get("tick_size", 0.01) or 0.01),
            multiplier=int(row.get("multiplier", row.get("lot_size", 1)) or 1),
            exchange=str(row.get("exchange", self.commodities[str(row["commodity"]).upper()].exchange)),
            segment=str(row.get("segment", self.commodities[str(row["commodity"]).upper()].segment)),
            first_notice_date=first_notice,
            last_trading_date=last_trading,
            quote_currency=str(row.get("quote_currency", "INR")),
            settlement_type=None if pd.isna(row.get("settlement_type")) or row.get("settlement_type") == "" else str(row.get("settlement_type")),
            source=source,
            is_fallback=False,
            metadata={
                "source_path": settings.contract_master.contract_catalog_path,
                "raw_row": {key: value for key, value in row.items() if key not in {"commodity", "symbol", "expiry_date", "lot_size", "tick_size", "multiplier", "exchange", "segment", "first_notice_date", "last_trading_date", "quote_currency", "settlement_type", "source"}},
            },
        )

    def _optional_date(self, value: object) -> Optional[date]:
        if value is None or value == "":
            return None
        try:
            return pd.Timestamp(value).date()
        except Exception:
            return None

    def get_commodity(self, symbol: str) -> Optional[Commodity]:
        return self.commodities.get(symbol)

    def get_active_contract(self, commodity: str, as_of_date: Optional[date] = None) -> Optional[Contract]:
        as_of_date = as_of_date or date.today()
        contracts = sorted(self.contracts.get(commodity, []), key=lambda contract: contract.active_until)
        for contract in contracts:
            roll_anchor = contract.last_trading_date if settings.contract_master.prefer_last_trading_date and contract.last_trading_date else contract.expiry_date
            roll_date = roll_anchor - timedelta(days=self.get_commodity(commodity).roll_days_before_expiry)
            if as_of_date <= roll_date:
                return contract
        return contracts[-1] if contracts else self._build_fallback_contract(commodity, as_of_date)

    def add_contracts(self, commodity: str, contracts: List[Contract]):
        self.contracts.setdefault(commodity, []).extend(contracts)

    def _build_fallback_contract(self, commodity: str, as_of_date: date) -> Optional[Contract]:
        commodity_meta = self.get_commodity(commodity)
        if commodity_meta is None:
            return None

        year = as_of_date.year % 100
        symbol = f"{commodity}{as_of_date.strftime('%b').upper()}{year:02d}"
        expiry_date = as_of_date + timedelta(days=settings.contract_master.fallback_expiry_days)

        return Contract(
            commodity=commodity,
            symbol=symbol,
            expiry_date=expiry_date,
            lot_size=commodity_meta.contract_multiplier,
            tick_size=commodity_meta.tick_size,
            multiplier=commodity_meta.contract_multiplier,
            exchange=commodity_meta.exchange,
            segment=commodity_meta.segment,
            quote_currency=commodity_meta.base_currency,
            source="deterministic_fallback",
            is_fallback=True,
            metadata={"reason": "No curated contract catalog entry was available."},
        )


contract_master = ContractMaster()
