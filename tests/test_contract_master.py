from datetime import date

import pandas as pd

from ..config.settings import settings
from ..data.contract_master.manager import ContractMaster


def test_contract_master_loads_curated_catalog(monkeypatch, tmp_path):
    path = tmp_path / "contracts.csv"
    pd.DataFrame(
        [
            {
                "commodity": "GOLD",
                "symbol": "GOLDAPR26",
                "expiry_date": "2026-04-28",
                "last_trading_date": "2026-04-26",
                "lot_size": 100,
                "tick_size": 1.0,
                "multiplier": 100,
                "exchange": "MCX",
                "segment": "bullion",
                "settlement_type": "cash",
                "source": "mcx_local_catalog",
            },
            {
                "commodity": "GOLD",
                "symbol": "GOLDMAY26",
                "expiry_date": "2026-05-28",
                "last_trading_date": "2026-05-26",
                "lot_size": 100,
                "tick_size": 1.0,
                "multiplier": 100,
                "exchange": "MCX",
                "segment": "bullion",
                "settlement_type": "cash",
                "source": "mcx_local_catalog",
            },
        ]
    ).to_csv(path, index=False)

    monkeypatch.setattr(settings.contract_master, "contract_catalog_path", str(path))
    manager = ContractMaster()
    active = manager.get_active_contract("GOLD", date(2026, 4, 1))

    assert active is not None
    assert active.symbol == "GOLDAPR26"
    assert active.source == "mcx_local_catalog"
    assert not active.is_fallback


def test_contract_master_fallback_marks_provenance(monkeypatch):
    monkeypatch.setattr(settings.contract_master, "contract_catalog_path", None)
    manager = ContractMaster()
    active = manager.get_active_contract("GOLD", date(2026, 4, 1))

    assert active is not None
    assert active.is_fallback
    assert active.source == "deterministic_fallback"
