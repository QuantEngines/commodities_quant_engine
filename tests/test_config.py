from ..config.settings import settings


def test_settings_loading():
    assert settings.regime_window_days == 252
    assert "GOLD" in settings.commodities
    assert "ALUMINIUM" in settings.commodities
    assert "NATURALGAS" in settings.commodities
    assert "CARDAMOM" in settings.commodities
    assert settings.commodities["GOLD"].exchange == "MCX"
    assert len(settings.commodities) >= 30
    assert "COMMODITIES_API" in settings.data_sources


def test_signal_and_adaptation_defaults():
    assert settings.signal.horizons == [1, 3, 5, 10, 20]
    assert settings.adaptation.min_sample_size >= 40
    assert settings.storage.signal_store == "signals"
    assert "CRUDEOILM" in settings.macro.commodity_sensitivities
    assert settings.evaluation_pricing.entry_price_field == "open"
    assert settings.contract_master.fallback_expiry_days >= 30
