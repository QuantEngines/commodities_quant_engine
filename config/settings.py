from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings

from .commodity_universe import default_macro_sensitivities, default_mcx_commodity_definitions


class CommodityConfig(BaseModel):
    symbol: str
    exchange: str
    segment: str
    contract_multiplier: int
    tick_size: float
    expiry_rule: str
    name: Optional[str] = None
    base_currency: str = "INR"
    roll_days_before_expiry: int = 5
    liquidity_threshold: int = 1000
    seasonality_class: str = "neutral"
    macro_sensitivity: List[str] = Field(default_factory=list)


class DataSourceConfig(BaseModel):
    name: str
    base_url: Optional[str] = None
    api_key: Optional[str] = None
    timeout: int = 30
    retry_attempts: int = 3
    config: Dict[str, Any] = Field(default_factory=dict)


class MacroDataSourceConfig(BaseModel):
    name: str
    adapter_class: str
    enabled: bool = False
    config: Dict[str, Any] = Field(default_factory=dict)


class MacroFeatureConfig(BaseModel):
    enabled: bool = True
    transforms: List[Dict[str, Any]] = Field(default_factory=list)


class MacroSettings(BaseModel):
    sources: Dict[str, MacroDataSourceConfig] = Field(default_factory=dict)
    features: Dict[str, MacroFeatureConfig] = Field(default_factory=dict)
    series_mappings: Dict[str, Dict[str, str]] = Field(default_factory=dict)
    commodity_sensitivities: Dict[str, List[str]] = Field(default_factory=default_macro_sensitivities)
    max_missing_pct: float = 0.5
    min_history_days: int = 252
    outlier_std_threshold: float = 5.0
    normalization_method: str = "z_score"
    normalization_window_days: int = 252
    cache_enabled: bool = True
    cache_dir: str = "data/cache/macro"
    max_cache_age_days: int = 1


class SignalSettings(BaseModel):
    horizons: List[int] = Field(default_factory=lambda: [1, 3, 5, 10, 20])
    directional_horizon_weights: Dict[str, float] = Field(
        default_factory=lambda: {
            "1": 0.10,
            "3": 0.20,
            "5": 0.30,
            "10": 0.25,
            "20": 0.15,
        }
    )
    directional_feature_names: List[str] = Field(
        default_factory=lambda: [
            "momentum_5d",
            "momentum_20d",
            "trend_strength_20d",
            "short_reversal_5d",
            "volatility_20d",
            "drawdown_20d",
            "volume_trend_20d",
            "carry_yield",
        ]
    )
    directional_feature_weights: Dict[str, Dict[str, float]] = Field(
        default_factory=lambda: {
            "1": {
                "momentum_5d": 0.30,
                "momentum_20d": 0.10,
                "trend_strength_20d": 0.10,
                "short_reversal_5d": -0.25,
                "volatility_20d": -0.15,
                "drawdown_20d": -0.05,
                "volume_trend_20d": 0.10,
                "carry_yield": 0.05,
            },
            "3": {
                "momentum_5d": 0.25,
                "momentum_20d": 0.15,
                "trend_strength_20d": 0.15,
                "short_reversal_5d": -0.15,
                "volatility_20d": -0.15,
                "drawdown_20d": -0.05,
                "volume_trend_20d": 0.10,
                "carry_yield": 0.05,
            },
            "5": {
                "momentum_5d": 0.20,
                "momentum_20d": 0.20,
                "trend_strength_20d": 0.20,
                "short_reversal_5d": -0.10,
                "volatility_20d": -0.15,
                "drawdown_20d": -0.05,
                "volume_trend_20d": 0.10,
                "carry_yield": 0.05,
            },
            "10": {
                "momentum_5d": 0.10,
                "momentum_20d": 0.25,
                "trend_strength_20d": 0.25,
                "short_reversal_5d": -0.05,
                "volatility_20d": -0.15,
                "drawdown_20d": -0.05,
                "volume_trend_20d": 0.10,
                "carry_yield": 0.05,
            },
            "20": {
                "momentum_5d": 0.05,
                "momentum_20d": 0.30,
                "trend_strength_20d": 0.30,
                "short_reversal_5d": -0.05,
                "volatility_20d": -0.15,
                "drawdown_20d": -0.05,
                "volume_trend_20d": 0.10,
                "carry_yield": 0.05,
            },
        }
    )
    directional_intercepts: Dict[str, float] = Field(
        default_factory=lambda: {str(horizon): 0.0 for horizon in [1, 3, 5, 10, 20]}
    )
    confidence_scale: float = 1.35
    feature_clip_abs: float = 5.0
    max_directional_score_abs: float = 2.5
    structural_prior_blend: float = 0.20
    regime_window_days: int = 252
    regime_model: str = "rule_based"
    hmm_states: int = 4
    hmm_min_history_rows: int = 120
    hmm_transition_blend: float = 0.40
    inefficiency_window_days: int = 20
    inefficiency_z_threshold: float = 2.0
    max_staleness_days: int = 3
    min_history_rows: int = 90


class CompositeSettings(BaseModel):
    directional_weight: float = 0.55
    inefficiency_weight: float = 0.20
    regime_weight: float = 0.15
    macro_weight: float = 0.10
    shipping_weight: float = 0.08
    risk_weight: float = 0.20
    neutral_threshold: float = 0.25
    weak_threshold: float = 0.60
    strong_threshold: float = 1.20


class NLPEventSettings(BaseModel):
    enabled: bool = False
    use_llm_extraction: bool = False
    max_items_per_cycle: int = 100
    directional_overlay_weight: float = 0.20
    regime_overlay_weight: float = 0.20
    confidence_uncertainty_weight: float = 0.25
    risk_penalty_weight: float = 0.30


class EvaluationSettings(BaseModel):
    horizons: List[int] = Field(default_factory=lambda: [1, 3, 5, 10, 20])
    confidence_buckets: int = 5
    degradation_window_signals: int = 20
    primary_horizon: int = 5
    event_window_days: int = 3
    entry_lag_bars: int = 1
    drift_thresholds_by_family: Dict[str, Dict[str, float]] = Field(
        default_factory=lambda: {
            "default": {
                "hit_rate_drop": 0.10,
                "regime_alignment_drop": 0.10,
                "brier_increase": 0.03,
            },
            "bullion": {
                "hit_rate_drop": 0.08,
                "regime_alignment_drop": 0.08,
                "brier_increase": 0.025,
            },
            "base_metals": {
                "hit_rate_drop": 0.10,
                "regime_alignment_drop": 0.10,
                "brier_increase": 0.03,
            },
            "energy": {
                "hit_rate_drop": 0.12,
                "regime_alignment_drop": 0.12,
                "brier_increase": 0.04,
            },
            "agri": {
                "hit_rate_drop": 0.15,
                "regime_alignment_drop": 0.15,
                "brier_increase": 0.05,
            },
        }
    )


class AdaptationSettings(BaseModel):
    enabled: bool = True
    rolling_window_signals: int = 250
    holdout_fraction: float = 0.25
    walk_forward_folds: int = 4
    purge_overlap_bars: int = 0
    embargo_bars: int = 0
    min_sample_size: int = 40
    min_hit_rate_improvement: float = 0.03
    min_rank_ic_improvement: float = 0.02
    min_improved_horizons: int = 1
    max_feature_drift: float = 1.5
    ridge_alpha: float = 1.0
    recency_halflife_signals: int = 60
    target_winsor_quantile: float = 0.02
    auto_promote: bool = False
    manual_approval_required: bool = True
    default_mode: str = "recommend"


class ContractMasterSettings(BaseModel):
    contract_catalog_path: Optional[str] = None
    prefer_last_trading_date: bool = True
    fallback_expiry_days: int = 30


class EvaluationPricingSettings(BaseModel):
    entry_price_field: str = "open"
    exit_price_field: str = "close"
    entry_slippage_bps: float = 3.0
    exit_slippage_bps: float = 3.0
    entry_spread_bps: float = 4.0
    exit_spread_bps: float = 4.0
    impact_coefficient_bps: float = 1.5
    turnover_cost_bps: float = 2.0
    max_slippage_from_range_fraction: float = 0.15
    low_volume_threshold_ratio: float = 0.35
    low_volume_slippage_multiplier: float = 1.5
    target_annualized_vol: float = 0.15
    max_abs_position: float = 1.0
    min_trade_confidence: float = 0.10
    vol_target_window_bars: int = 20
    annualization_days: int = 252


class StorageSettings(BaseModel):
    base_dir: str = "artifacts"
    market_data_store: str = "market_data"
    signal_store: str = "signals"
    evaluation_store: str = "evaluations"
    parameter_store: str = "parameters"
    report_store: str = "reports"


class ShippingSettings(BaseModel):
    enabled: bool = True
    aggregation_frequency: str = "1D"
    max_track_gap_hours: int = 24
    default_timezone: str = "UTC"
    sources_config_path: str = "config/shipping_sources.yaml"
    geographies_config_path: str = "config/shipping_geographies.yaml"
    features_config_path: str = "config/shipping_features.yaml"
    signal_rules_config_path: str = "config/shipping_signal_rules.yaml"
    allow_sparse_overlay: bool = True


class Settings(BaseSettings):
    data_sources: Dict[str, DataSourceConfig] = Field(default_factory=dict)
    commodities: Dict[str, CommodityConfig] = Field(default_factory=dict)
    macro: MacroSettings = Field(default_factory=MacroSettings)
    signal: SignalSettings = Field(default_factory=SignalSettings)
    composite: CompositeSettings = Field(default_factory=CompositeSettings)
    shipping: ShippingSettings = Field(default_factory=ShippingSettings)
    nlp_event: NLPEventSettings = Field(default_factory=NLPEventSettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    adaptation: AdaptationSettings = Field(default_factory=AdaptationSettings)
    contract_master: ContractMasterSettings = Field(default_factory=ContractMasterSettings)
    evaluation_pricing: EvaluationPricingSettings = Field(default_factory=EvaluationPricingSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    data_dir: str = "data"
    cache_dir: str = "cache"
    config_version: str = "2026-03-research-upgrade"

    model_config = ConfigDict(env_file=".env", case_sensitive=False, extra="ignore")

    @property
    def regime_window_days(self) -> int:
        return self.signal.regime_window_days

    @property
    def directional_horizons(self) -> List[int]:
        return self.signal.horizons

    @property
    def inefficiency_z_threshold(self) -> float:
        return self.signal.inefficiency_z_threshold

    @property
    def risk_penalty_vol_spike(self) -> float:
        return self.composite.risk_weight


settings = Settings()


if not settings.data_sources:
    settings.data_sources = {
        "ZERODHA": DataSourceConfig(
            name="ZERODHA",
            base_url=None,
            config={
                "api_key": "${ZERODHA_API_KEY}",  # Set via .env or environment
                "api_secret": "${ZERODHA_API_SECRET}",  # Set via .env or environment
                "access_token": "${ZERODHA_ACCESS_TOKEN}",  # Set via .env or environment
                "fallback_enabled": True,
            },
        ),
        "ICICI_BREEZE": DataSourceConfig(
            name="ICICI_BREEZE",
            base_url=None,
            config={
                "api_key": "${ICICI_BREEZE_API_KEY}",
                "secret_key": "${ICICI_BREEZE_SECRET_KEY}",
                "session_token": "${ICICI_BREEZE_SESSION_TOKEN}",
                "exchange_code": "MCX",
                "product_type": "futures",
                "fallback_enabled": True,
            },
        ),
        "MCX": DataSourceConfig(name="MCX", base_url=None),
        "COMMODITIES_API": DataSourceConfig(
            name="COMMODITIES_API",
            base_url="https://api.commodities-api.com/api",
            config={
                "symbol_map": {},
                "base_currency": "USD",
                "quote_currency": "USD",
                "invert_quotes": True,
                "use_for_reference_data_only": True,
            },
        ),
        "NCDEX": DataSourceConfig(name="NCDEX", base_url=None),
        "NSE": DataSourceConfig(name="NSE", base_url=None),
        "FBIL": DataSourceConfig(name="FBIL", base_url=None),
        "MOSPI": DataSourceConfig(name="MOSPI", base_url=None),
        "IMD": DataSourceConfig(name="IMD", base_url=None),
        "PPAC": DataSourceConfig(name="PPAC", base_url=None),
    }


if not settings.commodities:
    settings.commodities = {
        symbol: CommodityConfig(
            symbol=symbol,
            exchange="MCX",
            base_currency="INR",
            **definition,
        )
        for symbol, definition in default_mcx_commodity_definitions().items()
    }


if not settings.macro.sources:
    settings.macro.sources = {
        "official": MacroDataSourceConfig(
            name="Official Macro Data",
            adapter_class="data.ingestion.macro.providers.official_macro_adapter.OfficialMacroAdapter",
            enabled=True,
            config={
                "series_catalog": {},
                "event_calendar_catalog": {},
                "allow_fred": False,
            },
        ),
        "generic_news": MacroDataSourceConfig(
            name="Generic Free News",
            adapter_class="data.ingestion.macro.providers.generic_news_adapter.GenericNewsAdapter",
            enabled=True,
            config={
                "allow_paid_sources": False,
                "rss_feeds": {
                    "pib_economy": "https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3",
                    "economic_times": "https://economictimes.indiatimes.com/rssfeedsdefault.cms",
                },
            },
        ),
        "bloomberg": MacroDataSourceConfig(
            name="Bloomberg Terminal",
            adapter_class="data.ingestion.macro.providers.bloomberg_adapter.BloombergMacroAdapter",
            enabled=False,
            config={"mock_mode": True},
        ),
        "reuters": MacroDataSourceConfig(
            name="Reuters",
            adapter_class="data.ingestion.macro.providers.reuters_adapter.ReutersMacroAdapter",
            enabled=False,
            config={"mock_mode": True},
        ),
    }
