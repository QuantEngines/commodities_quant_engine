from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterable, List, Mapping, Optional

import pandas as pd

from ..config.settings import settings
from ..data.models import MacroFeature
from .features import (
    AnchorageFeatures,
    ChokepointStressFeatures,
    DwellTimeFeatures,
    PortCongestionFeatures,
    RouteDisruptionFeatures,
    ShippingMomentumFeatures,
    SpeedAnomalyFeatures,
    TankerFlowFeatures,
)
from .geospatial.zones import ZoneCatalog
from .models import ShippingFeatureVector, ShippingObservationWindow
from .processing import aggregate_shipping_metrics, compute_daily_quality, detect_vessel_events, normalize_port_calls, normalize_route_events, normalize_vessel_positions, reconstruct_tracks


class ShippingFeaturePipeline:
    BENCHMARK_GROUP_SPECS: Dict[str, Dict[str, object]] = {
        "dry_bulk": {
            "commodities": None,
            "segments": {"base_metals", "agri"},
            "series": {
                "bdi": 0.30,
                "bci": 0.20,
                "bpi": 0.15,
                "bsi": 0.15,
                "bulker_value": 0.20,
            },
            "label": "dry-bulk benchmarks",
        },
        "tanker": {
            "commodities": {"CRUDEOIL", "CRUDEOILM", "BRCRUDEOIL"},
            "segments": set(),
            "series": {
                "bdti": 0.45,
                "bcti": 0.25,
                "tanker_value": 0.30,
            },
            "label": "tanker benchmarks",
        },
        "lng": {
            "commodities": {"NATURALGAS", "NATGASMINI"},
            "segments": set(),
            "series": {
                "lng_rate": 0.60,
                "lng_carrier_value": 0.40,
            },
            "label": "LNG carrier benchmarks",
        },
    }

    def __init__(self, zone_catalog: Optional[ZoneCatalog] = None):
        self.zone_catalog = zone_catalog or ZoneCatalog.from_config()
        self.congestion = PortCongestionFeatures()
        self.anchorage = AnchorageFeatures()
        self.route_disruption = RouteDisruptionFeatures()
        self.chokepoint = ChokepointStressFeatures()
        self.tanker_flow = TankerFlowFeatures()
        self.dwell_time = DwellTimeFeatures()
        self.speed_anomaly = SpeedAnomalyFeatures()
        self.shipping_momentum = ShippingMomentumFeatures()

    def run(
        self,
        commodity: str,
        vessel_positions: Iterable[Mapping[str, object]] | pd.DataFrame,
        port_calls: Iterable[Mapping[str, object]] | pd.DataFrame | None = None,
        route_events: Iterable[Mapping[str, object]] | pd.DataFrame | None = None,
        macro_features: Optional[List[MacroFeature]] = None,
        as_of_timestamp: Optional[datetime] = None,
    ) -> List[ShippingFeatureVector]:
        positions = normalize_vessel_positions(vessel_positions, source="shipping_pipeline", commodity_tags=[commodity])
        if positions.empty:
            return []
        port_call_frame = normalize_port_calls(port_calls if port_calls is not None else pd.DataFrame(), source="shipping_pipeline")
        route_event_frame = normalize_route_events(route_events if route_events is not None else pd.DataFrame(), source="shipping_pipeline")
        tracked = reconstruct_tracks(positions)
        events = detect_vessel_events(tracked, self.zone_catalog)
        metrics = aggregate_shipping_metrics(tracked, events, self.zone_catalog, port_calls=port_call_frame)
        quality = compute_daily_quality(positions, events)

        feature_frame = pd.concat(
            [
                self.congestion.compute(metrics),
                self.anchorage.compute(metrics),
                self.route_disruption.compute(metrics, route_events=route_event_frame),
                self.chokepoint.compute(metrics, chokepoint_events=route_event_frame.loc[route_event_frame["route_id"].astype(str).str.contains("HORMUZ|BAB_|MALACCA", na=False)] if not route_event_frame.empty else pd.DataFrame()),
                self.tanker_flow.compute(metrics),
                self.dwell_time.compute(metrics),
                self.speed_anomaly.compute(metrics),
            ],
            axis=1,
        ).sort_index()
        feature_frame = feature_frame.loc[:, ~feature_frame.columns.duplicated()]
        momentum = self.shipping_momentum.compute(feature_frame)
        feature_frame = feature_frame.join(momentum, how="outer").join(quality.set_index("window_start"), how="outer")
        if feature_frame.empty and not quality.empty:
            feature_frame = quality.set_index("window_start").copy()
        benchmark_frame = self._shipping_market_benchmark_frame(
            commodity=commodity,
            macro_features=macro_features or [],
            shipping_frame=feature_frame,
        )
        if not benchmark_frame.empty:
            feature_frame = feature_frame.join(benchmark_frame, how="left")
        feature_frame = feature_frame.apply(pd.to_numeric, errors="coerce").fillna(0.0).sort_index()

        observation_start = pd.to_datetime(positions["timestamp"]).min().to_pydatetime()
        as_of = as_of_timestamp or pd.to_datetime(positions["timestamp"]).max().to_pydatetime()
        vectors: List[ShippingFeatureVector] = []
        for timestamp, row in feature_frame.iterrows():
            observation_window = ShippingObservationWindow(
                start_time=observation_start,
                end_time=pd.Timestamp(timestamp).to_pydatetime(),
                frequency=settings.shipping.aggregation_frequency,
                source="shipping_pipeline",
                commodity=commodity,
                geography_scope=["config_geographies"],
                source_ids=["shipping_pipeline"],
            )
            quality_score = float(row.get("shipping_data_quality_score", 0.0))
            vectors.append(
                ShippingFeatureVector(
                    timestamp=pd.Timestamp(timestamp).to_pydatetime(),
                    commodity=commodity,
                    features={column: float(row.get(column, 0.0)) for column in feature_frame.columns},
                    source="shipping_pipeline",
                    observation_window=observation_window,
                    quality_score=quality_score,
                    confidence_score=max(0.0, min(1.0, quality_score)),
                    geography_tags=["shipping_overlay"],
                    commodity_tags=[commodity],
                    key_drivers=self._key_drivers_from_row(row),
                    notes=["Derived from public/local shipping proxies."],
                )
            )
        return [vector for vector in vectors if vector.timestamp <= as_of]

    def _shipping_market_benchmark_frame(
        self,
        commodity: str,
        macro_features: List[MacroFeature],
        shipping_frame: pd.DataFrame,
    ) -> pd.DataFrame:
        benchmark_spec = self._benchmark_spec_for_commodity(commodity)
        if benchmark_spec is None or not macro_features or shipping_frame.empty:
            return pd.DataFrame(index=shipping_frame.index)

        feature_names = {
            "bdi_level": "bdi_benchmark_level",
            "bdi_zscore": "bdi_benchmark_zscore",
            "bdi_momentum_20d": "bdi_benchmark_momentum",
            "bdi_shock_flag": "bdi_benchmark_shock_flag",
            "bci_level": "bci_benchmark_level",
            "bci_zscore": "bci_benchmark_zscore",
            "bci_momentum_20d": "bci_benchmark_momentum",
            "bci_shock_flag": "bci_benchmark_shock_flag",
            "bpi_level": "bpi_benchmark_level",
            "bpi_zscore": "bpi_benchmark_zscore",
            "bpi_momentum_20d": "bpi_benchmark_momentum",
            "bpi_shock_flag": "bpi_benchmark_shock_flag",
            "bsi_level": "bsi_benchmark_level",
            "bsi_zscore": "bsi_benchmark_zscore",
            "bsi_momentum_20d": "bsi_benchmark_momentum",
            "bsi_shock_flag": "bsi_benchmark_shock_flag",
            "bdti_level": "bdti_benchmark_level",
            "bdti_zscore": "bdti_benchmark_zscore",
            "bdti_momentum_10d": "bdti_benchmark_momentum",
            "bdti_shock_flag": "bdti_benchmark_shock_flag",
            "bcti_level": "bcti_benchmark_level",
            "bcti_zscore": "bcti_benchmark_zscore",
            "bcti_momentum_10d": "bcti_benchmark_momentum",
            "bcti_shock_flag": "bcti_benchmark_shock_flag",
            "lng_rate_level": "lng_rate_benchmark_level",
            "lng_rate_zscore": "lng_rate_benchmark_zscore",
            "lng_rate_momentum_10d": "lng_rate_benchmark_momentum",
            "lng_rate_shock_flag": "lng_rate_benchmark_shock_flag",
            "bulker_value_level": "bulker_value_benchmark_level",
            "bulker_value_zscore": "bulker_value_benchmark_zscore",
            "bulker_value_momentum_20d": "bulker_value_benchmark_momentum",
            "bulker_value_shock_flag": "bulker_value_benchmark_shock_flag",
            "tanker_value_level": "tanker_value_benchmark_level",
            "tanker_value_zscore": "tanker_value_benchmark_zscore",
            "tanker_value_momentum_20d": "tanker_value_benchmark_momentum",
            "tanker_value_shock_flag": "tanker_value_benchmark_shock_flag",
            "lng_carrier_value_level": "lng_carrier_value_benchmark_level",
            "lng_carrier_value_zscore": "lng_carrier_value_benchmark_zscore",
            "lng_carrier_value_momentum_20d": "lng_carrier_value_benchmark_momentum",
            "lng_carrier_value_shock_flag": "lng_carrier_value_benchmark_shock_flag",
        }
        records: Dict[pd.Timestamp, Dict[str, float]] = {}
        for feature in macro_features:
            mapped_name = feature_names.get(feature.feature_name)
            if mapped_name is None:
                continue
            timestamp = pd.Timestamp(feature.timestamp).floor(settings.shipping.aggregation_frequency)
            row = records.setdefault(timestamp, {})
            row[mapped_name] = float(feature.value)

        if not records:
            return pd.DataFrame(index=shipping_frame.index)

        benchmark_frame = pd.DataFrame.from_dict(records, orient="index").sort_index()
        benchmark_frame = benchmark_frame.reindex(pd.DatetimeIndex(shipping_frame.index), method="ffill")
        benchmark_frame = benchmark_frame.apply(pd.to_numeric, errors="coerce")
        benchmark_frame = self._attach_specific_benchmark_supports(benchmark_frame)
        shipping_stress = shipping_frame.apply(self._shipping_stress_score, axis=1)
        benchmark_frame = self._attach_group_benchmark_aggregates(
            benchmark_frame=benchmark_frame,
            shipping_stress=shipping_stress,
            benchmark_spec=benchmark_spec,
        )
        return benchmark_frame

    def _benchmark_spec_for_commodity(self, commodity: str) -> Optional[Dict[str, object]]:
        commodity_config = settings.commodities.get(commodity)
        if commodity_config is None:
            return None
        commodity_segment = str(commodity_config.segment).lower()
        for spec in self.BENCHMARK_GROUP_SPECS.values():
            commodity_set = spec.get("commodities") or set()
            if commodity in commodity_set:
                return spec
            segment_set = spec.get("segments") or set()
            if commodity_segment in segment_set:
                return spec
        return None

    def _attach_specific_benchmark_supports(self, benchmark_frame: pd.DataFrame) -> pd.DataFrame:
        prefix_to_momentum_suffix = {
            "bdi": "20d",
            "bci": "20d",
            "bpi": "20d",
            "bsi": "20d",
            "bdti": "10d",
            "bcti": "10d",
            "lng_rate": "10d",
            "bulker_value": "20d",
            "tanker_value": "20d",
            "lng_carrier_value": "20d",
        }
        for prefix, _suffix in prefix_to_momentum_suffix.items():
            z_col = f"{prefix}_benchmark_zscore"
            m_col = f"{prefix}_benchmark_momentum"
            if z_col not in benchmark_frame.columns and m_col not in benchmark_frame.columns:
                continue
            z_values = benchmark_frame[z_col] if z_col in benchmark_frame.columns else 0.0
            m_values = benchmark_frame[m_col] if m_col in benchmark_frame.columns else 0.0
            benchmark_frame[f"{prefix}_benchmark_support"] = (
                pd.Series(z_values, index=benchmark_frame.index).fillna(0.0).clip(lower=0.0) * 0.35
                + pd.Series(m_values, index=benchmark_frame.index).fillna(0.0).clip(lower=0.0) * 2.0
            ).clip(upper=1.0)
            benchmark_frame[f"{prefix}_benchmark_active"] = (
                benchmark_frame[[column for column in benchmark_frame.columns if column.startswith(f"{prefix}_benchmark_")]].notna().any(axis=1)
            ).astype(float)
        return benchmark_frame

    def _attach_group_benchmark_aggregates(
        self,
        benchmark_frame: pd.DataFrame,
        shipping_stress: pd.Series,
        benchmark_spec: Dict[str, object],
    ) -> pd.DataFrame:
        series_weights = dict(benchmark_spec.get("series", {}) or {})
        shipping_stress = shipping_stress.reindex(benchmark_frame.index).fillna(0.0)
        weighted_zscore = pd.Series(0.0, index=benchmark_frame.index)
        weighted_momentum = pd.Series(0.0, index=benchmark_frame.index)
        weighted_support = pd.Series(0.0, index=benchmark_frame.index)
        total_weight = pd.Series(0.0, index=benchmark_frame.index)
        active_count = pd.Series(0.0, index=benchmark_frame.index)

        for prefix, weight in series_weights.items():
            active_col = f"{prefix}_benchmark_active"
            z_col = f"{prefix}_benchmark_zscore"
            m_col = f"{prefix}_benchmark_momentum"
            s_col = f"{prefix}_benchmark_support"
            if active_col not in benchmark_frame.columns:
                continue
            active_mask = benchmark_frame[active_col].fillna(0.0) > 0.5
            if not active_mask.any():
                continue
            total_weight.loc[active_mask] += float(weight)
            active_count.loc[active_mask] += 1.0
            if z_col in benchmark_frame.columns:
                weighted_zscore.loc[active_mask] += benchmark_frame.loc[active_mask, z_col].fillna(0.0) * float(weight)
            if m_col in benchmark_frame.columns:
                weighted_momentum.loc[active_mask] += benchmark_frame.loc[active_mask, m_col].fillna(0.0) * float(weight)
            if s_col in benchmark_frame.columns:
                weighted_support.loc[active_mask] += benchmark_frame.loc[active_mask, s_col].fillna(0.0) * float(weight)
            if z_col in benchmark_frame.columns:
                benchmark_frame[f"{prefix}_shipping_stress_score"] = shipping_stress
                benchmark_frame[f"{prefix}_shipping_divergence"] = benchmark_frame[z_col].fillna(0.0) - shipping_stress

        denominator = total_weight.replace(0.0, pd.NA)
        benchmark_frame["shipping_market_benchmark_zscore"] = weighted_zscore.div(denominator).fillna(0.0)
        benchmark_frame["shipping_market_benchmark_momentum"] = weighted_momentum.div(denominator).fillna(0.0)
        benchmark_frame["shipping_market_benchmark_support"] = weighted_support.div(denominator).fillna(0.0)
        benchmark_frame["shipping_market_stress_score"] = shipping_stress
        benchmark_frame["shipping_market_divergence"] = benchmark_frame["shipping_market_benchmark_zscore"] - benchmark_frame["shipping_market_stress_score"]
        benchmark_frame["shipping_market_benchmark_active"] = (total_weight > 0.0).astype(float)
        benchmark_frame["shipping_market_benchmark_count"] = active_count
        return benchmark_frame

    def _shipping_stress_score(self, row: pd.Series) -> float:
        disruption = float(row.get("port_congestion_score", 0.0))
        disruption += float(row.get("anchorage_buildup_score", 0.0))
        disruption += float(row.get("route_disruption_score", 0.0))
        disruption += float(row.get("chokepoint_stress_score", 0.0))
        return max(0.0, min(1.0, disruption / 4.0))

    def _key_drivers_from_row(self, row: pd.Series) -> List[str]:
        drivers: List[str] = []
        for feature_name, label in (
            ("port_congestion_score", "Port congestion"),
            ("anchorage_buildup_score", "Anchorage buildup"),
            ("route_disruption_score", "Route disruption"),
            ("chokepoint_stress_score", "Chokepoint stress"),
        ):
            if float(row.get(feature_name, 0.0)) > 0.25:
                drivers.append(f"{label} is elevated")
        if float(row.get("shipping_market_benchmark_active", 0.0)) > 0.5 and float(row.get("shipping_market_benchmark_support", 0.0)) > 0.10:
            if float(row.get("bdti_benchmark_active", 0.0)) > 0.5 or float(row.get("bcti_benchmark_active", 0.0)) > 0.5:
                drivers.append("Tanker benchmarks are confirming freight tightness")
            elif float(row.get("lng_rate_benchmark_active", 0.0)) > 0.5 or float(row.get("lng_carrier_value_benchmark_active", 0.0)) > 0.5:
                drivers.append("LNG carrier benchmarks are confirming shipping tightness")
            else:
                drivers.append("Dry-bulk benchmarks are confirming freight tightness")
        return drivers[:4]
