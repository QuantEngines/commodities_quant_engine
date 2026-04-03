"""
ShippingContextBuilder — Automatic shipping intelligence for signal generation.

Generates shipping context either from:
1. Real-time AIS/port data (when available)
2. Stub data enriched with commodity-specific heuristics (graceful fallback)
3. Empty context (silent fallback if disabled)

This layer ensures shipping is always available to the composite signal engine,
but gracefully degrades if data sources are unavailable.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd

from ..config.settings import settings
from ..data.models import MacroFeature
from ..shipping.models import ShippingFeatureVector
from ..shipping.pipeline import ShippingFeaturePipeline

logger = logging.getLogger(__name__)


class ShippingContextBuilder:
    """Builds shipping context for commodities, with graceful fallback to stubs."""

    # Commodity-to-vessel-class mapping for realistic stubs
    COMMODITY_VESSEL_CLASSES = {
        "CRUDEOIL": ["crude_tanker"],
        "BRCRUDEOIL": ["crude_tanker"],
        "NATURALGAS": ["lng_carrier"],
        "COPPER": ["general_cargo", "container"],
        "ALUMINIUM": ["general_cargo", "bulk_carrier"],
        "ZINC": ["bulk_carrier", "general_cargo"],
        "LEAD": ["bulk_carrier", "general_cargo"],
        "NICKEL": ["bulk_carrier", "general_cargo"],
        "TIN": ["bulk_carrier"],
        "COTTON": ["general_cargo", "container"],
        "WHEAT": ["bulk_carrier"],
        "SOYABEAN": ["bulk_carrier"],
        "MAIZE": ["bulk_carrier"],
        "GOLD": ["high_value_container"],
        "SILVER": ["high_value_container"],
    }

    # Key shipping routes for major commodities
    COMMODITY_ROUTES = {
        "CRUDEOIL": ["GULF_TO_SINGAPORE", "GULF_TO_ROTTERDAM", "WC_AFRICA_TO_AUS"],
        "BRCRUDEOIL": ["GULF_TO_SINGAPORE", "GULF_TO_US_GULF"],
        "NATURALGAS": ["QATAR_TO_APAC", "US_LNG_TO_APAC"],
        "COPPER": ["CHILE_TO_SHANGHAI", "PERU_TO_SHANGHAI"],
        "COTTON": ["HOUSTON_TO_SHANGHAI", "WEST_AFRICA_TO_INDIA"],
        "WHEAT": ["UKRAINE_TO_MIDEAST", "CANADA_TO_APAC", "US_GULF_TO_APAC"],
    }

    def __init__(self, zone_catalog_optional: Optional[object] = None):
        self.pipeline = ShippingFeaturePipeline()
        self.enabled = settings.shipping.enabled

    def build(
        self,
        commodity: str,
        as_of_timestamp: datetime,
        macro_features: Optional[List[MacroFeature]] = None,
        real_vessel_positions: Optional[pd.DataFrame] = None,
        real_port_calls: Optional[pd.DataFrame] = None,
        real_route_events: Optional[pd.DataFrame] = None,
    ) -> List[ShippingFeatureVector]:
        """
        Build shipping feature vectors for a commodity.
        
        Attempts real data first, falls back to stubs if needed.
        Returns empty list if shipping disabled or unavailable.
        """
        if not self.enabled:
            return []

        # Try real data
        if real_vessel_positions is not None and not real_vessel_positions.empty:
            try:
                features = self.pipeline.run(
                    commodity=commodity,
                    vessel_positions=real_vessel_positions,
                    port_calls=real_port_calls,
                    route_events=real_route_events,
                    macro_features=macro_features,
                    as_of_timestamp=as_of_timestamp,
                )
                if features:
                    logger.info(
                        f"Shipping context: {len(features)} feature vectors from real data for {commodity}"
                    )
                    return features
            except Exception as e:
                logger.warning(f"Failed to process real shipping data: {e}; falling back to stubs")

        # Try stubs
        try:
            stub_positions = self._generate_stub_vessel_positions(commodity, as_of_timestamp)
            stub_routes = self._generate_stub_route_events(commodity, as_of_timestamp)
            
            if stub_positions.empty and stub_routes.empty:
                logger.debug(f"No shipping stubs available for {commodity}")
                return []
            
            features = self.pipeline.run(
                commodity=commodity,
                vessel_positions=stub_positions,
                port_calls=pd.DataFrame(),
                route_events=stub_routes,
                macro_features=macro_features,
                as_of_timestamp=as_of_timestamp,
            )
            
            if features:
                logger.info(
                    f"Shipping context: {len(features)} feature vectors from stubs for {commodity}"
                )
            return features
        except Exception as e:
            logger.debug(f"Stub shipping generation failed: {e}")
            return []

    def _generate_stub_vessel_positions(
        self, commodity: str, as_of_timestamp: datetime
    ) -> pd.DataFrame:
        """
        Generate realistic stub AIS-like vessel positions.
        
        Creates a small fleet of vessels in key routes with realistic speeds/positions.
        """
        vessel_classes = self.COMMODITY_VESSEL_CLASSES.get(commodity, ["general_cargo"])
        routes = self.COMMODITY_ROUTES.get(commodity, [])

        if not routes:
            return pd.DataFrame()

        rows = []
        base_date = as_of_timestamp.date()
        
        # Single vessel per route for stub (more would be unrealistic CPU cost)
        for route_idx, route_id in enumerate(routes[:2]):  # Max 2 routes
            for vessel_idx in range(2):  # 2 vessels per route
                vessel_id = f"stub_{commodity}_v{route_idx}_{vessel_idx}"
                
                # Heuristic positions based on route
                if "GULF" in route_id:
                    lat_range, lon_range = (24, 28), (50, 56)
                elif "SINGAPORE" in route_id or "APAC" in route_id:
                    lat_range, lon_range = (1, 5), (103, 107)
                elif "SHANGHAI" in route_id:
                    lat_range, lon_range = (30, 32), (120, 122)
                elif "US" in route_id or "HOUSTON" in route_id:
                    lat_range, lon_range = (28, 30), (-94, -90)
                elif "ROTTERDAM" in route_id:
                    lat_range, lon_range = (51, 52), (4, 5)
                else:
                    lat_range, lon_range = (20, 35), (-20, 35)

                # Spread vessels across route over 18 days with varying speeds
                day_offset = (vessel_idx % 10) + (route_idx % 5) * 2
                timestamp = datetime.combine(base_date, datetime.min.time()) + timedelta(
                    days=day_offset, hours=vessel_idx * 6
                )
                
                lat = lat_range[0] + (lat_range[1] - lat_range[0]) * (0.1 + 0.8 * vessel_idx / 3)
                lon = lon_range[0] + (lon_range[1] - lon_range[0]) * (0.1 + 0.8 * vessel_idx / 3)
                speed = 8.0 + (2.0 * vessel_idx / 3)  # Knots, range: 8-10
                
                rows.append(
                    {
                        "vessel_id": vessel_id,
                        "timestamp": timestamp,
                        "latitude": lat,
                        "longitude": lon,
                        "speed_knots": speed,
                        "cargo_class": vessel_classes[vessel_idx % len(vessel_classes)],
                    }
                )

        return pd.DataFrame(rows) if rows else pd.DataFrame()

    def _generate_stub_route_events(
        self, commodity: str, as_of_timestamp: datetime
    ) -> pd.DataFrame:
        """
        Generate realistic stub route disruption events.
        
        Occasionally includes rerouting, port delays, weather disruption.
        """
        routes = self.COMMODITY_ROUTES.get(commodity, [])
        if not routes:
            return pd.DataFrame()

        rows = []
        base_date = as_of_timestamp.date()
        
        # Occasional disruptions (realistic: ~1 every 7 days per route)
        for route_idx, route_id in enumerate(routes[:2]):
            disruption_day = (route_idx * 5) % 18
            if 5 <= disruption_day <= 12:  # Mid-window disruption window
                severity = 0.35 + (0.15 * (disruption_day % 3)) / 3  # Range: 0.35-0.50
                event_timestamp = datetime.combine(
                    base_date + timedelta(days=disruption_day), datetime.min.time()
                )
                
                rows.append(
                    {
                        "route_id": route_id,
                        "timestamp": event_timestamp,
                        "event_type": "rerouting" if disruption_day % 2 == 0 else "port_delay",
                        "severity": severity,
                        "detour_distance_ratio": 0.10 + (0.10 * severity),
                    }
                )

        return pd.DataFrame(rows) if rows else pd.DataFrame()


# Singleton instance
shipping_context_builder = ShippingContextBuilder()
