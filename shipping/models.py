from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


Coordinate = Tuple[float, float]


@dataclass
class ShippingZone:
    zone_id: str
    name: str
    zone_type: str
    geometry_type: str
    coordinates: List[Coordinate]
    source: str = "config"
    confidence_score: float = 1.0
    geography_tags: List[str] = field(default_factory=list)
    commodity_tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PortDefinition:
    port_id: str
    name: str
    country: str
    zone: ShippingZone
    anchorage_zone_id: Optional[str] = None
    geography_tags: List[str] = field(default_factory=list)
    commodity_tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PortCluster:
    cluster_id: str
    name: str
    port_ids: List[str]
    geography_tags: List[str] = field(default_factory=list)
    commodity_tags: List[str] = field(default_factory=list)
    source: str = "config"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChokepointDefinition:
    chokepoint_id: str
    name: str
    zone: ShippingZone
    geography_tags: List[str] = field(default_factory=list)
    commodity_tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RouteCorridorDefinition:
    corridor_id: str
    name: str
    zone: ShippingZone
    geography_tags: List[str] = field(default_factory=list)
    commodity_tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VesselPosition:
    vessel_id: str
    timestamp: datetime
    latitude: float
    longitude: float
    source: str
    vessel_type: Optional[str] = None
    cargo_class: Optional[str] = None
    speed_knots: Optional[float] = None
    course_degrees: Optional[float] = None
    heading_degrees: Optional[float] = None
    draught_meters: Optional[float] = None
    confidence_score: Optional[float] = None
    data_quality_score: Optional[float] = None
    geography_tags: List[str] = field(default_factory=list)
    commodity_tags: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VesselTrack:
    track_id: str
    vessel_id: str
    start_time: datetime
    end_time: datetime
    positions: List[VesselPosition]
    source: str
    confidence_score: Optional[float] = None
    data_quality_score: Optional[float] = None
    vessel_type: Optional[str] = None
    cargo_class: Optional[str] = None
    geography_tags: List[str] = field(default_factory=list)
    commodity_tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VesselEvent:
    event_id: str
    vessel_id: str
    event_type: str
    timestamp: datetime
    source: str
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    zone_id: Optional[str] = None
    zone_type: Optional[str] = None
    severity: float = 0.0
    confidence_score: Optional[float] = None
    vessel_type: Optional[str] = None
    cargo_class: Optional[str] = None
    geography_tags: List[str] = field(default_factory=list)
    commodity_tags: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PortCall:
    call_id: str
    vessel_id: str
    port_id: str
    arrival_time: datetime
    source: str
    departure_time: Optional[datetime] = None
    berth_time: Optional[datetime] = None
    anchorage_start_time: Optional[datetime] = None
    anchorage_end_time: Optional[datetime] = None
    vessel_type: Optional[str] = None
    cargo_class: Optional[str] = None
    confidence_score: Optional[float] = None
    data_quality_score: Optional[float] = None
    geography_tags: List[str] = field(default_factory=list)
    commodity_tags: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ChokepointEvent:
    event_id: str
    chokepoint_id: str
    timestamp: datetime
    status: str
    source: str
    severity: float = 0.0
    expected_duration_hours: Optional[float] = None
    confidence_score: Optional[float] = None
    geography_tags: List[str] = field(default_factory=list)
    commodity_tags: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RouteEvent:
    event_id: str
    route_id: str
    timestamp: datetime
    event_type: str
    source: str
    severity: float = 0.0
    detour_distance_ratio: Optional[float] = None
    chokepoint_id: Optional[str] = None
    confidence_score: Optional[float] = None
    geography_tags: List[str] = field(default_factory=list)
    commodity_tags: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ShippingObservationWindow:
    start_time: datetime
    end_time: datetime
    frequency: str
    source: str
    commodity: Optional[str] = None
    geography_scope: List[str] = field(default_factory=list)
    source_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ShippingFeatureVector:
    timestamp: datetime
    commodity: Optional[str]
    features: Dict[str, float]
    source: str
    observation_window: ShippingObservationWindow
    quality_score: float = 0.0
    confidence_score: float = 0.0
    geography_tags: List[str] = field(default_factory=list)
    commodity_tags: List[str] = field(default_factory=list)
    key_drivers: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ShippingSignalContext:
    commodity: str
    timestamp: datetime
    shipping_alignment_score: float
    shipping_conflict_score: float
    shipping_risk_penalty: float
    shipping_data_quality_score: float
    shipping_data_quality_penalty: float
    shipping_support_boost: float
    shipping_directional_bias: float
    shipping_regime_bias: float
    shipping_summary: str
    shipping_explanation_summary: str
    key_shipping_drivers: List[str] = field(default_factory=list)
    route_chokepoint_notes: List[str] = field(default_factory=list)
    port_congestion_notes: List[str] = field(default_factory=list)
    shipping_features: Dict[str, float] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_active(self) -> bool:
        return bool(self.shipping_features)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def empty(cls, commodity: str, timestamp: datetime) -> "ShippingSignalContext":
        return cls(
            commodity=commodity,
            timestamp=timestamp,
            shipping_alignment_score=0.0,
            shipping_conflict_score=0.0,
            shipping_risk_penalty=0.0,
            shipping_data_quality_score=0.0,
            shipping_data_quality_penalty=0.0,
            shipping_support_boost=0.0,
            shipping_directional_bias=0.0,
            shipping_regime_bias=0.0,
            shipping_summary="No shipping overlay",
            shipping_explanation_summary="Shipping layer inactive or no usable data available.",
        )
