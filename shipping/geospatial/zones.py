from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

import pandas as pd

from ..config import load_shipping_geographies
from ..models import ChokepointDefinition, PortCluster, PortDefinition, RouteCorridorDefinition, ShippingZone
from .geometry_utils import zone_contains_point


def _build_zone(zone_id: str, name: str, zone_type: str, payload: Dict[str, object]) -> ShippingZone:
    geometry = dict(payload.get("geometry", {}) or {})
    geometry_type = str(geometry.get("type", "bbox"))
    raw_coordinates = list(geometry.get("coordinates", []) or [])
    if geometry_type == "bbox" and len(raw_coordinates) == 4:
        coordinates = [
            (float(raw_coordinates[0]), float(raw_coordinates[1])),
            (float(raw_coordinates[2]), float(raw_coordinates[3])),
        ]
    else:
        coordinates = [(float(pair[0]), float(pair[1])) for pair in raw_coordinates]
    return ShippingZone(
        zone_id=zone_id,
        name=name,
        zone_type=zone_type,
        geometry_type=geometry_type,
        coordinates=coordinates,
        geography_tags=list(payload.get("geography_tags", []) or []),
        commodity_tags=list(payload.get("commodity_tags", []) or []),
        metadata={key: value for key, value in payload.items() if key not in {"geometry", "geography_tags", "commodity_tags"}},
    )


@dataclass
class ZoneMatch:
    zone_id: str
    zone_type: str
    name: str
    geography_tags: List[str]
    commodity_tags: List[str]


class ZoneCatalog:
    def __init__(
        self,
        zones: Iterable[ShippingZone],
        ports: Optional[Dict[str, PortDefinition]] = None,
        clusters: Optional[Dict[str, PortCluster]] = None,
        chokepoints: Optional[Dict[str, ChokepointDefinition]] = None,
        corridors: Optional[Dict[str, RouteCorridorDefinition]] = None,
    ):
        self.zones = {zone.zone_id: zone for zone in zones}
        self.ports = ports or {}
        self.clusters = clusters or {}
        self.chokepoints = chokepoints or {}
        self.corridors = corridors or {}

    @classmethod
    def from_config(cls) -> "ZoneCatalog":
        payload = load_shipping_geographies()
        zones: List[ShippingZone] = []
        ports: Dict[str, PortDefinition] = {}
        clusters: Dict[str, PortCluster] = {}
        chokepoints: Dict[str, ChokepointDefinition] = {}
        corridors: Dict[str, RouteCorridorDefinition] = {}

        for port_payload in payload.get("ports", []) or []:
            zone = _build_zone(
                zone_id=str(port_payload["port_id"]),
                name=str(port_payload["name"]),
                zone_type="port",
                payload=port_payload,
            )
            zones.append(zone)
            ports[zone.zone_id] = PortDefinition(
                port_id=zone.zone_id,
                name=zone.name,
                country=str(port_payload.get("country", "unknown")),
                zone=zone,
                anchorage_zone_id=port_payload.get("anchorage_zone_id"),
                geography_tags=list(port_payload.get("geography_tags", []) or []),
                commodity_tags=list(port_payload.get("commodity_tags", []) or []),
            )

        for anchorage_payload in payload.get("anchorage_zones", []) or []:
            zones.append(
                _build_zone(
                    zone_id=str(anchorage_payload["zone_id"]),
                    name=str(anchorage_payload["name"]),
                    zone_type=str(anchorage_payload.get("zone_type", "anchorage")),
                    payload=anchorage_payload,
                )
            )

        for chokepoint_payload in payload.get("chokepoints", []) or []:
            zone = _build_zone(
                zone_id=str(chokepoint_payload["chokepoint_id"]),
                name=str(chokepoint_payload["name"]),
                zone_type="chokepoint",
                payload=chokepoint_payload,
            )
            zones.append(zone)
            chokepoints[zone.zone_id] = ChokepointDefinition(
                chokepoint_id=zone.zone_id,
                name=zone.name,
                zone=zone,
                geography_tags=list(chokepoint_payload.get("geography_tags", []) or []),
                commodity_tags=list(chokepoint_payload.get("commodity_tags", []) or []),
            )

        for corridor_payload in payload.get("route_corridors", []) or []:
            zone = _build_zone(
                zone_id=str(corridor_payload["corridor_id"]),
                name=str(corridor_payload["name"]),
                zone_type="corridor",
                payload=corridor_payload,
            )
            zones.append(zone)
            corridors[zone.zone_id] = RouteCorridorDefinition(
                corridor_id=zone.zone_id,
                name=zone.name,
                zone=zone,
                geography_tags=list(corridor_payload.get("geography_tags", []) or []),
                commodity_tags=list(corridor_payload.get("commodity_tags", []) or []),
            )

        for cluster_payload in payload.get("port_clusters", []) or []:
            cluster = PortCluster(
                cluster_id=str(cluster_payload["cluster_id"]),
                name=str(cluster_payload["name"]),
                port_ids=list(cluster_payload.get("port_ids", []) or []),
                geography_tags=list(cluster_payload.get("geography_tags", []) or []),
                commodity_tags=list(cluster_payload.get("commodity_tags", []) or []),
            )
            clusters[cluster.cluster_id] = cluster

        return cls(zones=zones, ports=ports, clusters=clusters, chokepoints=chokepoints, corridors=corridors)

    def get(self, zone_id: str) -> Optional[ShippingZone]:
        return self.zones.get(zone_id)

    def by_type(self, zone_type: str) -> List[ShippingZone]:
        return [zone for zone in self.zones.values() if zone.zone_type == zone_type]

    def match_point(self, latitude: float, longitude: float, zone_types: Optional[Iterable[str]] = None) -> List[ZoneMatch]:
        allowed = set(zone_types or [])
        matches: List[ZoneMatch] = []
        for zone in self.zones.values():
            if allowed and zone.zone_type not in allowed:
                continue
            if zone_contains_point(zone, latitude, longitude):
                matches.append(
                    ZoneMatch(
                        zone_id=zone.zone_id,
                        zone_type=zone.zone_type,
                        name=zone.name,
                        geography_tags=list(zone.geography_tags),
                        commodity_tags=list(zone.commodity_tags),
                    )
                )
        return matches

    def annotate_positions(self, positions: pd.DataFrame) -> pd.DataFrame:
        if positions.empty:
            return positions.copy()
        annotated = positions.copy()
        port_ids: List[List[str]] = []
        anchorage_ids: List[List[str]] = []
        chokepoint_ids: List[List[str]] = []
        corridor_ids: List[List[str]] = []
        geography_tags: List[List[str]] = []
        commodity_tags: List[List[str]] = []
        zone_names: List[List[str]] = []
        for row in annotated.itertuples(index=False):
            matches = self.match_point(float(row.latitude), float(row.longitude))
            by_type: Dict[str, List[str]] = {"port": [], "anchorage": [], "chokepoint": [], "corridor": []}
            tags_geo: List[str] = []
            tags_commodity: List[str] = []
            names: List[str] = []
            for match in matches:
                by_type.setdefault(match.zone_type, []).append(match.zone_id)
                tags_geo.extend(match.geography_tags)
                tags_commodity.extend(match.commodity_tags)
                names.append(match.name)
            port_ids.append(sorted(set(by_type.get("port", []))))
            anchorage_ids.append(sorted(set(by_type.get("anchorage", []))))
            chokepoint_ids.append(sorted(set(by_type.get("chokepoint", []))))
            corridor_ids.append(sorted(set(by_type.get("corridor", []))))
            geography_tags.append(sorted(set(tags_geo)))
            commodity_tags.append(sorted(set(tags_commodity)))
            zone_names.append(sorted(set(names)))
        annotated["port_ids"] = port_ids
        annotated["anchorage_ids"] = anchorage_ids
        annotated["chokepoint_ids"] = chokepoint_ids
        annotated["corridor_ids"] = corridor_ids
        annotated["matched_zone_names"] = zone_names
        annotated["geography_tags"] = geography_tags
        annotated["commodity_tags"] = commodity_tags
        return annotated
