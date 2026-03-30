from __future__ import annotations

from typing import Dict, List, Optional

from ..models import RouteCorridorDefinition
from .zones import ZoneCatalog


class RouteCatalog:
    def __init__(self, zone_catalog: ZoneCatalog):
        self.zone_catalog = zone_catalog
        self.corridors: Dict[str, RouteCorridorDefinition] = dict(zone_catalog.corridors)

    @classmethod
    def from_config(cls) -> "RouteCatalog":
        return cls(ZoneCatalog.from_config())

    def get(self, corridor_id: str) -> Optional[RouteCorridorDefinition]:
        return self.corridors.get(corridor_id)

    def corridors_for_point(self, latitude: float, longitude: float) -> List[str]:
        return [match.zone_id for match in self.zone_catalog.match_point(latitude, longitude, zone_types=["corridor"])]
