from __future__ import annotations

from typing import Dict, List, Optional

from ..models import ChokepointDefinition
from .zones import ZoneCatalog


class ChokepointCatalog:
    def __init__(self, zone_catalog: ZoneCatalog):
        self.zone_catalog = zone_catalog
        self.chokepoints: Dict[str, ChokepointDefinition] = dict(zone_catalog.chokepoints)

    @classmethod
    def from_config(cls) -> "ChokepointCatalog":
        return cls(ZoneCatalog.from_config())

    def get(self, chokepoint_id: str) -> Optional[ChokepointDefinition]:
        return self.chokepoints.get(chokepoint_id)

    def chokepoints_for_point(self, latitude: float, longitude: float) -> List[str]:
        return [match.zone_id for match in self.zone_catalog.match_point(latitude, longitude, zone_types=["chokepoint"])]
