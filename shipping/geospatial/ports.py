from __future__ import annotations

from typing import Dict, List, Optional

from ..models import PortCluster, PortDefinition
from .zones import ZoneCatalog


class PortCatalog:
    def __init__(self, zone_catalog: ZoneCatalog):
        self.zone_catalog = zone_catalog
        self.ports: Dict[str, PortDefinition] = dict(zone_catalog.ports)
        self.clusters: Dict[str, PortCluster] = dict(zone_catalog.clusters)

    @classmethod
    def from_config(cls) -> "PortCatalog":
        return cls(ZoneCatalog.from_config())

    def get(self, port_id: str) -> Optional[PortDefinition]:
        return self.ports.get(port_id)

    def cluster_for_port(self, port_id: str) -> Optional[PortCluster]:
        for cluster in self.clusters.values():
            if port_id in cluster.port_ids:
                return cluster
        return None

    def ports_for_point(self, latitude: float, longitude: float) -> List[str]:
        return [match.zone_id for match in self.zone_catalog.match_point(latitude, longitude, zone_types=["port"])]
