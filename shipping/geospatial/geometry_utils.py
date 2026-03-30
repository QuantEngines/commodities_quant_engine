from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

from ..models import Coordinate, ShippingZone


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius_km = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2.0) ** 2
    return 2.0 * radius_km * math.atan2(math.sqrt(a), math.sqrt(max(1.0 - a, 0.0)))


def point_in_bbox(latitude: float, longitude: float, bbox: Sequence[float]) -> bool:
    min_lat, min_lon, max_lat, max_lon = [float(item) for item in bbox]
    return min_lat <= latitude <= max_lat and min_lon <= longitude <= max_lon


def point_in_polygon(latitude: float, longitude: float, polygon: Iterable[Coordinate]) -> bool:
    points = list(polygon)
    if len(points) < 3:
        return False
    inside = False
    j = len(points) - 1
    for i, (lat_i, lon_i) in enumerate(points):
        lat_j, lon_j = points[j]
        intersects = ((lon_i > longitude) != (lon_j > longitude)) and (
            latitude < (lat_j - lat_i) * (longitude - lon_i) / ((lon_j - lon_i) or 1e-12) + lat_i
        )
        if intersects:
            inside = not inside
        j = i
    return inside


def zone_contains_point(zone: ShippingZone, latitude: float, longitude: float) -> bool:
    if zone.geometry_type == "bbox":
        if not zone.coordinates:
            return False
        bbox = [zone.coordinates[0][0], zone.coordinates[0][1], zone.coordinates[1][0], zone.coordinates[1][1]]
        return point_in_bbox(latitude, longitude, bbox)
    return point_in_polygon(latitude, longitude, zone.coordinates)
