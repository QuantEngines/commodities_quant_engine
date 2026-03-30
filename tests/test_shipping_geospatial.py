from datetime import datetime

import pandas as pd

from ..shipping.geospatial import ZoneCatalog


def test_zone_catalog_matches_ports_and_anchorages_from_config():
    catalog = ZoneCatalog.from_config()
    matches = catalog.match_point(25.16, 56.31)
    matched_ids = {match.zone_id for match in matches}

    assert "FUJAIRAH" in matched_ids
    assert "ANCHORAGE_FUJAIRAH" in matched_ids
    assert "GULF_TO_SINGAPORE" in matched_ids


def test_zone_catalog_matches_chokepoints_and_corridors():
    catalog = ZoneCatalog.from_config()
    matches = catalog.match_point(26.10, 56.90)
    matched_ids = {match.zone_id for match in matches}

    assert "HORMUZ" in matched_ids
    assert "GULF_TO_SINGAPORE" in matched_ids


def test_zone_catalog_annotates_position_frame():
    catalog = ZoneCatalog.from_config()
    positions = pd.DataFrame(
        [
            {"vessel_id": "v1", "timestamp": datetime(2026, 3, 1, 0, 0), "latitude": 25.16, "longitude": 56.31, "speed_knots": 0.4, "cargo_class": "crude_tanker"},
            {"vessel_id": "v1", "timestamp": datetime(2026, 3, 1, 6, 0), "latitude": 26.10, "longitude": 56.90, "speed_knots": 11.0, "cargo_class": "crude_tanker"},
        ]
    )

    annotated = catalog.annotate_positions(positions)

    assert "FUJAIRAH" in annotated.loc[0, "port_ids"]
    assert "ANCHORAGE_FUJAIRAH" in annotated.loc[0, "anchorage_ids"]
    assert "HORMUZ" in annotated.loc[1, "chokepoint_ids"]
