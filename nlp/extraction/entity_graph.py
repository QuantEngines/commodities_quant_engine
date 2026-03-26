from __future__ import annotations

from dataclasses import dataclass
from typing import List


_COUNTRIES = {
    "usa",
    "us",
    "india",
    "china",
    "russia",
    "ukraine",
    "saudi",
    "iran",
    "iraq",
    "uae",
    "qatar",
    "brazil",
}

_SHIPPING_LANES = {
    "suez",
    "hormuz",
    "malacca",
    "black sea",
    "panama",
    "red sea",
}

_PRODUCERS = {
    "opec",
    "aramco",
    "gazprom",
    "rio tinto",
    "bhp",
    "vale",
    "codelco",
    "nornickel",
}


@dataclass
class EntityGraphSignals:
    countries: List[str]
    shipping_lanes: List[str]
    producers: List[str]

    def encoded(self) -> List[str]:
        encoded: List[str] = []
        encoded.extend(f"country:{name}" for name in self.countries)
        encoded.extend(f"lane:{name}" for name in self.shipping_lanes)
        encoded.extend(f"producer:{name}" for name in self.producers)
        return encoded


def extract_entity_graph_signals(text: str) -> EntityGraphSignals:
    lowered = text.lower()
    countries = sorted(name for name in _COUNTRIES if name in lowered)
    shipping_lanes = sorted(name for name in _SHIPPING_LANES if name in lowered)
    producers = sorted(name for name in _PRODUCERS if name.lower() in lowered)
    return EntityGraphSignals(countries=countries, shipping_lanes=shipping_lanes, producers=producers)
