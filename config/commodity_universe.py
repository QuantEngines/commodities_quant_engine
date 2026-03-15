from __future__ import annotations

from typing import Dict, List


# Broad MCX-aligned commodity registry used as the local-first default universe.
# The symbol list intentionally focuses on commodity products rather than indices.
# Exact lot sizes, tick sizes, and expiry calendars should still come from curated
# contract masters when the user has them. These defaults are operational fallbacks
# so the engine can reason over the broader MCX commodity surface without pretending
# to have exchange-perfect metadata for every product variant.

GOLD_FAMILY = ["GOLD", "GOLDM", "GOLDGUINEA", "GOLDPETAL", "GOLDTEN"]
SILVER_FAMILY = ["SILVER", "SILVERM", "SILVERMIC", "SILVER1000"]
PRECIOUS_METALS = GOLD_FAMILY + SILVER_FAMILY

BASE_METALS = [
    "ALUMINIUM",
    "ALUMINI",
    "BRASSPHY",
    "COPPER",
    "COPPERM",
    "LEAD",
    "LEADMINI",
    "NICKEL",
    "NICKELM",
    "STEELREBAR",
    "TIN",
    "ZINC",
    "ZINCMINI",
]

ENERGY_COMPLEX = [
    "BRCRUDEOIL",
    "CRUDEOIL",
    "CRUDEOILM",
    "ELECDMBL",
    "NATURALGAS",
    "NATGASMINI",
]

AGRI_COMPLEX = [
    "ALMOND",
    "CARDAMOM",
    "CASTORSEED",
    "CHANA",
    "CORIANDER",
    "COTTON",
    "COTTONCNDY",
    "COTTONOIL",
    "CPO",
    "GUARGUM",
    "GUARSEED",
    "KAPAS",
    "KAPASKHALI",
    "MAIZE",
    "MENTHAOIL",
    "PEPPER",
    "POTATO",
    "RAWJUTE",
    "RBDPALMOLEIN",
    "RUBBER",
    "SOYABEAN",
    "SUGARMDEL",
    "SUGARMKOL",
    "SUGARSKLP",
    "WHEAT",
]

MCX_INDEX_PRODUCTS = ["MCXBULLDEX", "MCXENRGDEX", "MCXMETLDEX"]
MCX_COMMODITY_UNIVERSE = PRECIOUS_METALS + BASE_METALS + ENERGY_COMPLEX + AGRI_COMPLEX


def _family_metadata(names: Dict[str, str], segment: str, multiplier_map: Dict[str, int], tick_map: Dict[str, float], macro_sensitivity: List[str]) -> Dict[str, Dict[str, object]]:
    definitions: Dict[str, Dict[str, object]] = {}
    for symbol, name in names.items():
        definitions[symbol] = {
            "name": name,
            "segment": segment,
            "contract_multiplier": multiplier_map.get(symbol, 1),
            "tick_size": tick_map.get(symbol, 0.01),
            "expiry_rule": "monthly_last_friday",
            "macro_sensitivity": list(macro_sensitivity),
        }
    return definitions


def default_mcx_commodity_definitions() -> Dict[str, Dict[str, object]]:
    definitions: Dict[str, Dict[str, object]] = {}

    definitions.update(
        _family_metadata(
            {
                "GOLD": "Gold",
                "GOLDM": "Gold Mini",
                "GOLDGUINEA": "Gold Guinea",
                "GOLDPETAL": "Gold Petal",
                "GOLDTEN": "Gold Ten",
            },
            "bullion",
            {"GOLD": 100, "GOLDM": 10, "GOLDGUINEA": 8, "GOLDPETAL": 1, "GOLDTEN": 10},
            {"GOLD": 1.0, "GOLDM": 1.0, "GOLDGUINEA": 1.0, "GOLDPETAL": 1.0, "GOLDTEN": 1.0},
            ["real_rates", "risk_off", "usd_strength", "inflation_expectations"],
        )
    )

    definitions.update(
        _family_metadata(
            {
                "SILVER": "Silver",
                "SILVERM": "Silver Mini",
                "SILVERMIC": "Silver Micro",
                "SILVER1000": "Silver 1000",
            },
            "bullion",
            {"SILVER": 30, "SILVERM": 5, "SILVERMIC": 1, "SILVER1000": 1},
            {"SILVER": 1.0, "SILVERM": 1.0, "SILVERMIC": 1.0, "SILVER1000": 1.0},
            ["real_rates", "risk_off", "industrial_demand", "inflation_expectations"],
        )
    )

    definitions.update(
        _family_metadata(
            {
                "ALUMINIUM": "Aluminium",
                "ALUMINI": "Aluminium Mini",
                "BRASSPHY": "Brass Physical",
                "COPPER": "Copper",
                "COPPERM": "Copper Mini",
                "LEAD": "Lead",
                "LEADMINI": "Lead Mini",
                "NICKEL": "Nickel",
                "NICKELM": "Nickel Mini",
                "STEELREBAR": "Steel Rebar",
                "TIN": "Tin",
                "ZINC": "Zinc",
                "ZINCMINI": "Zinc Mini",
            },
            "base_metals",
            {
                "ALUMINIUM": 5000,
                "ALUMINI": 1000,
                "BRASSPHY": 1000,
                "COPPER": 1000,
                "COPPERM": 100,
                "LEAD": 5000,
                "LEADMINI": 1000,
                "NICKEL": 250,
                "NICKELM": 100,
                "STEELREBAR": 10000,
                "TIN": 250,
                "ZINC": 5000,
                "ZINCMINI": 1000,
            },
            {
                "ALUMINIUM": 0.05,
                "ALUMINI": 0.05,
                "BRASSPHY": 0.05,
                "COPPER": 0.05,
                "COPPERM": 0.05,
                "LEAD": 0.05,
                "LEADMINI": 0.05,
                "NICKEL": 0.1,
                "NICKELM": 0.1,
                "STEELREBAR": 0.05,
                "TIN": 0.1,
                "ZINC": 0.05,
                "ZINCMINI": 0.05,
            },
            ["growth_expectations", "usd_strength", "industrial_production"],
        )
    )

    definitions.update(
        _family_metadata(
            {
                "BRCRUDEOIL": "Brent Crude Oil",
                "CRUDEOIL": "Crude Oil",
                "CRUDEOILM": "Crude Oil Mini",
                "ELECDMBL": "Electricity Daily Base Load",
                "NATURALGAS": "Natural Gas",
                "NATGASMINI": "Natural Gas Mini",
            },
            "energy",
            {
                "BRCRUDEOIL": 100,
                "CRUDEOIL": 100,
                "CRUDEOILM": 10,
                "ELECDMBL": 1,
                "NATURALGAS": 1250,
                "NATGASMINI": 250,
            },
            {
                "BRCRUDEOIL": 1.0,
                "CRUDEOIL": 1.0,
                "CRUDEOILM": 1.0,
                "ELECDMBL": 1.0,
                "NATURALGAS": 0.1,
                "NATGASMINI": 0.1,
            },
            ["growth_expectations", "usd_strength", "geopolitical_risk"],
        )
    )

    definitions.update(
        _family_metadata(
            {
                "ALMOND": "Almond",
                "CARDAMOM": "Cardamom",
                "CASTORSEED": "Castor Seed",
                "CHANA": "Chana",
                "CORIANDER": "Coriander",
                "COTTON": "Cotton",
                "COTTONCNDY": "Cotton Candy",
                "COTTONOIL": "Cotton Seed Wash Oil",
                "CPO": "Crude Palm Oil",
                "GUARGUM": "Guar Gum",
                "GUARSEED": "Guar Seed",
                "KAPAS": "Kapas",
                "KAPASKHALI": "Kapas Khali",
                "MAIZE": "Maize",
                "MENTHAOIL": "Mentha Oil",
                "PEPPER": "Pepper",
                "POTATO": "Potato",
                "RAWJUTE": "Raw Jute",
                "RBDPALMOLEIN": "RBD Palmolein",
                "RUBBER": "Rubber",
                "SOYABEAN": "Soyabean",
                "SUGARMDEL": "Sugar M Delhi",
                "SUGARMKOL": "Sugar M Kolhapur",
                "SUGARSKLP": "Sugar S Kolhapur",
                "WHEAT": "Wheat",
            },
            "agri",
            {},
            {},
            ["weather_risk", "supply_shock", "usd_strength"],
        )
    )

    return definitions


def default_macro_sensitivities() -> Dict[str, List[str]]:
    return {
        **{symbol: ["real_rates", "risk_off", "usd_strength", "inflation_expectations"] for symbol in GOLD_FAMILY},
        **{symbol: ["real_rates", "risk_off", "industrial_demand", "inflation_expectations"] for symbol in SILVER_FAMILY},
        **{symbol: ["growth_expectations", "usd_strength", "industrial_production"] for symbol in BASE_METALS},
        **{symbol: ["growth_expectations", "usd_strength", "geopolitical_risk"] for symbol in ENERGY_COMPLEX},
        **{symbol: ["weather_risk", "supply_shock", "usd_strength"] for symbol in AGRI_COMPLEX},
    }


def default_news_topics() -> Dict[str, List[str]]:
    shared_gold = ["gold", "bullion", "central bank buying", "real rates"]
    shared_silver = ["silver", "bullion", "industrial demand", "solar demand"]
    shared_metals = ["industrial metals", "china growth", "manufacturing", "lme"]
    shared_energy = ["oil", "energy", "opec", "inventory", "refining"]
    shared_gas = ["natural gas", "lng", "gas demand", "weather"]
    shared_agri = ["weather", "crop", "mandi", "sowing", "harvest"]
    return {
        **{symbol: shared_gold for symbol in GOLD_FAMILY},
        **{symbol: shared_silver for symbol in SILVER_FAMILY},
        **{symbol: ["aluminium", *shared_metals] for symbol in ("ALUMINIUM", "ALUMINI")},
        "BRASSPHY": ["brass", "copper", "zinc", "fabrication"],
        **{symbol: ["copper", *shared_metals] for symbol in ("COPPER", "COPPERM")},
        **{symbol: ["lead", "battery demand", *shared_metals] for symbol in ("LEAD", "LEADMINI")},
        **{symbol: ["nickel", "stainless steel", *shared_metals] for symbol in ("NICKEL", "NICKELM")},
        "STEELREBAR": ["steel", "construction", "infrastructure", "iron ore"],
        "TIN": ["tin", "electronics", "solder", "industrial metals"],
        **{symbol: ["zinc", *shared_metals] for symbol in ("ZINC", "ZINCMINI")},
        **{symbol: ["crude oil", "oil", "opec", "petroleum", "inventory"] for symbol in ("CRUDEOIL", "CRUDEOILM")},
        "BRCRUDEOIL": ["brent", "crude oil", "north sea", "opec"],
        "ELECDMBL": ["electricity", "power demand", "power exchange", "heatwave"],
        "NATURALGAS": shared_gas,
        "NATGASMINI": shared_gas,
        **{symbol: [symbol.lower(), *shared_agri] for symbol in AGRI_COMPLEX},
        "COTTON": ["cotton", "textiles", "weather", "exports"],
        "COTTONCNDY": ["cotton", "textiles", "weather", "exports"],
        "COTTONOIL": ["cottonseed oil", "edible oil", "oilseed", "weather"],
        "CPO": ["crude palm oil", "palm oil", "edible oil", "biodiesel"],
        "GUARGUM": ["guar gum", "guar", "shale demand", "exports"],
        "GUARSEED": ["guar seed", "guar", "weather", "exports"],
        "KAPAS": ["kapas", "cotton", "weather", "arrivals"],
        "KAPASKHALI": ["kapas khali", "oilcake", "feed demand", "cottonseed"],
        "RBDPALMOLEIN": ["rbd palmolein", "palm oil", "edible oil", "imports"],
        "SUGARMDEL": ["sugar", "ethanol", "cane", "mills"],
        "SUGARMKOL": ["sugar", "ethanol", "cane", "mills"],
        "SUGARSKLP": ["sugar", "ethanol", "cane", "mills"],
    }


def commodity_keyword_map() -> Dict[str, List[str]]:
    base_map = {
        "GOLD": ["gold", "bullion"],
        "GOLDM": ["gold", "bullion"],
        "GOLDGUINEA": ["gold guinea", "gold", "bullion"],
        "GOLDPETAL": ["gold petal", "gold", "bullion"],
        "GOLDTEN": ["gold ten", "gold", "bullion"],
        "SILVER": ["silver"],
        "SILVERM": ["silver"],
        "SILVERMIC": ["silver micro", "silver"],
        "SILVER1000": ["silver 1000", "silver"],
        "ALUMINIUM": ["aluminium", "aluminum"],
        "ALUMINI": ["aluminium", "aluminum"],
        "BRASSPHY": ["brass"],
        "COPPER": ["copper"],
        "COPPERM": ["copper"],
        "LEAD": ["lead metal", "lead prices"],
        "LEADMINI": ["lead metal", "lead prices"],
        "NICKEL": ["nickel"],
        "NICKELM": ["nickel"],
        "STEELREBAR": ["steel rebar", "steel"],
        "TIN": ["tin metal", "tin prices"],
        "ZINC": ["zinc"],
        "ZINCMINI": ["zinc"],
        "BRCRUDEOIL": ["brent", "crude oil", "petroleum"],
        "CRUDEOIL": ["crude", "oil", "petroleum"],
        "CRUDEOILM": ["crude", "oil", "petroleum"],
        "ELECDMBL": ["electricity", "power demand"],
        "NATURALGAS": ["natural gas", "lng", "gas"],
        "NATGASMINI": ["natural gas", "lng", "gas"],
        "ALMOND": ["almond", "badam"],
        "CARDAMOM": ["cardamom", "elaichi"],
        "CASTORSEED": ["castor seed", "castor"],
        "CHANA": ["chana", "gram"],
        "CORIANDER": ["coriander", "dhania"],
        "COTTON": ["cotton"],
        "COTTONCNDY": ["cotton", "cotton candy"],
        "COTTONOIL": ["cottonseed oil", "cotton oil"],
        "CPO": ["crude palm oil", "palm oil"],
        "GUARGUM": ["guar gum", "guar"],
        "GUARSEED": ["guar seed", "guar"],
        "KAPAS": ["kapas", "cotton"],
        "KAPASKHALI": ["kapas khali", "oilcake", "feed"],
        "MAIZE": ["maize", "corn"],
        "MENTHAOIL": ["mentha oil", "mint oil", "mentha"],
        "PEPPER": ["pepper", "black pepper"],
        "POTATO": ["potato"],
        "RAWJUTE": ["raw jute", "jute"],
        "RBDPALMOLEIN": ["rbd palmolein", "palmolein", "palm oil"],
        "RUBBER": ["rubber", "natural rubber"],
        "SOYABEAN": ["soyabean", "soybean", "soya"],
        "SUGARMDEL": ["sugar", "cane sugar"],
        "SUGARMKOL": ["sugar", "cane sugar"],
        "SUGARSKLP": ["sugar", "cane sugar"],
        "WHEAT": ["wheat"],
    }
    return {symbol: list(keywords) for symbol, keywords in base_map.items()}


def precious_metal_family(symbol: str) -> bool:
    return symbol in PRECIOUS_METALS


def industrial_metal_family(symbol: str) -> bool:
    return symbol in BASE_METALS
