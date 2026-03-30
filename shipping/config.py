from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

from ..config.settings import settings

PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _resolve_config_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return PACKAGE_ROOT / path


def _load_payload(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore

        loaded = yaml.safe_load(text)
    except Exception:
        loaded = json.loads(text)
    return loaded or {}


def load_shipping_config(path_str: str) -> Dict[str, Any]:
    return _load_payload(_resolve_config_path(path_str))


@lru_cache(maxsize=1)
def load_shipping_sources(path_str: Optional[str] = None) -> Dict[str, Any]:
    return load_shipping_config(path_str or settings.shipping.sources_config_path)


@lru_cache(maxsize=1)
def load_shipping_geographies(path_str: Optional[str] = None) -> Dict[str, Any]:
    return load_shipping_config(path_str or settings.shipping.geographies_config_path)


@lru_cache(maxsize=1)
def load_shipping_features(path_str: Optional[str] = None) -> Dict[str, Any]:
    return load_shipping_config(path_str or settings.shipping.features_config_path)


@lru_cache(maxsize=1)
def load_shipping_signal_rules(path_str: Optional[str] = None) -> Dict[str, Any]:
    return load_shipping_config(path_str or settings.shipping.signal_rules_config_path)


def clear_shipping_config_cache() -> None:
    load_shipping_sources.cache_clear()
    load_shipping_geographies.cache_clear()
    load_shipping_features.cache_clear()
    load_shipping_signal_rules.cache_clear()
