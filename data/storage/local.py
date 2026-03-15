from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd

from ...config.settings import settings

PACKAGE_ROOT = Path(__file__).resolve().parents[2]


class LocalStorage:
    """Local parquet and JSON artifact storage with research-friendly layouts."""

    def __init__(self, base_dir: Optional[str] = None):
        base_path = base_dir or settings.storage.base_dir
        resolved_base = Path(base_path)
        if not resolved_base.is_absolute():
            resolved_base = PACKAGE_ROOT / resolved_base
        self.base_dir = resolved_base
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def resolve(self, *parts: str) -> Path:
        path = self.base_dir.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def save_dataframe(self, df: pd.DataFrame, name: str, partition_cols: Optional[list] = None):
        path = self.resolve(f"{name}.parquet")
        if partition_cols:
            df.to_parquet(path, index=False, partition_cols=partition_cols)
        else:
            df.to_parquet(path, index=False)

    def load_dataframe(self, name: str) -> pd.DataFrame:
        path = self.resolve(f"{name}.parquet")
        if path.exists():
            return pd.read_parquet(path)
        return pd.DataFrame()

    def append_dataframe(
        self,
        df: pd.DataFrame,
        domain: str,
        name: str,
        dedupe_on: Optional[Iterable[str]] = None,
    ) -> Path:
        path = self.resolve(domain, f"{name}.parquet")
        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, df], ignore_index=True)
        else:
            combined = df.copy()

        if dedupe_on:
            combined = combined.drop_duplicates(subset=list(dedupe_on), keep="last")

        combined.to_parquet(path, index=False)
        return path

    def load_domain_dataframe(self, domain: str, name: str) -> pd.DataFrame:
        path = self.resolve(domain, f"{name}.parquet")
        if path.exists():
            return pd.read_parquet(path)
        return pd.DataFrame()

    def write_json(self, domain: str, name: str, payload: Dict[str, Any]) -> Path:
        path = self.resolve(domain, f"{name}.json")
        path.write_text(json.dumps(payload, indent=2, default=str))
        return path

    def read_json(self, domain: str, name: str) -> Dict[str, Any]:
        path = self.resolve(domain, f"{name}.json")
        if path.exists():
            return json.loads(path.read_text())
        return {}

    def append_jsonl(
        self,
        domain: str,
        name: str,
        records: Iterable[Dict[str, Any]],
        compress: bool = True,
    ) -> Path:
        suffix = ".jsonl.gz" if compress else ".jsonl"
        path = self.resolve(domain, f"{name}{suffix}")
        opener = gzip.open if compress else open
        with opener(path, "at", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, separators=(",", ":"), default=str))
                handle.write("\n")
        return path

    def load_jsonl(self, domain: str, name: str) -> list[Dict[str, Any]]:
        compressed = self.resolve(domain, f"{name}.jsonl.gz")
        plain = self.resolve(domain, f"{name}.jsonl")
        if compressed.exists():
            with gzip.open(compressed, "rt", encoding="utf-8") as handle:
                return [json.loads(line) for line in handle if line.strip()]
        if plain.exists():
            with open(plain, "rt", encoding="utf-8") as handle:
                return [json.loads(line) for line in handle if line.strip()]
        return []

    def save_raw_data(self, data: Any, name: str):
        path = self.resolve("raw", name)
        if isinstance(data, str):
            path.write_text(data)
        else:
            path.write_text(json.dumps(data, indent=2, default=str))

    def load_raw_data(self, name: str) -> str:
        path = self.resolve("raw", name)
        if path.exists():
            return path.read_text()
        return ""


storage = LocalStorage()
