from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from ...config.settings import settings
from ..models import DataQualityReport


REQUIRED_COLUMNS = ("open", "high", "low", "close", "volume")


class MarketDataValidator:
    """Validate market data before research or signal generation."""

    def validate(
        self,
        data: pd.DataFrame,
        as_of: Optional[datetime] = None,
        min_history_rows: Optional[int] = None,
    ) -> DataQualityReport:
        issues = []
        stats = {
            "rows": float(len(data)),
            "missing_close_pct": 0.0,
            "duplicate_index_rows": 0.0,
        }

        if data.empty:
            return DataQualityReport(
                flag="incomplete",
                issues=["No market data supplied."],
                stats=stats,
                as_of=as_of or datetime.now(timezone.utc).replace(tzinfo=None),
                is_valid=False,
            )

        if not isinstance(data.index, pd.DatetimeIndex):
            issues.append("Market data index must be a DatetimeIndex.")
        else:
            if not data.index.is_monotonic_increasing:
                issues.append("Market data index is not sorted in ascending time order.")
            duplicate_count = int(data.index.duplicated().sum())
            stats["duplicate_index_rows"] = float(duplicate_count)
            if duplicate_count:
                issues.append("Market data contains duplicate timestamps.")

        for column in REQUIRED_COLUMNS:
            if column not in data.columns:
                issues.append(f"Missing required column: {column}.")

        if "close" in data.columns:
            stats["missing_close_pct"] = float(data["close"].isna().mean())
            if data["close"].isna().any():
                issues.append("Close series contains missing values.")

        if {"high", "low", "close"}.issubset(data.columns):
            invalid_bars = ((data["high"] < data["low"]) | (data["close"] > data["high"]) | (data["close"] < data["low"])).sum()
            stats["invalid_bar_rows"] = float(invalid_bars)
            if invalid_bars:
                issues.append("Some OHLC rows violate basic price bounds.")

        if {"open", "high", "low", "close"}.issubset(data.columns):
            non_positive_prices = (data[["open", "high", "low", "close"]] <= 0).any(axis=1).sum()
            stats["non_positive_price_rows"] = float(non_positive_prices)
            if non_positive_prices:
                issues.append("Some OHLC rows contain non-positive prices.")

        if "volume" in data.columns:
            negative_volume = int((data["volume"] < 0).sum())
            stats["negative_volume_rows"] = float(negative_volume)
            if negative_volume:
                issues.append("Volume contains negative values.")

        required_rows = min_history_rows or settings.signal.min_history_rows
        if len(data) < required_rows:
            issues.append(f"Only {len(data)} rows available; need at least {required_rows}.")

        if as_of and isinstance(data.index, pd.DatetimeIndex):
            last_ts = data.index[-1].to_pydatetime()
            staleness_days = float((pd.Timestamp(as_of) - pd.Timestamp(last_ts)).days)
            stats["staleness_days"] = max(0.0, staleness_days)
            if staleness_days > settings.signal.max_staleness_days:
                issues.append(f"Latest market data is stale by {staleness_days:.0f} days.")

        flag = "good"
        has_schema_or_index_issue = any(
            "Missing required column" in issue
            or "must be a DatetimeIndex" in issue
            or "not sorted in ascending" in issue
            or "duplicate timestamps" in issue
            for issue in issues
        )
        has_value_integrity_issue = any(
            "missing values" in issue.lower()
            or "violate basic price bounds" in issue.lower()
            or "non-positive prices" in issue.lower()
            or "negative values" in issue.lower()
            for issue in issues
        )
        has_history_issue = any("need at least" in issue for issue in issues)
        is_valid = not (has_schema_or_index_issue or has_value_integrity_issue or has_history_issue)

        if not is_valid:
            flag = "incomplete"
        elif any("stale" in issue.lower() for issue in issues):
            flag = "stale"

        return DataQualityReport(
            flag=flag,
            issues=issues,
            stats=stats,
            as_of=as_of or (data.index[-1].to_pydatetime() if isinstance(data.index, pd.DatetimeIndex) else datetime.now(timezone.utc).replace(tzinfo=None)),
            is_valid=is_valid,
        )
