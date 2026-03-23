from __future__ import annotations

from typing import Literal

import pandas as pd

from ..config.settings import settings


ExecutionPhase = Literal["entry", "exit"]


def direction_to_sign(direction: str) -> float:
    if direction == "long":
        return 1.0
    if direction == "short":
        return -1.0
    return 0.0


def resolve_price(row: pd.Series, field: str) -> float:
    value = row.get(field)
    if pd.notna(value):
        return float(value)
    fallback = row.get("close")
    if pd.notna(fallback):
        return float(fallback)
    raise ValueError(f"Execution price field '{field}' is unavailable for this bar.")


def slippage_rate(
    row: pd.Series,
    phase: ExecutionPhase,
    median_volume: float | None = None,
    participation: float = 0.0,
) -> float:
    config = settings.execution
    base_bps = config.entry_slippage_bps if phase == "entry" else config.exit_slippage_bps
    spread_bps = config.entry_spread_bps if phase == "entry" else config.exit_spread_bps
    reference_field = config.entry_price_field if phase == "entry" else config.exit_price_field
    reference_price = max(resolve_price(row, reference_field), 1e-12)
    high = float(row.get("high", reference_price))
    low = float(row.get("low", reference_price))
    intraday_range = max(0.0, high - low) / reference_price
    impact_bps = max(0.0, float(participation)) * float(config.impact_coefficient_bps)
    base_rate = (base_bps + 0.5 * spread_bps + impact_bps) / 10000.0
    capped_range_component = min(intraday_range * config.max_slippage_from_range_fraction, intraday_range)
    liquidity_multiplier = 1.0
    if median_volume is not None and median_volume > 0:
        current_volume = float(row.get("volume", median_volume))
        if current_volume < median_volume * config.low_volume_threshold_ratio:
            liquidity_multiplier = config.low_volume_slippage_multiplier
    return (base_rate + capped_range_component) * liquidity_multiplier


def execution_price(
    row: pd.Series,
    direction: str,
    phase: ExecutionPhase,
    median_volume: float | None = None,
    participation: float = 0.0,
) -> float:
    if direction == "neutral":
        reference_field = settings.execution.entry_price_field if phase == "entry" else settings.execution.exit_price_field
        return resolve_price(row, reference_field)

    reference_field = settings.execution.entry_price_field if phase == "entry" else settings.execution.exit_price_field
    reference_price = resolve_price(row, reference_field)
    rate = slippage_rate(row, phase=phase, median_volume=median_volume, participation=participation)

    is_buy = (phase == "entry" and direction == "long") or (phase == "exit" and direction == "short")
    if is_buy:
        return reference_price * (1.0 + rate)
    return reference_price * (1.0 - rate)
