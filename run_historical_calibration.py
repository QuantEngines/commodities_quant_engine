from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List

import pandas as pd

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from commodities_quant_engine.analytics.backtest.parameter_tuner import MacroParameterTuner
from commodities_quant_engine.config.settings import settings
from commodities_quant_engine.data.storage.local import LocalStorage
from commodities_quant_engine.data.ingestion import market_data_service


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download historical commodity data, persist it, and run governed parameter calibration."
    )
    parser.add_argument(
        "commodities",
        nargs="+",
        help="Commodity symbols, for example GOLD CRUDEOIL COPPER",
    )
    parser.add_argument("--start-date", required=True, help="Start date in YYYY-MM-DD format")
    parser.add_argument("--end-date", required=True, help="End date in YYYY-MM-DD format")
    parser.add_argument("--provider", help="Override exchange/provider for fetching history")
    parser.add_argument("--contract", help="Optional explicit contract symbol")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Ignore cached history and fetch from provider again",
    )
    parser.add_argument(
        "--approve",
        action="store_true",
        help="Approve and promote a successful candidate version if governance checks pass",
    )
    parser.add_argument(
        "--validation-days",
        type=int,
        default=90,
        help="Number of trailing calendar days reserved for validation metrics",
    )
    return parser.parse_args()


def split_validation_window(price_data: pd.DataFrame, validation_days: int) -> tuple[date, date] | tuple[None, None]:
    if price_data.empty or validation_days <= 0:
        return None, None
    end = price_data.index[-1].date()
    start = max(price_data.index[0].date(), end - timedelta(days=validation_days))
    if start >= end:
        return None, None
    return start, end


def run_calibration(
    commodities: List[str],
    start_date: date,
    end_date: date,
    provider: str | None,
    contract: str | None,
    refresh: bool,
    approve: bool,
    validation_days: int,
) -> Dict[str, object]:
    storage = LocalStorage()
    tuner = MacroParameterTuner(storage=storage)
    summary: Dict[str, object] = {
        "generated_at": pd.Timestamp.utcnow().isoformat(),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "provider": provider,
        "contract": contract,
        "approve": approve,
        "validation_days": validation_days,
        "commodities": {},
    }

    for commodity in commodities:
        print(f"\n[{commodity}] Fetching historical price data...")
        frame = market_data_service.load_or_fetch_price_frame(
            commodity=commodity,
            start_date=start_date,
            end_date=end_date,
            contract=contract,
            exchange=provider,
            refresh=refresh,
            persist=True,
        )
        if frame.empty:
            summary["commodities"][commodity] = {"status": "no_data"}
            print(f"  No data returned for {commodity}")
            continue

        cache_path = market_data_service.cache_price_frame(
            commodity=commodity,
            price_frame=frame,
            start_date=start_date,
            end_date=end_date,
            contract=contract,
            exchange=provider,
        )
        print(f"  Cached {len(frame)} rows to {cache_path}")

        validation_start, validation_end = split_validation_window(frame, validation_days)
        result = tuner.optimize_parameters(
            commodity=commodity,
            train_start=start_date,
            train_end=end_date,
            validation_start=validation_start,
            validation_end=validation_end,
            price_data=frame,
            approve=approve,
            dry_run=not approve,
        )

        validation_payload = {
            split: {
                "total_return": metrics.total_return,
                "sharpe_ratio": metrics.sharpe_ratio,
                "win_rate": metrics.win_rate,
                "signal_accuracy": metrics.signal_accuracy,
            }
            for split, metrics in result.validation_results.items()
        }
        summary["commodities"][commodity] = {
            "status": "ok",
            "rows": int(len(frame)),
            "cached_path": cache_path,
            "promoted": bool(result.promoted),
            "incumbent_version": result.incumbent_parameters.name,
            "candidate_version": result.candidate_parameters.name if result.candidate_parameters else None,
            "evidence": result.evidence,
            "validation_results": validation_payload,
        }
        print(
            f"  Calibration complete | incumbent={result.incumbent_parameters.name} | candidate={summary['commodities'][commodity]['candidate_version']} | promoted={result.promoted}"
        )

    report_name = f"historical_calibration_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}"
    storage.write_json(settings.storage.report_store, report_name, summary)
    print(f"\nSaved calibration summary to {settings.storage.base_dir}/{settings.storage.report_store}/{report_name}.json")
    return summary


def main() -> int:
    args = parse_args()
    run_calibration(
        commodities=[commodity.upper() for commodity in args.commodities],
        start_date=date.fromisoformat(args.start_date),
        end_date=date.fromisoformat(args.end_date),
        provider=args.provider,
        contract=args.contract,
        refresh=args.refresh,
        approve=args.approve,
        validation_days=args.validation_days,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())