from __future__ import annotations

import argparse
from datetime import date, datetime
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from commodities_quant_engine.config.settings import settings
from commodities_quant_engine.data.ingestion import market_data_service
from commodities_quant_engine.data.storage.local import LocalStorage
from commodities_quant_engine.workflow import ResearchWorkflow

FEATURED_COMMODITIES = [
    "GOLD",
    "SILVER",
    "CRUDEOIL",
    "NATURALGAS",
    "COPPER",
    "ZINC",
    "LEAD",
]
LIVE_SIGNAL_DOMAIN = "live_signals"
LIVE_STATE_DOMAIN = "live_state"


def build_demo_price_data(periods: int = 280, end_date: Optional[str] = None) -> pd.DataFrame:
    """Create a deterministic demo dataset for a quick local smoke run."""
    end_timestamp = pd.Timestamp(end_date) if end_date else pd.Timestamp.today().normalize()
    index = pd.bdate_range(end=end_timestamp, periods=periods)
    base = np.linspace(0.0, 18.0, periods)
    seasonal = np.sin(np.linspace(0.0, 8.0 * np.pi, periods)) * 1.5
    close = 62000.0 + base * 20.0 + seasonal * 35.0
    frame = pd.DataFrame(
        {
            "open": close - 15.0,
            "high": close + 35.0,
            "low": close - 35.0,
            "close": close,
            "volume": np.linspace(18000, 26000, periods).astype(int),
            "open_interest": np.linspace(110000, 145000, periods).astype(int),
        },
        index=index,
    )
    frame.index.name = "timestamp"
    return frame


def load_price_data(
    commodity: str,
    price_file: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
    use_provider: bool,
    provider: Optional[str],
) -> pd.DataFrame:
    if price_file:
        path = Path(price_file)
        if not path.exists():
            raise FileNotFoundError(f"Price file not found: {price_file}")
        if path.suffix == ".parquet":
            frame = pd.read_parquet(path)
        else:
            frame = pd.read_csv(path)
        if "timestamp" not in frame.columns:
            raise ValueError("Price file must contain a 'timestamp' column.")
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        return frame.set_index("timestamp").sort_index()

    if use_provider or provider:
        if not start_date or not end_date:
            raise ValueError("--use-provider/--provider requires both --start-date and --end-date.")
        frame = market_data_service.load_price_frame(
            commodity=commodity,
            start_date=date.fromisoformat(start_date),
            end_date=date.fromisoformat(end_date),
            exchange=provider,
        )
        if frame.empty:
            raise ValueError("Provider returned no data. Check local catalog mappings or provider settings.")
        return frame

    return build_demo_price_data(end_date=end_date)


def try_load_price_data(
    commodity: str,
    price_file: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
    use_provider: bool,
    provider: Optional[str],
    allow_demo: bool,
) -> Tuple[Optional[pd.DataFrame], Optional[str], str]:
    if allow_demo:
        try:
            return load_price_data(
                commodity=commodity,
                price_file=price_file,
                start_date=start_date,
                end_date=end_date,
                use_provider=use_provider,
                provider=provider,
            ), None, "demo"
        except Exception as exc:
            return None, str(exc), "demo"

    try:
        frame = load_price_data(
            commodity=commodity,
            price_file=price_file,
            start_date=start_date,
            end_date=end_date,
            use_provider=use_provider,
            provider=provider,
        )
    except Exception as exc:
        return None, f"No data available for this commodity: {commodity}. {exc}", "provider"

    if frame is None or frame.empty:
        return None, f"No data available for this commodity: {commodity}.", "provider"
    return frame, None, "provider" if use_provider else "file"


def run_cycle(
    commodity: str,
    price_data: pd.DataFrame,
    storage: Optional[LocalStorage] = None,
    holdout_bars: int = 10,
    approve_adaptation: bool = False,
) -> Dict[str, Any]:
    workflow = ResearchWorkflow(storage=storage)
    holdout_bars = max(1, min(holdout_bars, len(price_data) - 1))
    signal_input = price_data.iloc[:-holdout_bars] if len(price_data) > holdout_bars else price_data
    signal_package = workflow.run_signal_cycle(
        commodity=commodity,
        price_data=signal_input,
        as_of_timestamp=signal_input.index[-1].to_pydatetime(),
    )
    evaluation = workflow.run_evaluation_cycle(
        commodity=commodity,
        price_data=price_data,
        as_of_timestamp=price_data.index[-1].to_pydatetime(),
    )
    adaptation = workflow.run_adaptation_cycle(
        commodity=commodity,
        dry_run=not approve_adaptation,
        approve=approve_adaptation,
    )
    return {
        "signal_package": signal_package,
        "evaluation": evaluation,
        "adaptation": adaptation,
    }


def run_signal_cycle_only(
    commodity: str,
    price_data: pd.DataFrame,
    storage: Optional[LocalStorage] = None,
) -> Dict[str, Any]:
    workflow = ResearchWorkflow(storage=storage)
    signal_package = workflow.run_signal_cycle(
        commodity=commodity,
        price_data=price_data,
        as_of_timestamp=price_data.index[-1].to_pydatetime(),
        persist_snapshot=False,
        persist_report=False,
    )
    return {
        "signal_package": signal_package,
        "workflow": workflow,
    }


def format_summary(results: Dict[str, Any]) -> str:
    suggestion = results["signal_package"].suggestion
    evaluation = results["evaluation"]
    adaptation = results["adaptation"]
    lines = [
        "Commodities Quant Engine Local Run",
        f"Commodity: {suggestion.commodity}",
        f"Signal ID: {suggestion.signal_id}",
        f"Timestamp: {suggestion.timestamp}",
        f"Category: {suggestion.final_category}",
        f"Direction: {suggestion.preferred_direction}",
        f"Confidence: {suggestion.confidence_score:.2f}",
        f"Composite Score: {suggestion.composite_score:.2f}",
        f"Evaluation Sample Size: {evaluation.summary_metrics.get('sample_size', 0)}",
        f"Evaluation Hit Rate: {evaluation.summary_metrics.get('overall_hit_rate', 0.0):.2%}",
        f"Evaluation Avg Return: {evaluation.summary_metrics.get('overall_average_return', 0.0):.4f}",
        f"Adaptation Mode: {adaptation.mode}",
        f"Adaptation Reason: {adaptation.reason}",
    ]
    if adaptation.candidate_version_id:
        lines.append(f"Candidate Version: {adaptation.candidate_version_id}")
    if evaluation.degradation_alerts:
        lines.append("Degradation Alerts:")
        lines.extend(f"- {alert}" for alert in evaluation.degradation_alerts)
    return "\n".join(lines)


def format_live_summary(entry: Dict[str, Any], poll_timestamp: datetime) -> str:
    package = entry["results"]["signal_package"]
    suggestion = package.suggestion
    lines = [
        "Commodities Quant Engine Live Run",
        f"Poll Timestamp: {poll_timestamp}",
        f"Commodity: {suggestion.commodity}",
        f"Market Timestamp: {suggestion.timestamp}",
        f"Category: {suggestion.final_category}",
        f"Direction: {suggestion.preferred_direction}",
        f"Confidence: {suggestion.confidence_score:.2f}",
        f"Composite Score: {suggestion.composite_score:.2f}",
        f"Snapshot Persisted: {'yes' if entry.get('snapshot_persisted') else 'no'}",
        f"Live Log Path: {entry.get('live_log_path', 'n/a')}",
    ]
    return "\n".join(lines)


def format_signal_row(commodity: str, results: Optional[Dict[str, Any]], status: str, message: Optional[str] = None) -> str:
    if results is None:
        return f"- {commodity}: {status}. {message or ''}".rstrip()
    suggestion = results["signal_package"].suggestion
    return (
        f"- {commodity}: {suggestion.final_category} | direction={suggestion.preferred_direction} "
        f"| confidence={suggestion.confidence_score:.2f} | score={suggestion.composite_score:.2f}"
    )


def format_live_signal_row(entry: Dict[str, Any]) -> str:
    if entry["results"] is None:
        return f"- {entry['commodity']}: skipped. {entry['message'] or ''}".rstrip()
    suggestion = entry["results"]["signal_package"].suggestion
    persisted = "stored" if entry.get("snapshot_persisted") else "observed"
    return (
        f"- {entry['commodity']}: {suggestion.final_category} | direction={suggestion.preferred_direction} "
        f"| confidence={suggestion.confidence_score:.2f} | score={suggestion.composite_score:.2f} "
        f"| market_ts={suggestion.timestamp} | {persisted}"
    )


def run_for_commodity(
    commodity: str,
    price_file: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
    use_provider: bool,
    provider: Optional[str],
    storage: Optional[LocalStorage],
    holdout_bars: int,
    approve_adaptation: bool,
    allow_demo: bool,
) -> Dict[str, Any]:
    price_data, error_message, data_mode = try_load_price_data(
        commodity=commodity,
        price_file=price_file,
        start_date=start_date,
        end_date=end_date,
        use_provider=use_provider,
        provider=provider,
        allow_demo=allow_demo,
    )
    if price_data is None:
        return {
            "commodity": commodity,
            "status": "skipped",
            "message": error_message or f"No data available for this commodity: {commodity}.",
            "data_mode": data_mode,
            "results": None,
        }

    results = run_cycle(
        commodity=commodity,
        price_data=price_data,
        storage=storage,
        holdout_bars=holdout_bars,
        approve_adaptation=approve_adaptation,
    )
    return {
        "commodity": commodity,
        "status": "ok",
        "message": None,
        "data_mode": data_mode,
        "results": results,
    }


def _live_state_payload(results: Dict[str, Any], poll_timestamp: datetime, data_mode: str) -> Dict[str, Any]:
    suggestion = results["signal_package"].suggestion
    return {
        "poll_timestamp": poll_timestamp.isoformat(),
        "market_timestamp": suggestion.timestamp.isoformat() if hasattr(suggestion.timestamp, "isoformat") else str(suggestion.timestamp),
        "direction": suggestion.preferred_direction,
        "category": suggestion.final_category,
        "confidence": round(float(suggestion.confidence_score), 4),
        "composite_score": round(float(suggestion.composite_score), 4),
        "signal_id": suggestion.signal_id,
        "data_mode": data_mode,
    }


def should_persist_live_snapshot(
    storage: LocalStorage,
    commodity: str,
    results: Dict[str, Any],
    poll_timestamp: datetime,
    data_mode: str,
) -> bool:
    current_state = _live_state_payload(results, poll_timestamp, data_mode)
    previous_state = storage.read_json(LIVE_STATE_DOMAIN, commodity)
    keys = ("market_timestamp", "direction", "category", "confidence", "composite_score", "data_mode")
    changed = any(previous_state.get(key) != current_state.get(key) for key in keys)
    if changed or not previous_state:
        storage.write_json(LIVE_STATE_DOMAIN, commodity, current_state)
        return True
    return False


def persist_live_observation(
    storage: LocalStorage,
    commodity: str,
    results: Dict[str, Any],
    poll_timestamp: datetime,
    data_mode: str,
    snapshot_persisted: bool,
) -> str:
    suggestion = results["signal_package"].suggestion
    snapshot = results["signal_package"].snapshot
    payload = {
        "poll_timestamp": poll_timestamp.isoformat(),
        "market_timestamp": suggestion.timestamp.isoformat() if hasattr(suggestion.timestamp, "isoformat") else str(suggestion.timestamp),
        "commodity": commodity,
        "signal_id": suggestion.signal_id,
        "active_contract": suggestion.active_contract,
        "category": suggestion.final_category,
        "direction": suggestion.preferred_direction,
        "confidence": round(float(suggestion.confidence_score), 6),
        "composite_score": round(float(suggestion.composite_score), 6),
        "regime_label": suggestion.regime_label,
        "suggested_horizon": int(suggestion.suggested_holding_horizon),
        "data_mode": data_mode,
        "snapshot_persisted": bool(snapshot_persisted),
        "model_version": snapshot.model_version,
        "config_version": snapshot.config_version,
    }
    path = storage.append_jsonl(LIVE_SIGNAL_DOMAIN, f"{commodity}_{poll_timestamp.strftime('%Y%m%d')}", [payload], compress=True)
    return str(path)


def run_live_for_commodity(
    commodity: str,
    price_file: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
    use_provider: bool,
    provider: Optional[str],
    storage: LocalStorage,
    allow_demo: bool,
    poll_timestamp: datetime,
) -> Dict[str, Any]:
    price_data, error_message, data_mode = try_load_price_data(
        commodity=commodity,
        price_file=price_file,
        start_date=start_date,
        end_date=end_date,
        use_provider=use_provider,
        provider=provider,
        allow_demo=allow_demo,
    )
    if price_data is None:
        return {
            "commodity": commodity,
            "status": "skipped",
            "message": error_message or f"No data available for this commodity: {commodity}.",
            "data_mode": data_mode,
            "results": None,
            "snapshot_persisted": False,
        }

    results = run_signal_cycle_only(commodity=commodity, price_data=price_data, storage=storage)
    snapshot_persisted = should_persist_live_snapshot(storage, commodity, results, poll_timestamp, data_mode)
    if snapshot_persisted:
        results["workflow"].evaluation_engine.persist_signal_snapshots([results["signal_package"].snapshot], commodity=commodity)
    live_log_path = persist_live_observation(storage, commodity, results, poll_timestamp, data_mode, snapshot_persisted)
    return {
        "commodity": commodity,
        "status": "ok",
        "message": None,
        "data_mode": data_mode,
        "results": results,
        "snapshot_persisted": snapshot_persisted,
        "live_log_path": live_log_path,
    }


def run_live_loop(
    commodities: List[str],
    price_file: Optional[str],
    start_date: Optional[str],
    end_date: Optional[str],
    use_provider: bool,
    provider: Optional[str],
    storage: Optional[LocalStorage],
    allow_demo: bool,
    show_markdown: bool,
    refresh_seconds: int,
    max_iterations: Optional[int],
) -> int:
    resolved_storage = storage or LocalStorage()
    iteration = 0
    try:
        while True:
            poll_started = time.time()
            poll_timestamp = datetime.now()
            entries = [
                run_live_for_commodity(
                    commodity=commodity,
                    price_file=price_file,
                    start_date=start_date,
                    end_date=end_date,
                    use_provider=use_provider,
                    provider=provider,
                    storage=resolved_storage,
                    allow_demo=allow_demo,
                    poll_timestamp=poll_timestamp,
                )
                for commodity in commodities
            ]

            if len(entries) > 1:
                print(f"Live Signal Poll {iteration + 1} | {poll_timestamp} | refresh={refresh_seconds}s")
                for entry in entries:
                    print(format_live_signal_row(entry))
            else:
                entry = entries[0]
                if entry["results"] is None:
                    print(entry["message"])
                else:
                    print(format_live_summary(entry, poll_timestamp))
                    if show_markdown and iteration == 0:
                        print()
                        print(entry["results"]["signal_package"].suggestion.to_markdown())

            iteration += 1
            if max_iterations is not None and iteration >= max_iterations:
                return 0
            elapsed = time.time() - poll_started
            time.sleep(max(0.0, refresh_seconds - elapsed))
    except KeyboardInterrupt:
        print("\nLive polling stopped.")
        return 0


def format_multi_commodity_report(entries: List[Dict[str, Any]], heading: str) -> str:
    lines = [heading]
    for entry in entries:
        lines.append(format_signal_row(entry["commodity"], entry["results"], entry["status"], entry["message"]))
    return "\n".join(lines)


def parse_commodity_selection(selection: str, available: List[str]) -> List[str]:
    normalized = selection.strip()
    if not normalized:
        raise ValueError("No selection provided.")

    if normalized.lower() in {"0", "all", "a", "*"}:
        return available

    chosen: List[str] = []
    tokens = [token for token in re.split(r"[\s,]+", normalized) if token]
    for token in tokens:
        if token.isdigit():
            index = int(token)
            if not 1 <= index <= len(available):
                raise ValueError(f"Selection index out of range: {token}")
            commodity = available[index - 1]
        else:
            commodity = token.upper()
            if commodity not in available:
                raise ValueError(f"Unknown commodity: {token}")
        if commodity not in chosen:
            chosen.append(commodity)

    if not chosen:
        raise ValueError("No valid commodities selected.")
    return chosen


def split_featured_and_other_commodities(available: List[str]) -> Tuple[List[str], List[str]]:
    featured = [commodity for commodity in FEATURED_COMMODITIES if commodity in available]
    featured_set = set(featured)
    others = [commodity for commodity in available if commodity not in featured_set]
    return featured, others


def parse_featured_selection(selection: str, featured: List[str], has_others: bool) -> Tuple[List[str], bool, bool]:
    normalized = selection.strip()
    if not normalized:
        raise ValueError("No selection provided.")

    if normalized.lower() in {"0", "all", "a", "*"}:
        return [], True, False

    chosen: List[str] = []
    wants_others = False
    others_index = len(featured) + 1 if has_others else None
    tokens = [token for token in re.split(r"[\s,]+", normalized) if token]
    for token in tokens:
        lowered = token.lower()
        if has_others and lowered in {"others", "other", "o"}:
            wants_others = True
            continue
        if token.isdigit():
            index = int(token)
            if has_others and others_index is not None and index == others_index:
                wants_others = True
                continue
            if not 1 <= index <= len(featured):
                raise ValueError(f"Selection index out of range: {token}")
            commodity = featured[index - 1]
        else:
            commodity = token.upper()
            if commodity not in featured:
                raise ValueError(f"Unknown featured commodity: {token}")
        if commodity not in chosen:
            chosen.append(commodity)

    if not chosen and not wants_others:
        raise ValueError("No valid commodities selected.")
    return chosen, False, wants_others


def prompt_for_other_commodities(others: List[str], input_func=input, output_func=print) -> List[str]:
    if not others:
        output_func("No additional commodities are configured beyond the featured list.")
        return []

    output_func("Other configured commodities:")
    output_func("0. All other commodities")
    for index, commodity in enumerate(others, start=1):
        output_func(f"{index}. {commodity}")
    output_func("Enter one or more values separated by commas or spaces, or type back to return to the featured list")

    while True:
        response = input_func("Others selection: ")
        normalized = response.strip().lower()
        if normalized in {"back", "b"}:
            return []
        try:
            return parse_commodity_selection(response, others)
        except ValueError as exc:
            output_func(f"Invalid selection. {exc}")


def prompt_for_commodities(available: List[str], input_func=input, output_func=print) -> List[str]:
    featured, others = split_featured_and_other_commodities(available)
    output_func("Select commodities to run:")
    output_func("0. All featured commodities (combine featured results together)")
    for index, commodity in enumerate(featured, start=1):
        output_func(f"{index}. {commodity}")
    if others:
        output_func(f"{len(featured) + 1}. Others")
    output_func("Enter one or more values separated by commas or spaces, for example: 1,3 or GOLD SILVER")

    while True:
        response = input_func("Selection: ")
        try:
            selected_featured, wants_all, wants_others = parse_featured_selection(response, featured, bool(others))
            if wants_all:
                return featured or available
            selected = list(selected_featured)
            if wants_others:
                selected_others = prompt_for_other_commodities(others, input_func=input_func, output_func=output_func)
                for commodity in selected_others:
                    if commodity not in selected:
                        selected.append(commodity)
            if selected:
                return selected
            output_func("Invalid selection. No valid commodities selected.")
        except ValueError as exc:
            output_func(f"Invalid selection. {exc}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a local commodities signal, evaluation, and adaptation cycle.")
    parser.add_argument("--commodity", choices=sorted(settings.commodities.keys()))
    parser.add_argument("--commodities", nargs="+", choices=sorted(settings.commodities.keys()), help="Run one or more selected commodities.")
    parser.add_argument("--price-file", help="CSV or parquet file with a timestamp column and OHLCV data.")
    parser.add_argument("--use-provider", action="store_true", help="Load market data from the configured local-first provider registry.")
    parser.add_argument("--provider", choices=sorted(settings.data_sources.keys()), help="Optional provider override, for example COMMODITIES_API for mapped reference data.")
    parser.add_argument("--all-commodities", action="store_true", help="Run the workflow for all configured commodities.")
    parser.add_argument("--list-commodities", action="store_true", help="List configured commodities and their latest signal summaries.")
    parser.add_argument("--start-date", help="ISO date for provider loading or demo end anchoring.")
    parser.add_argument("--end-date", help="ISO date for provider loading or demo end anchoring.")
    parser.add_argument("--holdout-bars", type=int, default=10, help="Bars reserved after signal issuance for evaluation.")
    parser.add_argument("--watch", action="store_true", help="Continuously refresh selected commodities instead of running once.")
    parser.add_argument("--refresh-seconds", type=int, default=30, help="Polling interval in seconds for live mode.")
    parser.add_argument("--max-iterations", type=int, help="Optional cap on the number of live refresh iterations.")
    parser.add_argument("--show-markdown", action="store_true", help="Print the full markdown suggestion after the summary.")
    parser.add_argument("--approve-adaptation", action="store_true", help="Allow the adaptive engine to promote a candidate if all safeguards pass.")
    parser.add_argument("--artifacts-dir", help="Optional override for artifact output directory.")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    raw_argv = list(argv) if argv is not None else sys.argv[1:]
    args = parser.parse_args(raw_argv)
    if args.commodity and args.commodities:
        parser.error("Use either --commodity or --commodities, not both.")
    if args.all_commodities and (args.commodity or args.commodities):
        parser.error("--all-commodities cannot be combined with --commodity or --commodities.")
    if args.provider and args.price_file:
        parser.error("--provider cannot be combined with --price-file.")
    if args.all_commodities and args.price_file:
        parser.error("--all-commodities cannot be used with --price-file. Use provider mode or run one commodity at a time.")
    if args.list_commodities and args.price_file:
        parser.error("--list-commodities cannot be used with --price-file. Use provider mode or list configured commodities without a file.")
    if args.refresh_seconds <= 0:
        parser.error("--refresh-seconds must be greater than 0.")
    if args.max_iterations is not None and args.max_iterations <= 0:
        parser.error("--max-iterations must be greater than 0.")
    storage = LocalStorage(args.artifacts_dir) if args.artifacts_dir else None
    no_explicit_mode = len(raw_argv) == 0
    interactive_selection = no_explicit_mode and sys.stdin.isatty() and not args.list_commodities
    default_list_mode = no_explicit_mode and not interactive_selection and args.commodity is None and not args.all_commodities and not args.list_commodities
    multi_mode = args.all_commodities or args.list_commodities or default_list_mode
    available_commodities = sorted(settings.commodities.keys())

    if interactive_selection:
        selected = prompt_for_commodities(available_commodities)
        commodities = selected
        multi_mode = len(commodities) > 1
    elif args.all_commodities:
        commodities = available_commodities
    elif args.commodities:
        commodities = list(dict.fromkeys(args.commodities))
        multi_mode = len(commodities) > 1
    elif args.commodity:
        commodities = [args.commodity]
        multi_mode = False
    elif multi_mode:
        commodities = available_commodities
    else:
        commodities = ["GOLD"]
    allow_demo = not args.use_provider and not args.provider and not args.price_file
    live_mode = bool(args.watch or interactive_selection)

    if live_mode and not args.list_commodities:
        return run_live_loop(
            commodities=commodities,
            price_file=args.price_file,
            start_date=args.start_date,
            end_date=args.end_date,
            use_provider=args.use_provider,
            provider=args.provider,
            storage=storage,
            allow_demo=allow_demo,
            show_markdown=args.show_markdown,
            refresh_seconds=args.refresh_seconds,
            max_iterations=args.max_iterations,
        )

    entries = [
        run_for_commodity(
            commodity=commodity,
            price_file=args.price_file,
            start_date=args.start_date,
            end_date=args.end_date,
            use_provider=args.use_provider,
            provider=args.provider,
            storage=storage,
            holdout_bars=args.holdout_bars,
            approve_adaptation=args.approve_adaptation,
            allow_demo=allow_demo,
        )
        for commodity in commodities
    ]

    if multi_mode:
        heading = "Configured Commodities And Latest Signals" if (args.list_commodities or default_list_mode) else "All Commodity Runs"
        print(format_multi_commodity_report(entries, heading))
        if args.show_markdown:
            for entry in entries:
                if entry["results"] is not None:
                    print()
                    print(entry["results"]["signal_package"].suggestion.to_markdown())
        return 0

    entry = entries[0]
    if entry["results"] is None:
        print(entry["message"])
        return 0

    results = entry["results"]
    print(format_summary(results))
    if args.show_markdown:
        print()
        print(results["signal_package"].suggestion.to_markdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
