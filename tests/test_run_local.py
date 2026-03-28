from ..run_local import (
    build_demo_price_data,
    build_parser,
    format_multi_commodity_report,
    format_summary,
    parse_commodity_selection,
    parse_featured_selection,
    persist_live_observation,
    prompt_for_commodities,
    run_cycle,
    run_for_commodity,
    run_signal_cycle_only,
    should_persist_live_snapshot,
    split_featured_and_other_commodities,
)
from ..data.storage.local import LocalStorage


def test_run_local_demo_cycle(tmp_path):
    storage = LocalStorage(str(tmp_path))
    price_data = build_demo_price_data(periods=120, end_date="2025-12-31")
    results = run_cycle(
        commodity="GOLD",
        price_data=price_data,
        storage=storage,
        holdout_bars=10,
        approve_adaptation=False,
    )

    summary = format_summary(results)
    assert "Commodity: GOLD" in summary
    assert "Directional Term Structure:" in summary
    assert "Feature Highlights:" in summary
    assert results["signal_package"].suggestion.signal_id
    assert results["evaluation"].summary_metrics["sample_size"] >= 1


def test_run_for_commodity_skips_missing_provider_data(tmp_path):
    storage = LocalStorage(str(tmp_path))
    entry = run_for_commodity(
        commodity="GOLD",
        price_file=None,
        start_date="2025-01-01",
        end_date="2025-03-01",
        use_provider=True,
        provider=None,
        storage=storage,
        holdout_bars=10,
        approve_adaptation=False,
        allow_demo=False,
    )
    assert entry["results"] is None
    assert entry["status"] == "skipped"
    assert "No data available for this commodity: GOLD" in entry["message"]


def test_format_multi_commodity_report_includes_signal_and_skip_lines(tmp_path):
    storage = LocalStorage(str(tmp_path))
    results = run_cycle(
        commodity="GOLD",
        price_data=build_demo_price_data(periods=120, end_date="2025-12-31"),
        storage=storage,
        holdout_bars=10,
        approve_adaptation=False,
    )
    report = format_multi_commodity_report(
        [
            {"commodity": "GOLD", "results": results, "status": "ok", "message": None},
            {"commodity": "COPPER", "results": None, "status": "skipped", "message": "No data available for this commodity: COPPER."},
        ],
        "Configured Commodities And Latest Signals",
    )
    assert "Configured Commodities And Latest Signals" in report
    assert "GOLD:" in report
    assert "COPPER: skipped." in report


def test_parser_defaults_allow_multi_commodity_launcher_mode():
    parser = build_parser()
    args = parser.parse_args([])
    assert args.commodity is None
    assert args.commodities is None
    assert not args.all_commodities
    assert not args.list_commodities


def test_parser_supports_multiple_selected_commodities():
    parser = build_parser()
    args = parser.parse_args(["--commodities", "GOLD", "SILVER"])
    assert args.commodities == ["GOLD", "SILVER"]


def test_parser_supports_provider_override():
    parser = build_parser()
    args = parser.parse_args(["--commodity", "GOLD", "--provider", "COMMODITIES_API", "--start-date", "2025-01-01", "--end-date", "2025-01-10"])
    assert args.provider == "COMMODITIES_API"


def test_parser_supports_live_refresh_options():
    parser = build_parser()
    args = parser.parse_args(["--commodity", "GOLD", "--watch", "--refresh-seconds", "30", "--max-iterations", "2"])
    assert args.watch
    assert args.refresh_seconds == 30
    assert args.max_iterations == 2


def test_parse_commodity_selection_supports_all_and_numeric_choices():
    available = ["COPPER", "CRUDEOIL", "GOLD", "SILVER"]
    assert parse_commodity_selection("all", available) == available
    assert parse_commodity_selection("1,3", available) == ["COPPER", "GOLD"]
    assert parse_commodity_selection("gold silver", available) == ["GOLD", "SILVER"]


def test_split_featured_and_other_commodities_prioritizes_liquid_contracts():
    featured, others = split_featured_and_other_commodities(["ALUMINIUM", "COPPER", "GOLD", "LEAD", "SILVER", "TIN"])

    assert featured == ["GOLD", "SILVER", "COPPER", "LEAD"]
    assert others == ["ALUMINIUM", "TIN"]


def test_parse_featured_selection_supports_others_branch():
    selected, wants_all, wants_others = parse_featured_selection("1 others", ["GOLD", "SILVER"], has_others=True)

    assert selected == ["GOLD"]
    assert not wants_all
    assert wants_others


def test_prompt_for_commodities_all_selects_featured_only():
    messages = []

    def fake_output(message):
        messages.append(message)

    selected = prompt_for_commodities(
        ["ALUMINIUM", "GOLD", "SILVER", "TIN"],
        input_func=lambda prompt: "0",
        output_func=fake_output,
    )

    assert selected == ["GOLD", "SILVER"]
    assert any("All featured commodities (combine featured results together)" in message for message in messages)


def test_prompt_for_commodities_retries_after_invalid_input():
    responses = iter(["invalid", "2"])
    messages = []

    def fake_input(prompt):
        return next(responses)

    def fake_output(message):
        messages.append(message)

    selected = prompt_for_commodities(["GOLD", "SILVER"], input_func=fake_input, output_func=fake_output)
    assert selected == ["SILVER"]
    assert any("All featured commodities (combine featured results together)" in message for message in messages)
    assert any("Invalid selection." in message for message in messages)


def test_prompt_for_commodities_supports_featured_plus_others():
    responses = iter(["1 2", "2"])

    def fake_input(prompt):
        return next(responses)

    selected = prompt_for_commodities(["ALUMINIUM", "CARDAMOM", "GOLD"], input_func=fake_input, output_func=lambda _: None)
    assert selected == ["GOLD", "CARDAMOM"]


def test_live_snapshot_persistence_skips_unchanged_state(tmp_path):
    storage = LocalStorage(str(tmp_path))
    results = run_signal_cycle_only(
        commodity="GOLD",
        price_data=build_demo_price_data(periods=120, end_date="2025-12-31"),
        storage=storage,
    )
    poll_timestamp = results["signal_package"].suggestion.timestamp

    first = should_persist_live_snapshot(storage, "GOLD", results, poll_timestamp, "demo")
    second = should_persist_live_snapshot(storage, "GOLD", results, poll_timestamp, "demo")

    assert first
    assert not second


def test_live_observation_is_appended_to_compressed_log(tmp_path):
    storage = LocalStorage(str(tmp_path))
    results = run_signal_cycle_only(
        commodity="GOLD",
        price_data=build_demo_price_data(periods=120, end_date="2025-12-31"),
        storage=storage,
    )
    poll_timestamp = results["signal_package"].suggestion.timestamp
    path = persist_live_observation(
        storage=storage,
        commodity="GOLD",
        results=results,
        poll_timestamp=poll_timestamp,
        data_mode="demo",
        snapshot_persisted=True,
    )

    records = storage.load_jsonl("live_signals", f"GOLD_{poll_timestamp.strftime('%Y%m%d')}")
    assert path.endswith(".jsonl.gz")
    assert len(records) == 1
    assert records[0]["commodity"] == "GOLD"
    assert records[0]["exchange"]
    assert "directional_scores" in records[0]
    assert "component_scores" in records[0]
    assert "feature_highlights" in records[0]
