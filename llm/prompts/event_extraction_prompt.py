EVENT_EXTRACTION_SYSTEM_PROMPT = """
You are a commodity event extraction engine.
Return strict JSON only.
Do not return free-form analysis.
Required fields:
- event_type
- commodity_scope
- asset_scope
- expected_direction
- confidence
- persistence_horizon
- event_strength
- uncertainty_score
- regime_relevance
- supply_demand_axis
- volatility_implication
- summary
- entities_keywords
""".strip()


def build_event_extraction_user_prompt(raw_text: str, commodity_scope: list[str], source_id: str) -> str:
	scope = ", ".join(commodity_scope) if commodity_scope else "UNKNOWN"
	return (
		"Extract one normalized commodity event JSON from the text below.\n"
		"Use only these enums where applicable:\n"
		"event_type: supply_disruption, supply_recovery, demand_strength, demand_weakness, "
		"inventory_buildup, inventory_drawdown, weather_risk, policy_supportive, policy_negative, "
		"sanctions/geopolitics, shipping/logistics_issue, producer_guidance_change, "
		"currency_macro_shift, rates_macro_shift, inflation_macro_shift, industrial_activity_signal, unknown\n"
		"asset_scope: single_commodity, sector_basket, macro_wide\n"
		"expected_direction: bullish, bearish, neutral, mixed\n"
		"persistence_horizon: very_short, short, medium, long\n"
		"supply_demand_axis: supply, demand, macro, mixed\n"
		"volatility_implication: lower, unchanged, higher\n"
		"All numeric scores must be in [0, 1].\n"
		f"commodity_scope default: [{scope}]\n"
		f"source_id default: {source_id}\n"
		"Text:\n"
		f"{raw_text}"
	)
