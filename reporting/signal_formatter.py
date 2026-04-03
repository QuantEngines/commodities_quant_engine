"""
Signal output formatting with structured tables and improved readability.
Replaces scattered formatting logic from main.py.
"""

from typing import Any, Dict, List, Optional, Tuple
import pandas as pd


def format_optional_number(value: Optional[float], digits: int = 2, signed: bool = False) -> str:
    """Format optional float with sign and digit control."""
    if value is None:
        return "—"
    format_spec = f"{'+' if signed else ''}.{digits}f"
    return format(float(value), format_spec)


def format_optional_pct(value: Optional[float], digits: int = 2) -> str:
    """Format optional float as percentage."""
    if value is None:
        return "—"
    return f"{float(value):.{digits}%}"


def compact_list(items: List[str], limit: int = 3) -> str:
    """Compact comma-separated list with limit."""
    filtered = [str(item).strip() for item in items if str(item).strip()]
    if not filtered:
        return "None"
    if len(filtered) <= limit:
        return "; ".join(filtered)
    return "; ".join(filtered[:limit]) + f" (+{len(filtered) - limit})"


def top_abs_items(values: Dict[str, Any], limit: int = 4) -> List[Tuple[str, float]]:
    """Get top N items by absolute value."""
    normalized: List[Tuple[str, float]] = []
    for key, value in values.items():
        try:
            normalized.append((str(key), float(value)))
        except (TypeError, ValueError):
            continue
    normalized.sort(key=lambda item: abs(item[1]), reverse=True)
    return normalized[:limit]


def format_score_table(
    scores: Dict[str, Any],
    title: str = "Component Scores",
    limit: int = 6,
) -> str:
    """Format scores as a clean ASCII table."""
    if not scores:
        return f"{title}: None"
    
    items = top_abs_items(scores, limit=limit)
    if not items:
        return f"{title}: None"
    
    max_name_len = max(len(name) for name, _ in items)
    lines = [f"\n{title}:"]
    lines.append("├─ " + "─" * (max_name_len + 12))
    
    for name, value in items:
        sign = "+" if value > 0 else "−"
        bar_len = min(25, max(1, int(abs(value) * 25)))
        bar = "█" * bar_len if value > 0 else "░" * bar_len
        lines.append(f"│ {name:<{max_name_len}} {sign} {value:>7.3f} {bar}")
    
    lines.append("└─" + "─" * (max_name_len + 12))
    return "\n".join(lines)


def format_metrics_table(
    metrics: Dict[str, Any],
    title: str = "Metrics",
) -> str:
    """Format metrics as a two-column table."""
    if not metrics:
        return ""
    
    lines = [f"\n{title}:"]
    max_key_len = max(len(str(k)) for k in metrics.keys())
    
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted_value = f"{value:.4f}"
        else:
            formatted_value = str(value)
        lines.append(f"  {str(key):<{max_key_len}} : {formatted_value}")
    
    return "\n".join(lines)


def format_decision_box(suggestion: Any) -> str:
    """Format the main decision box."""
    lines = [
        "╔════════════════════════════════════════════════════════════╗",
        f"║ {suggestion.commodity:^58} ║",
        f"║ Signal ID: {(suggestion.signal_id or 'n/a'):<47} ║",
        "╠════════════════════════════════════════════════════════════╣",
        f"║ Decision:  {suggestion.final_category:<50} ║",
        f"║ Direction: {suggestion.preferred_direction:<50} ║",
        f"║ Confidence: {format_optional_number(suggestion.confidence_score, digits=3):<48} ║",
        f"║ Composite Score: {format_optional_number(suggestion.composite_score, digits=3):<42} ║",
        "╠════════════════════════════════════════════════════════════╣",
        f"║ Regime: {suggestion.regime_label:<52} ║",
        f"║ Horizon: {suggestion.suggested_holding_horizon}D | Entry: {suggestion.suggested_entry_style:<34} ║",
        f"║ Data Quality: {suggestion.data_quality_flag:<46} ║",
        "╚════════════════════════════════════════════════════════════╝",
    ]
    return "\n".join(lines)


def format_signal_summary(suggestion: Any, evaluation: Any = None) -> str:
    """Create comprehensive signal summary with tables."""
    diagnostics = suggestion.diagnostics or {}
    component_scores = diagnostics.get("component_scores", {})
    directional_scores = suggestion.directional_scores or {}
    directional_confidences = diagnostics.get("directional_confidences", {})
    feature_vector = diagnostics.get("feature_vector", {})
    event_features = diagnostics.get("event_intelligence_features", {})
    
    output = []
    
    # Decision box
    output.append(format_decision_box(suggestion))
    
    # Key metrics
    output.append(format_metrics_table({
        "Timestamp": str(suggestion.timestamp),
        "Exchange": suggestion.exchange,
        "Contract": suggestion.active_contract,
        "Model Version": suggestion.model_version or "default",
        "Config Version": suggestion.config_version or "default",
    }, "Signal Metadata"))
    
    # Directional structure
    if directional_scores:
        dir_items = [(str(h), float(v)) for h, v in directional_scores.items()]
        dir_items.sort(key=lambda x: x[0])
        output.append(format_metrics_table(
            {f"{h}D": f"{v:.3f}" for h, v in dir_items},
            "Directional Term Structure"
        ))
    
    # Component scores table
    if component_scores:
        output.append(format_score_table(component_scores, "Component Breakdown", limit=8))
    
    # Regime and macro context
    regime_info = {
        "Label": suggestion.regime_label,
        "Macro Alignment": format_optional_number(suggestion.macro_alignment_score),
        "Macro Conflict": format_optional_number(suggestion.macro_conflict_score),
        "Macro Event Risk": "High" if suggestion.macro_event_risk_flag else "Low",
        "Confidence Adj.": format_optional_number(suggestion.macro_confidence_adjustment, signed=True),
    }
    output.append(format_metrics_table(regime_info, "Regime Context"))
    
    # Shipping context (if available)
    if suggestion.shipping_summary:
        shipping_info = {
            "Summary": suggestion.shipping_summary,
            "Alignment": format_optional_number(suggestion.shipping_alignment_score),
            "Conflict": format_optional_number(suggestion.shipping_conflict_score),
            "Risk Penalty": format_optional_number(suggestion.shipping_risk_penalty),
            "Data Quality": format_optional_number(suggestion.shipping_data_quality_score),
        }
        output.append(format_metrics_table(shipping_info, "Shipping Intelligence"))
    
    # Top features
    if feature_vector:
        top_features = top_abs_items(feature_vector, limit=10)
        if top_features:
            output.append("\nTop Features (by magnitude):")
            for name, value in top_features:
                sign = "↑" if value > 0 else "↓"
                output.append(f"  {sign} {name:<30} {value:>10.4f}")
    
    # Drivers and risks
    output.append("\nSignal Drivers & Risks:")
    output.append(f"  Supporting: {compact_list(suggestion.key_supporting_drivers, limit=4)}")
    output.append(f"  Contradictions: {compact_list(suggestion.key_contradictory_drivers, limit=3)}")
    output.append(f"  Principal Risks: {compact_list(suggestion.principal_risks, limit=4)}")
    
    # Macro features
    if suggestion.key_macro_drivers:
        output.append(f"  Macro Drivers: {compact_list(suggestion.key_macro_drivers, limit=3)}")
    if suggestion.key_macro_risks:
        output.append(f"  Macro Risks: {compact_list(suggestion.key_macro_risks, limit=3)}")
    
    # Shipping features (if available)
    if suggestion.key_shipping_drivers:
        output.append(f"  Shipping Drivers: {compact_list(suggestion.key_shipping_drivers, limit=3)}")
    
    # Evaluation metrics (if available)
    if evaluation:
        eval_summary = evaluation.summary_metrics or {}
        eval_info = {
            "Sample Size": int(eval_summary.get("sample_size", 0)),
            "Hit Rate": format_optional_pct(eval_summary.get("overall_hit_rate")),
            "Avg Return": format_optional_number(eval_summary.get("overall_average_return"), digits=4),
            "Rank IC": format_optional_number(eval_summary.get("overall_rank_ic"), digits=2),
            "Brier Score": format_optional_number(eval_summary.get("overall_brier_score"), digits=3),
        }
        output.append(format_metrics_table(eval_info, "Evaluation Performance"))
        
        if evaluation.degradation_alerts:
            output.append(f"\n⚠ Degradation Alerts ({len(evaluation.degradation_alerts)}):")
            for alert in evaluation.degradation_alerts[:5]:
                output.append(f"  • {alert}")
    
    # Explanation
    if suggestion.explanation_summary:
        output.append(f"\nThesis:\n  {suggestion.explanation_summary}")
    
    return "\n".join(output)


def format_multi_commodity_table(entries: List[Dict[str, Any]]) -> str:
    """Format multiple commodities as a table."""
    if not entries:
        return "No results"
    
    lines = [
        "╔════════════════════════════════════════════════════════════════════════════════════╗",
        "║ Commodity Results Summary                                                          ║",
        "╠════════════════════════════════════════════════════════════════════════════════════╣",
    ]
    
    for entry in entries:
        if entry["results"] is None:
            lines.append(f"║ {entry['commodity']:<10} │ ✗ {entry.get('message', 'Skipped'):<65} ║")
        else:
            suggestion = entry["results"]["signal_package"].suggestion
            status = f"✓ {suggestion.final_category:<5} │ " \
                     f"{suggestion.preferred_direction:<4} │ " \
                     f"Conf: {suggestion.confidence_score:.2f}"
            lines.append(f"║ {entry['commodity']:<10} │ {status:<68} ║")
    
    lines.append("╚════════════════════════════════════════════════════════════════════════════════════╝")
    return "\n".join(lines)
