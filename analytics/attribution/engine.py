from __future__ import annotations

from typing import Dict, Iterable

from ...data.models import SignalSnapshot


class AttributionEngine:
    """Simple component attribution summaries for evaluated signal snapshots."""

    def summarize_component_attribution(
        self,
        snapshots: Iterable[SignalSnapshot],
    ) -> Dict[str, object]:
        snapshots = list(snapshots)
        if not snapshots:
            return {
                "sample_size": 0,
                "component_mean": {},
                "component_abs_mean": {},
                "top_component": None,
            }

        component_sum: Dict[str, float] = {}
        component_abs_sum: Dict[str, float] = {}
        for snapshot in snapshots:
            for component, value in snapshot.component_scores.items():
                value_f = float(value)
                component_sum[component] = component_sum.get(component, 0.0) + value_f
                component_abs_sum[component] = component_abs_sum.get(component, 0.0) + abs(value_f)

        sample_size = len(snapshots)
        component_mean = {k: v / sample_size for k, v in component_sum.items()}
        component_abs_mean = {k: v / sample_size for k, v in component_abs_sum.items()}
        top_component = None
        if component_abs_mean:
            top_component = max(component_abs_mean.items(), key=lambda kv: kv[1])[0]

        return {
            "sample_size": sample_size,
            "component_mean": component_mean,
            "component_abs_mean": component_abs_mean,
            "top_component": top_component,
        }
