from __future__ import annotations

from typing import Dict

from ...data.models import EvaluationArtifact


class DiagnosticsEngine:
    """Evaluation-health diagnostics with simple status thresholds."""

    def summarize_evaluation_health(self, artifact: EvaluationArtifact) -> Dict[str, object]:
        sample_size = int(artifact.summary_metrics.get("sample_size", 0) or 0)
        avg_hit_rate = float(artifact.summary_metrics.get("average_hit_rate", 0.0) or 0.0)
        degradation_alerts = list(artifact.degradation_alerts or [])
        alert_count = len(degradation_alerts)

        if sample_size < 20:
            status = "insufficient_data"
        elif alert_count >= 2 or avg_hit_rate < 0.45:
            status = "red"
        elif alert_count >= 1 or avg_hit_rate < 0.52:
            status = "yellow"
        else:
            status = "green"

        return {
            "commodity": artifact.commodity,
            "sample_size": sample_size,
            "average_hit_rate": avg_hit_rate,
            "degradation_alert_count": alert_count,
            "degradation_alerts": degradation_alerts,
            "status": status,
        }
