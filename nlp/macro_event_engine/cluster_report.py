from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from ...config.settings import settings
from ...data.storage.local import LocalStorage


class ClusterReportGenerator:
    """Writes per-signal cluster membership and representative-selection rationale to artifacts/reports."""

    def generate(
        self,
        cluster_manifest: List[Dict[str, Any]],
        diagnostics: Dict[str, Any],
        commodity: str,
        signal_id: str,
        as_of_timestamp: datetime,
        storage: LocalStorage,
    ) -> Dict[str, Path]:
        """Generate a JSON + Markdown cluster report. Returns {"json": Path, "markdown": Path}."""
        json_payload: Dict[str, Any] = {
            "signal_id": signal_id,
            "commodity": commodity,
            "generated_at": as_of_timestamp.isoformat(),
            "raw_event_count": int(diagnostics.get("raw_event_count", 0)),
            "cluster_count": int(diagnostics.get("cluster_count", 0)),
            "dedup_ratio": float(diagnostics.get("dedup_ratio", 0.0)),
            "clusters": cluster_manifest,
        }

        report_name = f"{commodity}_{signal_id}_cluster_report"
        json_path = storage.write_json(settings.storage.report_store, report_name, json_payload)

        md_path = storage.resolve(settings.storage.report_store, f"{report_name}.md")
        md_path.write_text(self._render_markdown(json_payload), encoding="utf-8")

        return {"json": json_path, "markdown": md_path}

    # ------------------------------------------------------------------
    # Internal rendering
    # ------------------------------------------------------------------

    def _render_markdown(self, payload: Dict[str, Any]) -> str:
        raw_count = payload["raw_event_count"]
        cluster_count = payload["cluster_count"]
        dedup_pct = payload["dedup_ratio"] * 100.0

        lines: List[str] = [
            "# Event Cluster Inspection Report",
            "",
            f"**Signal ID:** {payload['signal_id']}  ",
            f"**Commodity:** {payload['commodity']}  ",
            f"**Generated:** {payload['generated_at']}  ",
            f"**Raw events:** {raw_count} | "
            f"**Clusters:** {cluster_count} | "
            f"**Dedup ratio:** {dedup_pct:.1f}%",
            "",
            "---",
        ]

        clusters = payload.get("clusters", [])
        if not clusters:
            lines += ["", "_No event clusters were produced for this signal._", ""]
            return "\n".join(lines)

        for cluster in clusters:
            cid = cluster.get("cluster_id", "???")
            etype = cluster.get("event_type", "unknown")
            size = cluster.get("cluster_size", 1)
            scale = cluster.get("dedup_scale", 1.0)
            rep_summary = cluster.get("representative_summary", "")
            rep_conf = cluster.get("representative_confidence_raw", 0.0)
            rep_strength = cluster.get("representative_event_strength_raw", 0.0)
            rationale = cluster.get("representative_rationale", "")
            max_jaccard = cluster.get("max_intra_jaccard", 1.0)
            members: List[Dict[str, Any]] = cluster.get("members", [])

            noun = "member" if size == 1 else "members"
            lines += [
                "",
                f"## {cid} \u2014 `{etype}` ({size} {noun})",
                "",
                f"**Representative:** {rep_summary}  ",
                f"**Rationale:** {rationale}  ",
                (
                    f"**Dedup scale:** {scale:.4f} \u00b7 "
                    f"**Confidence (raw):** {rep_conf:.3f} \u00b7 "
                    f"**Event strength (raw):** {rep_strength:.3f} \u00b7 "
                    f"**Max intra-cluster Jaccard:** {max_jaccard:.3f}"
                ),
                "",
            ]

            if members:
                lines.append("| # | Rep | Source ID | Summary | Confidence | Join Jaccard |")
                lines.append("|---|-----|-----------|---------|------------|--------------|")
                for i, member in enumerate(members, start=1):
                    rep_marker = "\u2713" if member.get("is_representative") else ""
                    src = member.get("source_id") or "\u2014"
                    summ = (member.get("summary") or "")[:90].replace("|", "\\|")
                    conf = member.get("confidence", 0.0)
                    jac = member.get("join_jaccard", 1.0)
                    lines.append(f"| {i} | {rep_marker} | `{src}` | {summ} | {conf:.3f} | {jac:.3f} |")
                lines.append("")

            lines.append("---")

        return "\n".join(lines)
