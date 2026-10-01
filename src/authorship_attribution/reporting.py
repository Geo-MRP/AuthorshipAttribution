"""Privacy-safe comparison report renderers."""

from __future__ import annotations

import json
from typing import Literal

from .contracts import ComparisonReport
from .fingerprints import canonical_json


def _score(value: float | None) -> str:
    return "—" if value is None else f"{value:.6f}"


def render_compare_report(
    report: ComparisonReport, *, output_format: Literal["table", "json"]
) -> str:
    """Render independent engine fields and the mandatory first-line banner."""
    report._validate()
    def clean_error(code: str | None) -> str | None:
        allowed = {"MINIMUMS_NOT_MET", "ENGINE_TIMEOUT", "ENGINE_FAILURE", "BASELINE_CONFIG_MISMATCH", "BASELINE_INVALID", "LOCAL_MODEL_UNAVAILABLE", "LOCAL_MODEL_INVALID", "EXTERNAL_NOT_ALLOWED", "INPUT_INVALID", "ARTIFACT_INVALID"}
        return code if code in allowed else ("ENGINE_FAILURE" if code is not None else None)

    if clean_error(report.stylometry.error) != report.stylometry.error or clean_error(report.semantic.error) != report.semantic.error:
        # Rebuild contract-validated records so report serialization cannot
        # accidentally disclose an engine exception's arbitrary message.
        from dataclasses import replace
        stylo = replace(report.stylometry, error=clean_error(report.stylometry.error))
        semantic = replace(report.semantic, error=clean_error(report.semantic.error))
        report = replace(report, stylometry=stylo, semantic=semantic)
    if output_format == "json":
        payload = canonical_json(report).decode("utf-8")
        return f"{report.banner}\n{payload}"
    rows = [("Field", "Semantic", "Stylometry")]
    for label, field in (
        ("Raw cosine", "raw_cosine"),
        ("Known-different baseline percentile", "calibrated_percentile"),
        ("Status", "status"),
    ):
        semantic = getattr(report.semantic, field)
        stylometry = getattr(report.stylometry, field)
        if field in {"raw_cosine", "calibrated_percentile"}:
            semantic = _score(semantic)
            stylometry = _score(stylometry)
        rows.append((label, str(semantic), str(stylometry)))
    rows.append(("Candidate messages", str(report.semantic.n_messages["candidate"]), str(report.stylometry.n_messages["candidate"])))
    rows.append(("Suspect messages", str(report.semantic.n_messages["suspect"]), str(report.stylometry.n_messages["suspect"])))
    if report.semantic.error is not None or report.stylometry.error is not None:
        rows.append(("Sanitized error", str(report.semantic.error or "—"), str(report.stylometry.error or "—")))

    widths = [max(len(row[column]) for row in rows) for column in range(3)]
    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    table_lines = [border]
    table_lines.extend(
        "| " + " | ".join(value.ljust(width) for value, width in zip(row, widths)) + " |"
        for row in rows
    )
    table_lines.append(border)
    lines = [report.banner, f"Flagged account: {report.flagged_account_id} | Suspect account: {report.suspect_account_id}", *table_lines]
    return "\n".join(lines)


def render_evaluation_report(report: object, *, output_format: Literal["table", "json"]) -> str:
    """Render two independent evaluation summaries without a merged metric."""
    if output_format == "json":
        return canonical_json(report).decode("utf-8")
    stylometry = report.stylometry
    semantic = report.semantic
    lines = [report.note, "Engine      ROC-AUC  Precision  Recall  Support(same/different)  Excluded  Threshold percentile  Precision constraint"]
    for item in (semantic, stylometry):
        auc = "—" if item.roc_auc is None else f"{item.roc_auc:.4f}"
        lines.append(
            f"{item.engine:<11} {auc:<7} {item.precision:.4f}     {item.recall:.4f}  "
            f"{item.support_same}/{item.support_different:<21} {item.excluded_insufficient_data:<8} "
            f"{item.threshold_percentile:.2f}                 {'met' if item.precision_constraint_met else 'unmet'}"
        )
    return "\n".join(lines)


__all__ = ["render_compare_report", "render_evaluation_report"]
