"""Privacy-safe three-engine and heuristic-fusion report renderers."""

from __future__ import annotations

import json
from typing import Literal

from .contracts import ComparisonReport, EvaluationReport
from .errors import ContractValidationError, ErrorCode
from .fingerprints import canonical_json


def _score(value: float | None) -> str:
    return "—" if value is None else f"{value:.6f}"


def render_compare_report(
    report: ComparisonReport, *, output_format: Literal["table", "json"]
) -> str:
    """Render independent engine values and the mandatory review banner."""
    if not isinstance(report, ComparisonReport):
        raise ContractValidationError("report", ErrorCode.INPUT_INVALID)
    report._validate()
    if output_format not in ("table", "json"):
        raise ContractValidationError("output_format", ErrorCode.INPUT_INVALID)
    if output_format == "json":
        return f"{report.banner}\n{canonical_json(report).decode('utf-8')}"

    engines = (report.stylometry, report.semantic, report.chronobiology)
    combined = report.combined
    rows = [("Field", "Stylometry", "Semantic", "Chronobiology", "Combined")]
    rows.append(("Status", *(item.status for item in engines), combined.status))
    rows.append(("Raw cosine", *(_score(item.raw_cosine) for item in engines), "—"))
    rows.append((
        "Known-different baseline percentile",
        *(_score(item.calibrated_percentile) for item in engines), "—",
    ))
    rows.append(("Candidate messages", *(str(item.n_messages["candidate"]) for item in engines), "—"))
    rows.append(("Suspect messages", *(str(item.n_messages["suspect"]) for item in engines), "—"))
    rows.append(("Candidate characters", *(str(item.n_chars["candidate"]) for item in engines), "—"))
    rows.append(("Suspect characters", *(str(item.n_chars["suspect"]) for item in engines), "—"))
    rows.append(("Sanitized error", *(item.error or "—" for item in engines), combined.error or "—"))
    rows.append(("Combined score (heuristic)", "—", "—", "—", _score(combined.combined_score)))
    rows.append(("Weighted percentile", "—", "—", "—", _score(combined.weighted_percentile)))
    rows.append(("Fisher tail score (heuristic)", "—", "—", "—", _score(combined.fisher_tail_score)))
    rows.append(("Active engines", "—", "—", "—", ", ".join(combined.active_engines) or "—"))
    weights = json.dumps(combined.weights_used, sort_keys=True, separators=(",", ":"))
    rows.append(("Weights used", "—", "—", "—", weights))

    widths = [max(len(row[column]) for row in rows) for column in range(5)]
    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    table_lines = [border]
    table_lines.extend(
        "| " + " | ".join(value.ljust(width) for value, width in zip(row, widths, strict=True)) + " |"
        for row in rows
    )
    table_lines.append(border)
    flagged = json.dumps(report.flagged_account_id, ensure_ascii=False)
    suspect = json.dumps(report.suspect_account_id, ensure_ascii=False)
    return "\n".join(
        [
            report.banner,
            f"Flagged account: {flagged} | Suspect account: {suspect}",
            *table_lines,
            "Combined score is a review-prioritization heuristic, not a probability.",
        ]
    )


def render_evaluation_report(
    report: EvaluationReport, *, output_format: Literal["table", "json"]
) -> str:
    """Render three engine evaluations and the combined heuristic evaluation."""
    if not isinstance(report, EvaluationReport):
        raise ContractValidationError("report", ErrorCode.INPUT_INVALID)
    report._validate()
    if output_format not in ("table", "json"):
        raise ContractValidationError("output_format", ErrorCode.INPUT_INVALID)
    if output_format == "json":
        return canonical_json(report).decode("utf-8")

    lines = [
        report.note,
        "Engine         ROC-AUC  PR-AUC   Precision  Recall  Support(same/different)  Excluded  Threshold  Constraint",
    ]
    for item in (report.stylometry, report.semantic, report.chronobiology, report.combined):
        auc = "—" if item.roc_auc is None else f"{item.roc_auc:.4f}"
        pr = "—" if item.pr_auc is None else f"{item.pr_auc:.4f}"
        lines.append(
            f"{item.engine:<14} {auc:<8} {pr:<8} {item.precision:<10.4f} {item.recall:<7.4f} "
            f"{item.support_same}/{item.support_different:<21} "
            f"{item.excluded_insufficient_data:<9} {item.threshold_percentile:<10.2f} "
            f"{'met' if item.precision_constraint_met else 'unmet'}"
        )
    lines.append("Combined metrics describe a review-prioritization heuristic, not a probability.")
    return "\n".join(lines)


__all__ = ["render_compare_report", "render_evaluation_report"]
