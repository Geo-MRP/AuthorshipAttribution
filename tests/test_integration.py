from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path

import pytest

from authorship_attribution import (
    AccountProfile,
    BaselineConfig,
    BaselineDistribution,
    ComparisonReport,
    EngineResult,
    LabeledPair,
    RunConfig,
)
from authorship_attribution.baselines import baseline_needs_refresh
from authorship_attribution.calibration import calibrate_cosine
from authorship_attribution.evaluate import _components, _partition_accounts
from authorship_attribution.reporting import render_compare_report


def _result(engine: str, *, status: str = "OK", raw: float | None = 0.4, percentile: float | None = 50.0, error: str | None = None) -> EngineResult:
    return EngineResult(
        engine=engine, raw_cosine=raw, calibrated_percentile=percentile,
        status=status, n_messages={"candidate": 210, "suspect": 220},
        n_chars={"candidate": 5100, "suspect": 5300}, error=error,
    )


def _baseline() -> BaselineDistribution:
    return BaselineDistribution(
        schema_version="1.0", baseline_id="b-id", engine="semantic",
        created_at_utc="2026-10-01T00:00:00Z", source_snapshot_id="snapshot",
        configuration_fingerprint="a" * 64, random_seed=1729,
        calibration_population="known_different_account_pairs", n_pairs=4,
        sorted_raw_cosines=[-0.5, 0.1, 0.1, 0.9], minimum_messages=200,
        minimum_chars=5000, strata_counts={"same-game|same-channel|same-month": 4},
        centering_vector_path=None,
    )


def _pair(left: str, right: str, label: str) -> LabeledPair:
    return LabeledPair("1.0", left, right, label, "snapshot")


def test_report_has_permanent_banner_and_two_independent_results() -> None:
    report = ComparisonReport(
        "1.0", "flagged", "suspect", _result("stylometry"), _result("semantic"),
        ComparisonReport.BANNER,
    )
    rendered = render_compare_report(report, output_format="table")
    assert rendered.splitlines()[0] == ComparisonReport.BANNER
    assert "Semantic" in rendered and "Stylometry" in rendered
    assert "Known-different baseline percentile" in rendered
    assert "combined" not in rendered.casefold()
    encoded = render_compare_report(report, output_format="json")
    assert encoded.splitlines()[0] == ComparisonReport.BANNER
    assert "combined_score" not in encoded


def test_compare_report_table_lines_have_equal_length() -> None:
    report = ComparisonReport(
        "1.0", "flagged", "suspect", _result("stylometry"), _result("semantic"),
        ComparisonReport.BANNER,
    )
    rendered = render_compare_report(report, output_format="table")
    table_lines = [line for line in rendered.splitlines() if line.startswith(("+", "|"))]
    assert len({len(line) for line in table_lines}) == 1


@pytest.mark.parametrize(("score", "expected"), [(-0.8, 0.0), (0.1, 75.0), (1.0, 100.0)])
def test_contract_calibration_is_inclusive_for_ties_and_boundaries(score: float, expected: float) -> None:
    assert calibrate_cosine(score, _baseline()) == expected


def test_same_label_components_are_kept_in_one_partition() -> None:
    pairs = [_pair("a", "b", "same"), _pair("b", "c", "same"), _pair("d", "e", "same"), _pair("f", "g", "different")]
    groups = _components(pairs)
    assert {"a", "b", "c"} in groups
    assert {"d", "e"} in groups
    partitions = _partition_accounts(pairs, 1729, (0.6, 0.2, 0.2))
    assert partitions["a"] == partitions["b"] == partitions["c"]
    assert partitions["d"] == partitions["e"]


def test_refresh_detects_month_and_pair_growth_and_fingerprint() -> None:
    baseline = _baseline()
    config = BaselineConfig()
    assert baseline_needs_refresh(baseline, current_fingerprint="b" * 64, new_eligible_pairs=0, config=config)
    assert baseline_needs_refresh(baseline, current_fingerprint="a" * 64, new_eligible_pairs=1, config=config)
    assert baseline_needs_refresh(
        baseline, current_fingerprint="a" * 64, new_eligible_pairs=0, config=config,
        now=datetime(2026, 11, 5, tzinfo=UTC),
    )


def test_integration_module_outputs_avoid_content_payloads(tmp_path: Path) -> None:
    marker = "SYNTHETIC_PRIVATE_RAW_CHAT_7319"
    report = ComparisonReport("1.0", "flagged", "suspect", _result("stylometry"), _result("semantic"), ComparisonReport.BANNER)
    out = render_compare_report(report, output_format="json")
    assert marker not in out
    assert marker not in json.dumps(_baseline().to_dict())
