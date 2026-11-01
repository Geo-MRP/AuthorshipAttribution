from __future__ import annotations

from dataclasses import replace
from itertools import product
import math

import pytest

from authorship_attribution.calibration import (
    calibrate_cosine,
    check_baseline_compat,
    compute_combined_result,
)
from authorship_attribution.contracts import BaselineDistribution, EngineResult
from authorship_attribution.errors import ContractValidationError, ErrorCode


_DIGEST = "a" * 64
_ENGINE_ORDER = ("stylometry", "semantic", "chronobiology")
_BASE_WEIGHTS = {"stylometry": 0.40, "semantic": 0.35, "chronobiology": 0.25}
_STATUSES = ("OK", "INSUFFICIENT_DATA", "TIMEOUT", "ERROR")


def _baseline() -> BaselineDistribution:
    return BaselineDistribution(
        schema_version="1.0",
        baseline_id="baseline",
        engine="semantic",
        created_at_utc="2026-01-02T03:04:05Z",
        source_snapshot_id="snapshot",
        configuration_fingerprint=_DIGEST,
        random_seed=1729,
        calibration_population="known_different_account_pairs",
        n_pairs=5,
        sorted_raw_cosines=[-1.0, -0.2, 0.1, 0.1, 1.0],
        minimum_messages=200,
        minimum_chars=5000,
        strata_counts={"game": 5},
        centering_vector_path=None,
    )


def _engine_result(
    engine: str, *, status: str = "OK", percentile: float = 75.0
) -> EngineResult:
    if status == "OK":
        raw_cosine: float | None = 0.25
        calibrated_percentile: float | None = percentile
        error = None
    else:
        raw_cosine = None
        calibrated_percentile = None
        error = {
            "INSUFFICIENT_DATA": ErrorCode.MINIMUMS_NOT_MET.value,
            "TIMEOUT": ErrorCode.ENGINE_TIMEOUT.value,
            "ERROR": ErrorCode.ENGINE_FAILURE.value,
        }[status]
    return EngineResult(
        engine=engine,  # type: ignore[arg-type]
        raw_cosine=raw_cosine,
        calibrated_percentile=calibrated_percentile,
        status=status,  # type: ignore[arg-type]
        n_messages={"candidate": 2, "suspect": 3},
        n_chars={"candidate": 8, "suspect": 12},
        error=error,
    )


@pytest.mark.parametrize(
    ("raw_cosine", "expected"),
    [
        (-1.0, 20.0),
        (-0.5, 20.0),
        (0.1, 80.0),
        (0.5, 80.0),
        (1.0, 100.0),
        (-1.0 - 5e-10, 20.0),
        (1.0 + 5e-10, 100.0),
    ],
)
def test_calibrate_cosine_uses_inclusive_empirical_percentile(
    raw_cosine: float, expected: float
) -> None:
    assert calibrate_cosine(raw_cosine, _baseline()) == expected


@pytest.mark.parametrize("raw_cosine", [-1.1, 1.1, float("nan"), float("inf")])
def test_calibrate_cosine_rejects_materially_out_of_range_values(raw_cosine: float) -> None:
    with pytest.raises(ContractValidationError):
        calibrate_cosine(raw_cosine, _baseline())


def test_check_baseline_compatibility_codes() -> None:
    baseline = _baseline()

    assert check_baseline_compat(baseline, "stylometry", _DIGEST) is ErrorCode.BASELINE_CONFIG_MISMATCH
    assert check_baseline_compat(baseline, "semantic", "b" * 64) is ErrorCode.BASELINE_CONFIG_MISMATCH

    object.__setattr__(baseline, "n_pairs", 6)
    assert check_baseline_compat(baseline, "semantic", _DIGEST) is ErrorCode.BASELINE_INVALID


def test_calibrate_cosine_rejects_invalid_baseline() -> None:
    baseline = _baseline()
    object.__setattr__(baseline, "sorted_raw_cosines", [0.3, 0.2])
    with pytest.raises(ContractValidationError):
        calibrate_cosine(0.2, baseline)


@pytest.mark.parametrize(
    "active_names",
    [
        ("stylometry", "semantic", "chronobiology"),
        ("stylometry",),
        ("semantic",),
        ("chronobiology",),
        ("stylometry", "semantic"),
        ("stylometry", "chronobiology"),
        ("semantic", "chronobiology"),
    ],
)
def test_combined_result_fuses_each_successful_engine_subset(
    active_names: tuple[str, ...],
) -> None:
    percentiles = {"stylometry": 10.0, "semantic": 58.0, "chronobiology": 91.0}
    results = tuple(
        _engine_result(
            name,
            status="OK" if name in active_names else "TIMEOUT",
            percentile=percentiles[name],
        )
        for name in _ENGINE_ORDER
    )

    combined = compute_combined_result(*results)

    active = [name for name in _ENGINE_ORDER if name in active_names]
    active_total = math.fsum(_BASE_WEIGHTS[name] for name in active)
    expected_weights = {name: _BASE_WEIGHTS[name] / active_total for name in active}
    expected_weighted = math.fsum(
        expected_weights[name] * percentiles[name] for name in active
    )
    evidence = math.fsum(
        expected_weights[name]
        * -math.log(max(0.0005, 1.0 - percentiles[name] / 100.0 + 0.0005))
        for name in active
    )
    expected_tail = 100.0 * min(
        1.0, max(0.0, 1.0005 - math.exp(-evidence))
    )
    expected_score = 0.5 * expected_weighted + 0.5 * expected_tail

    assert combined.status == "OK"
    assert combined.active_engines == active
    assert list(combined.weights_used) == active
    assert combined.weights_used == pytest.approx(expected_weights, abs=1e-12)
    assert math.fsum(combined.weights_used.values()) == pytest.approx(1.0, abs=1e-12)
    assert combined.weighted_percentile == pytest.approx(expected_weighted, abs=1e-12)
    assert combined.fisher_tail_score == pytest.approx(expected_tail, abs=1e-12)
    assert combined.combined_score == pytest.approx(expected_score, abs=1e-12)


@pytest.mark.parametrize("percentile", [0.0, 50.0, 100.0])
def test_one_active_engine_preserves_endpoint_percentiles(percentile: float) -> None:
    combined = compute_combined_result(
        _engine_result("stylometry", percentile=percentile),
        _engine_result("semantic", status="INSUFFICIENT_DATA"),
        _engine_result("chronobiology", status="ERROR"),
    )

    assert combined.active_engines == ["stylometry"]
    assert combined.combined_score == pytest.approx(percentile, abs=1e-12)
    assert combined.weighted_percentile == pytest.approx(percentile, abs=1e-12)
    assert combined.fisher_tail_score == pytest.approx(percentile, abs=1e-12)


def test_combined_result_renormalizes_weights_after_timeout() -> None:
    combined = compute_combined_result(
        _engine_result("stylometry", percentile=80.0),
        _engine_result("semantic", percentile=20.0),
        _engine_result("chronobiology", status="TIMEOUT"),
    )

    assert combined.weights_used == pytest.approx(
        {"stylometry": 0.4 / 0.75, "semantic": 0.35 / 0.75}, abs=1e-12
    )
    assert combined.weighted_percentile == pytest.approx(52.0, abs=1e-12)
    assert combined.fisher_tail_score == pytest.approx(61.79428782333969, abs=1e-12)
    assert combined.combined_score == pytest.approx(56.89714391166984, abs=1e-12)


@pytest.mark.parametrize("statuses", list(product(_STATUSES, repeat=3)))
def test_combined_result_handles_every_status_combination(
    statuses: tuple[str, str, str],
) -> None:
    results = tuple(
        _engine_result(name, status=status)
        for name, status in zip(_ENGINE_ORDER, statuses, strict=True)
    )
    combined = compute_combined_result(*results)

    active = [name for name, status in zip(_ENGINE_ORDER, statuses, strict=True) if status == "OK"]
    if active:
        assert combined.status == "OK"
        assert combined.active_engines == active
        assert set(combined.weights_used) == set(active)
        assert all(
            status == "OK"
            for name, status in zip(_ENGINE_ORDER, statuses, strict=True)
            if name in combined.active_engines
        )
        assert not any(
            status != "OK"
            for name, status in zip(_ENGINE_ORDER, statuses, strict=True)
            if name in combined.active_engines
        )
        assert math.fsum(combined.weights_used.values()) == pytest.approx(1.0, abs=1e-12)
        assert combined.combined_score is not None
        assert combined.weighted_percentile is not None
        assert combined.fisher_tail_score is not None
        assert combined.error is None
        return

    if all(status == "INSUFFICIENT_DATA" for status in statuses):
        expected_status = "INSUFFICIENT_DATA"
        expected_error = ErrorCode.MINIMUMS_NOT_MET.value
    elif "TIMEOUT" in statuses:
        expected_status = "TIMEOUT"
        expected_error = ErrorCode.ENGINE_TIMEOUT.value
    else:
        expected_status = "ERROR"
        first_error_index = next(
            index for index, status in enumerate(statuses) if status != "OK"
        )
        expected_error = {
            "INSUFFICIENT_DATA": ErrorCode.MINIMUMS_NOT_MET.value,
            "ERROR": ErrorCode.ENGINE_FAILURE.value,
        }[statuses[first_error_index]]

    assert combined.status == expected_status
    assert combined.error == expected_error
    assert combined.combined_score is None
    assert combined.weighted_percentile is None
    assert combined.fisher_tail_score is None
    assert combined.active_engines == []
    assert combined.weights_used == {}


def test_combined_result_uses_first_error_without_timeout() -> None:
    combined = compute_combined_result(
        _engine_result("stylometry", status="INSUFFICIENT_DATA"),
        _engine_result("semantic", status="ERROR"),
        _engine_result("chronobiology", status="INSUFFICIENT_DATA"),
    )

    assert combined.status == "ERROR"
    assert combined.error == ErrorCode.MINIMUMS_NOT_MET.value


def test_combined_result_rejects_invalid_engine_records_and_order() -> None:
    valid_results = (
        _engine_result("stylometry"),
        _engine_result("semantic"),
        _engine_result("chronobiology"),
    )
    with pytest.raises(ContractValidationError):
        compute_combined_result(
            _engine_result("semantic"),
            _engine_result("stylometry"),
            _engine_result("chronobiology"),
        )
    with pytest.raises(ContractValidationError):
        compute_combined_result(valid_results[0], object(), valid_results[2])  # type: ignore[arg-type]

    mutated = _engine_result("semantic")
    object.__setattr__(mutated, "calibrated_percentile", float("nan"))
    with pytest.raises(ContractValidationError):
        compute_combined_result(valid_results[0], mutated, valid_results[2])

    invalid_status = _engine_result("semantic")
    object.__setattr__(invalid_status, "status", "BROKEN")
    with pytest.raises(ContractValidationError):
        compute_combined_result(valid_results[0], invalid_status, valid_results[2])


@pytest.mark.parametrize(
    "weights",
    [
        [],
        {"stylometry": 0.5, "semantic": 0.5},
        {"stylometry": 0.5, "semantic": 0.3, "chronobiology": 0.2, "extra": 0.0},
        {"stylometry": 1, "semantic": 0.0, "chronobiology": 0.0},
        {"stylometry": True, "semantic": 0.0, "chronobiology": 0.0},
        {"stylometry": 0.0, "semantic": 0.5, "chronobiology": 0.5},
        {"stylometry": -0.1, "semantic": 0.5, "chronobiology": 0.6},
        {"stylometry": float("nan"), "semantic": 0.5, "chronobiology": 0.5},
        {"stylometry": float("inf"), "semantic": 0.5, "chronobiology": 0.5},
        {"stylometry": 1.1, "semantic": -0.05, "chronobiology": -0.05},
        {"stylometry": 0.4, "semantic": 0.35, "chronobiology": 0.3},
    ],
)
def test_combined_result_rejects_invalid_custom_weights(weights: object) -> None:
    results = (
        _engine_result("stylometry"),
        _engine_result("semantic"),
        _engine_result("chronobiology"),
    )
    with pytest.raises(ContractValidationError):
        compute_combined_result(*results, weights=weights)  # type: ignore[arg-type]
