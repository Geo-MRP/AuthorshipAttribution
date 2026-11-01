"""The sole baseline percentile calibration implementation."""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Mapping
import math
from typing import cast

from .contracts import BaselineDistribution, CombinedResult, EngineResult
from .errors import ContractValidationError, ErrorCode


def calibrate_cosine(raw_cosine: float, baseline: BaselineDistribution) -> float:
    """Map a cosine to its inclusive empirical baseline percentile.

    Scores within 1e-9 outside the cosine domain are clamped to accommodate
    floating-point roundoff. Larger excursions and non-finite values are
    invalid contract input.
    """

    if type(raw_cosine) is not float or not math.isfinite(raw_cosine):
        raise ContractValidationError("raw_cosine", ErrorCode.INPUT_INVALID)
    baseline._validate()
    if raw_cosine < -1.0:
        if raw_cosine < -1.0 - 1e-9:
            raise ContractValidationError("raw_cosine", ErrorCode.INPUT_INVALID)
        raw_cosine = -1.0
    elif raw_cosine > 1.0:
        if raw_cosine > 1.0 + 1e-9:
            raise ContractValidationError("raw_cosine", ErrorCode.INPUT_INVALID)
        raw_cosine = 1.0
    return 100.0 * bisect_right(baseline.sorted_raw_cosines, raw_cosine) / baseline.n_pairs


def check_baseline_compat(
    baseline: BaselineDistribution, engine: str, expected_fingerprint: str
) -> ErrorCode | None:
    """Return the stable reason a baseline cannot be used, if any."""

    if baseline.engine != engine or baseline.configuration_fingerprint != expected_fingerprint:
        return ErrorCode.BASELINE_CONFIG_MISMATCH
    try:
        baseline._validate()
    except ContractValidationError:
        return ErrorCode.BASELINE_INVALID
    return None


_FUSION_ENGINE_ORDER = ("stylometry", "semantic", "chronobiology")
_FUSION_EPSILON = 0.0005


def _invalid_fusion_input(field_name: str) -> None:
    raise ContractValidationError(field_name, ErrorCode.INPUT_INVALID)


def compute_combined_result(
    stylometry: EngineResult,
    semantic: EngineResult,
    chronobiology: EngineResult,
    *,
    weights: Mapping[str, float] | None = None,
) -> CombinedResult:
    results = (stylometry, semantic, chronobiology)
    for engine_name, result in zip(_FUSION_ENGINE_ORDER, results, strict=True):
        if not isinstance(result, EngineResult):
            _invalid_fusion_input(engine_name)
        result._validate()
        if result.engine != engine_name:
            _invalid_fusion_input(engine_name)

    supplied_weights: Mapping[str, float]
    if weights is None:
        supplied_weights = {
            "stylometry": 0.40,
            "semantic": 0.35,
            "chronobiology": 0.25,
        }
    else:
        supplied_weights = weights
    if not isinstance(supplied_weights, Mapping) or set(supplied_weights) != set(_FUSION_ENGINE_ORDER):
        _invalid_fusion_input("weights")
    original_weights: dict[str, float] = {}
    for name in _FUSION_ENGINE_ORDER:
        value = supplied_weights[name]
        if type(value) is not float or not math.isfinite(value) or value <= 0.0 or value > 1.0:
            _invalid_fusion_input("weights")
        original_weights[name] = value
    if not math.isclose(math.fsum(original_weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        _invalid_fusion_input("weights")

    active_results = [result for result in results if result.status == "OK"]
    if not active_results:
        if all(result.status == "INSUFFICIENT_DATA" for result in results):
            status = "INSUFFICIENT_DATA"
            error = ErrorCode.MINIMUMS_NOT_MET.value
        elif any(result.status == "TIMEOUT" for result in results):
            status = "TIMEOUT"
            error = ErrorCode.ENGINE_TIMEOUT.value
        else:
            status = "ERROR"
            error = next(result.error for result in results if result.error is not None)
        return CombinedResult(
            status=status,
            combined_score=None,
            weighted_percentile=None,
            fisher_tail_score=None,
            active_engines=[],
            weights_used={},
            error=error,
        )

    active_weight_total = math.fsum(original_weights[result.engine] for result in active_results)
    weights_used = {
        result.engine: original_weights[result.engine] / active_weight_total
        for result in active_results
    }
    weighted_percentile = math.fsum(
        weights_used[result.engine] * cast(float, result.calibrated_percentile)
        for result in active_results
    )
    weighted_percentile = min(100.0, max(0.0, weighted_percentile))
    evidence = math.fsum(
        weights_used[result.engine]
        * -math.log(
            max(
                _FUSION_EPSILON,
                1.0 - cast(float, result.calibrated_percentile) / 100.0 + _FUSION_EPSILON,
            )
        )
        for result in active_results
    )
    fisher_tail_score = 100.0 * min(
        1.0, max(0.0, 1.0 + _FUSION_EPSILON - math.exp(-evidence))
    )
    combined_score = min(
        100.0, max(0.0, 0.5 * weighted_percentile + 0.5 * fisher_tail_score)
    )
    return CombinedResult(
        status="OK",
        combined_score=combined_score,
        weighted_percentile=weighted_percentile,
        fisher_tail_score=fisher_tail_score,
        active_engines=[result.engine for result in active_results],
        weights_used=weights_used,
        error=None,
    )
