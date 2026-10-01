"""The sole baseline percentile calibration implementation."""

from __future__ import annotations

from bisect import bisect_right
import math

from .contracts import BaselineDistribution
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
