from __future__ import annotations

from dataclasses import replace

import pytest

from authorship_attribution.calibration import calibrate_cosine, check_baseline_compat
from authorship_attribution.contracts import BaselineDistribution
from authorship_attribution.errors import ContractValidationError, ErrorCode


_DIGEST = "a" * 64


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
