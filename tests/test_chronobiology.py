from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import math
import socket

import numpy as np
import pytest

from authorship_attribution.calibration import calibrate_cosine
from authorship_attribution.chronobiology import (
    compute_chronobiology_raw,
    extract_chronobiology_features,
    fit_chronobiology_artifact,
    score_chronobiology,
)
from authorship_attribution.contracts import (
    AccountProfile,
    BaselineDistribution,
    ChronobiologyArtifact,
    ChronobiologyConfig,
    TemporalStreamChunk,
)
from authorship_attribution.errors import ContractValidationError, ErrorCode
from authorship_attribution.fileformats import read_json_artifact, write_json_artifact
from authorship_attribution.fingerprints import (
    canonical_json,
    chronobiology_fingerprint,
    sha256_hex,
)


_PREPROCESSING = "a" * 64


def _chunk(
    account: str,
    session: str,
    chunk_id: str,
    timestamps: list[str],
    lengths: list[int],
) -> TemporalStreamChunk:
    return TemporalStreamChunk(
        schema_version="1.0",
        chunk_id=chunk_id,
        account_id=account,
        session_id=session,
        timestamps_utc=timestamps,
        char_counts=lengths,
        n_messages=len(timestamps),
        n_chars=sum(lengths),
    )


def _profile(
    stream: list[TemporalStreamChunk],
    *,
    preprocessing: str = _PREPROCESSING,
) -> AccountProfile:
    timestamps = [timestamp for chunk in stream for timestamp in chunk.timestamps_utc]
    n_messages = sum(chunk.n_messages for chunk in stream)
    n_chars = sum(chunk.n_chars for chunk in stream)
    return AccountProfile(
        schema_version="1.0",
        account_id=stream[0].account_id,
        game_ids=["game"],
        primary_game_id="game",
        primary_channel_id=None,
        active_months_utc=sorted({timestamp[:7] for timestamp in timestamps}),
        n_messages=n_messages,
        n_chars=n_chars,
        session_count=len({chunk.session_id for chunk in stream}),
        activity_histogram_utc=[0] * 24,
        utc_hour_filter=None,
        chunk_ids=[chunk.chunk_id for chunk in stream],
        raw_stream_path="raw-must-not-be-opened",
        normalized_stream_path="normalized-must-not-be-opened",
        status="READY",
        minimum_messages=1,
        minimum_chars=1,
        preprocessing_fingerprint=preprocessing,
        temporal_stream_path="temporal-must-not-be-opened",
    )


def _baseline(
    artifact: ChronobiologyArtifact,
    config: ChronobiologyConfig,
    values: list[float],
) -> BaselineDistribution:
    return BaselineDistribution(
        schema_version="1.0",
        baseline_id="synthetic-baseline",
        engine="chronobiology",
        created_at_utc="2026-10-01T00:00:00Z",
        source_snapshot_id="synthetic-snapshot",
        configuration_fingerprint=artifact.configuration_fingerprint,
        random_seed=1729,
        calibration_population="known_different_account_pairs",
        n_pairs=len(values),
        sorted_raw_cosines=sorted(values),
        minimum_messages=config.minimum_messages,
        minimum_chars=config.minimum_chars,
        strata_counts={},
        centering_vector_path=None,
    )


@pytest.fixture
def fitted(tmp_path: Path) -> tuple[
    ChronobiologyConfig,
    ChronobiologyArtifact,
    list[TemporalStreamChunk],
    list[TemporalStreamChunk],
]:
    config = ChronobiologyConfig(minimum_messages=1, minimum_chars=1)
    first = [_chunk("account-a", "session-a", "chunk-a", ["2026-10-01T00:00:00Z"], [5])]
    second = [_chunk("account-b", "session-b", "chunk-b", ["2026-10-01T12:00:00Z"], [7])]
    artifact = fit_chronobiology_artifact(
        [first, second], config=config, preprocessing_fp=_PREPROCESSING, output_dir=tmp_path
    )
    return config, artifact, first, second


def test_full_122_feature_single_event_golden() -> None:
    stream = [_chunk("a", "s", "c", ["1970-01-01T00:00:00Z"], [2])]
    result = extract_chronobiology_features(stream)
    assert result.dtype == np.dtype(np.float64)
    assert result.shape == (122,)
    assert np.isfinite(result).all()

    expected: list[float] = []
    for _ in (1, 2, 3, 4, 6):
        expected.extend((1.0, 0.0, 1.0))
    for harmonic in (1, 2, 3):
        angle = 2.0 * math.pi * harmonic * 259200.0 / 604800.0
        real = math.cos(angle)
        imag = math.sin(angle)
        expected.extend((real, imag, math.hypot(real, imag)))
    kernel = [math.exp(-0.5 * min(index, 48 - index) ** 2) for index in range(48)]
    total = math.fsum(kernel)
    smoothed = [value / total for value in kernel]
    expected.extend(smoothed)
    entropy = -math.fsum(value * math.log(value) for value in smoothed if value > 0.0)
    real = math.fsum(
        value * math.cos(2.0 * math.pi * (index + 0.5) / 48.0)
        for index, value in enumerate(smoothed)
    )
    imag = math.fsum(
        value * math.sin(2.0 * math.pi * (index + 0.5) / 48.0)
        for index, value in enumerate(smoothed)
    )
    variance = min(1.0, max(0.0, 1.0 - math.hypot(real, imag)))
    trough_start = 8
    trough_angle = 2.0 * math.pi * ((trough_start + 6) % 48) / 48.0
    trough_mass = math.fsum(smoothed[(trough_start + offset) % 48] for offset in range(12))
    expected.extend((entropy, variance, math.cos(trough_angle), math.sin(trough_angle), trough_mass, 0.0, 0.0))
    tau = math.log1p(1800.0)
    expected.extend([0.0] * 15 + [1.0])
    expected.extend((tau, 0.0, 0.0, 0.0, tau, tau, tau, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    expected.extend((0.0, 0.0, 0.0, 0.0, 0.0, math.log1p(1.0), 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.0))
    assert len(expected) == 122
    np.testing.assert_allclose(result, np.asarray(expected), rtol=1e-12, atol=1e-14)
    assert result[76] == pytest.approx(5.053299110804103e-15, rel=1e-12, abs=0.0)
    assert result[74] == pytest.approx(-0.25881904510252063, abs=1e-14)
    assert result[75] == pytest.approx(0.9659258262890683, abs=1e-14)
    assert math.fsum(float(value) for value in result[24:72]) == pytest.approx(1.0)


def test_uniform_activity_uses_first_rounded_trough_start() -> None:
    from datetime import UTC, datetime, timedelta

    start = datetime(1970, 1, 1, tzinfo=UTC)
    timestamps = [
        (start + timedelta(minutes=30 * index)).isoformat().replace("+00:00", "Z")
        for index in range(48)
    ]
    features = extract_chronobiology_features([_chunk("a", "s", "c", timestamps, [1] * 48)])
    np.testing.assert_allclose(features[24:72], np.full(48, 1.0 / 48.0), atol=1e-15)
    assert features[74] == pytest.approx(math.cos(2.0 * math.pi * 6.0 / 48.0))
    assert features[75] == pytest.approx(math.sin(2.0 * math.pi * 6.0 / 48.0))
    assert features[76] == pytest.approx(0.25)


def test_weekly_origin_weekend_ratio_and_phase_shift() -> None:
    sunday = _chunk("a", "sun", "c1", ["1970-01-04T00:00:00Z"], [1])
    monday = _chunk("a", "mon", "c2", ["1970-01-05T00:00:00Z"], [1])
    features = extract_chronobiology_features([sunday, monday])
    sunday_angle = 2.0 * math.pi * 6.0 / 7.0
    assert features[15] == pytest.approx((math.cos(sunday_angle) + 1.0) / 2.0)
    assert features[16] == pytest.approx(math.sin(sunday_angle) / 2.0, abs=1e-14)
    assert features[77] == 0.5
    assert features[78] == pytest.approx(1.0, abs=1e-12)

    midday = _chunk("a", "midday", "c3", ["1970-01-05T12:00:00Z"], [1])
    cancelled = extract_chronobiology_features([sunday, monday, midday])
    assert cancelled[78] == 0.0


def test_pre_epoch_and_fractional_timestamp_behavior() -> None:
    stream = [
        _chunk(
            "a", "s", "c",
            ["1969-12-31T23:59:59.999999Z", "1970-01-01T00:00:00.123456789Z"],
            [2, 3],
        )
    ]
    first = extract_chronobiology_features(stream)
    second = extract_chronobiology_features(stream)
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    assert first[95] == pytest.approx(math.log1p(0.5))


def test_same_timestamp_velocity_and_fano_degeneracy() -> None:
    stream = [
        _chunk("a", "s", "c", ["1970-01-05T00:00:00Z"] * 3, [2, 3, 5])
    ]
    features = extract_chronobiology_features(stream)
    np.testing.assert_allclose(features[108:112], [4.0, 1.0, 4.0, 4.8])
    assert features[95] == pytest.approx(math.log1p(0.5))
    assert features[96] == features[97] == features[98] == 0.0
    assert features[102] == -1.0
    assert features[103] == 0.0
    assert features[112] == 0.0
    np.testing.assert_array_equal(features[104:108], np.zeros(4))


def test_single_message_sessions_use_global_intervals_but_no_memory_pairs() -> None:
    stream = [
        _chunk("a", "s1", "c1", ["2026-01-01T00:00:00Z"], [1]),
        _chunk("a", "s2", "c2", ["2026-01-01T00:10:00Z"], [2]),
        _chunk("a", "s3", "c3", ["2026-01-01T00:31:00Z"], [3]),
    ]
    features = extract_chronobiology_features(stream)
    intervals = [600.0, 1260.0]
    tau = np.log1p(np.asarray(intervals, dtype=np.float64))
    assert features[95] == pytest.approx(float(np.mean(tau)))
    assert features[96] == pytest.approx(float(np.std(tau)))
    np.testing.assert_allclose(features[99:102], np.quantile(tau, [0.1, 0.5, 0.9], method="linear"))
    assert features[102] == pytest.approx((330.0 - 930.0) / (330.0 + 930.0))
    assert features[103] == 0.0
    np.testing.assert_array_equal(features[108:113], np.zeros(5))
    assert features[117] == 1.0


def test_memory_pairs_do_not_cross_sessions_and_require_three_pairs() -> None:
    one_session = [
        _chunk(
            "a", "s", "c",
            [
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:01Z",
                "2026-01-01T00:00:03Z",
                "2026-01-01T00:00:07Z",
                "2026-01-01T00:00:15Z",
            ],
            [1, 1, 1, 1, 1],
        )
    ]
    assert extract_chronobiology_features(one_session)[103] == pytest.approx(1.0)
    short = [
        _chunk(
            "a", "s", "c",
            [
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:01Z",
                "2026-01-01T00:00:03Z",
                "2026-01-01T00:00:07Z",
            ],
            [1, 1, 1, 1],
        )
    ]
    assert extract_chronobiology_features(short)[103] == 0.0
    separate = [
        _chunk(
            "a", "s1", "c1",
            ["2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z", "2026-01-01T00:00:05Z"],
            [1, 1, 1],
        ),
        _chunk(
            "a", "s2", "c2",
            ["2026-01-01T00:01:00Z", "2026-01-01T00:01:08Z", "2026-01-01T00:01:10Z"],
            [1, 1, 1],
        ),
    ]
    assert extract_chronobiology_features(separate)[103] == 0.0


def test_log_moments_and_composition_correlation_use_locked_domains() -> None:
    stream = [
        _chunk(
            "a", "s", "c",
            [
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:01Z",
                "2026-01-01T00:00:03Z",
                "2026-01-01T00:00:07Z",
                "2026-01-01T00:00:15Z",
            ],
            [1, 1, 2, 4, 8],
        )
    ]
    features = extract_chronobiology_features(stream)
    intervals = np.asarray([1.0, 2.0, 4.0, 8.0])
    tau = np.log1p(intervals)
    mean = float(np.mean(tau))
    std = float(np.std(tau))
    expected_histogram = np.histogram(tau, bins=16, range=(0.0, math.log(1801.0)))[0] / 4.0
    np.testing.assert_array_equal(features[79:95], expected_histogram)
    assert features[95] == pytest.approx(mean)
    assert features[96] == pytest.approx(std)
    assert features[97] == pytest.approx(float(np.mean(((tau - mean) / std) ** 3)))
    assert features[98] == pytest.approx(float(np.mean(((tau - mean) / std) ** 4) - 3.0))
    np.testing.assert_allclose(features[99:102], np.quantile(tau, [0.1, 0.5, 0.9], method="linear"))
    expected_burst = (float(np.std(intervals)) - float(np.mean(intervals))) / (
        float(np.std(intervals)) + float(np.mean(intervals))
    )
    assert features[102] == pytest.approx(expected_burst)
    assert features[112] == pytest.approx(1.0)
    np.testing.assert_allclose(features[108:112], [1.0, 0.0, 1.0, 1.0])


def test_no_qualifying_velocity_has_four_zero_features() -> None:
    stream = [
        _chunk(
            "a", "s", "c",
            ["2026-01-01T00:00:00Z", "2026-01-01T00:05:00Z", "2026-01-01T00:10:00Z"],
            [2, 3, 5],
        )
    ]
    features = extract_chronobiology_features(stream)
    np.testing.assert_array_equal(features[108:112], np.zeros(4))


def test_fano_endpoint_and_sparse_long_session() -> None:
    endpoint = [
        _chunk(
            "a", "s", "c",
            ["1970-01-05T00:00:00Z", "1970-01-05T00:01:00Z", "1970-01-05T00:02:00Z"],
            [1, 1, 1],
        )
    ]
    assert extract_chronobiology_features(endpoint)[104] == pytest.approx(math.log1p(1.0 / 6.0))
    long_session = [
        _chunk("a", "s", "c", ["1900-01-01T00:00:00Z", "2100-01-01T00:00:00Z"], [1, 1])
    ]
    assert np.isfinite(extract_chronobiology_features(long_session)).all()


def test_splitting_a_session_across_chunks_preserves_every_feature() -> None:
    timestamps = ["2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z", "2026-01-01T00:00:03Z"]
    one = [_chunk("a", "s", "one", timestamps, [2, 3, 5])]
    split = [
        _chunk("a", "s", "left", timestamps[:1], [2]),
        _chunk("a", "s", "right", timestamps[1:], [3, 5]),
    ]
    np.testing.assert_array_equal(
        extract_chronobiology_features(one), extract_chronobiology_features(split)
    )


def test_invalid_streams_are_sanitized() -> None:
    first = _chunk("a", "s1", "c1", ["2026-01-01T00:00:00Z"], [1])
    later = _chunk("a", "s2", "c2", ["2026-01-01T00:01:00Z"], [1])
    other = _chunk("other", "s2", "c2", ["2026-01-01T00:01:00Z"], [1])
    duplicate = _chunk("a", "s2", "c1", ["2026-01-01T00:01:00Z"], [1])
    reappeared = _chunk("a", "s1", "c3", ["2026-01-01T00:02:00Z"], [1])
    earlier = _chunk("a", "s2", "c2", ["2025-12-31T23:59:59Z"], [1])
    marker = "SYNTHETIC_PRIVATE_ERROR_MARKER_2371"
    bad_timestamp = _chunk("a", "s", "c", ["2026-01-01T00:00:00Z"], [1])
    object.__setattr__(bad_timestamp, "timestamps_utc", [marker])

    cases = [
        ([], ErrorCode.MINIMUMS_NOT_MET),
        ([marker], ErrorCode.INPUT_INVALID),
        ([first, other], ErrorCode.INPUT_INVALID),
        ([first, duplicate], ErrorCode.INPUT_INVALID),
        ([first, later, reappeared], ErrorCode.INPUT_INVALID),
        ([first, earlier], ErrorCode.INPUT_INVALID),
        ([bad_timestamp], ErrorCode.INPUT_INVALID),
    ]
    for stream, code in cases:
        with pytest.raises(ContractValidationError) as caught:
            extract_chronobiology_features(stream)
        assert caught.value.code is code
        assert marker not in str(caught.value)


def test_unrepresentable_feature_value_is_engine_failure() -> None:
    stream = [
        _chunk("a", "s", "c", ["2026-01-01T00:00:00Z", "2026-01-01T00:00:01Z"], [1, 10**400])
    ]
    with pytest.raises(ContractValidationError) as caught:
        extract_chronobiology_features(stream)
    assert str(caught.value) == "temporal_features:ENGINE_FAILURE"


def test_fitting_identity_is_order_independent_and_payload_is_content_free(
    tmp_path: Path,
) -> None:
    marker = "SYNTHETIC_PRIVATE_ACCOUNT_MARKER_1837"
    config = ChronobiologyConfig(minimum_messages=1, minimum_chars=1)
    first = [_chunk(marker, "s1", "c1", ["2026-01-01T00:00:00Z"], [1])]
    second = [_chunk("other", "s2", "c2", ["2026-01-01T12:00:00Z"], [1])]
    artifact_a = fit_chronobiology_artifact(
        [first, second], config=config, preprocessing_fp=_PREPROCESSING, output_dir=tmp_path / "a"
    )
    artifact_b = fit_chronobiology_artifact(
        [second, first], config=config, preprocessing_fp=_PREPROCESSING, output_dir=tmp_path / "b"
    )
    assert artifact_a.artifact_sha256 == artifact_b.artifact_sha256
    assert artifact_a.configuration_fingerprint == artifact_b.configuration_fingerprint
    assert Path(artifact_a.payload_path).read_bytes() == Path(artifact_b.payload_path).read_bytes()
    payload = read_json_artifact(Path(artifact_a.payload_path))
    assert set(payload) == {
        "feature_spec_version",
        "block_weights",
        "feature_means",
        "feature_stds",
        "preprocessing_fingerprint",
        "baseline_corpus_identity",
        "baseline_account_count",
    }
    assert payload["baseline_account_count"] == 2
    assert sha256_hex(canonical_json(payload)) == artifact_a.artifact_sha256
    assert artifact_a.artifact_id == "chronobiology-" + artifact_a.artifact_sha256
    assert Path(artifact_a.payload_path).name == artifact_a.artifact_id + ".json"
    assert marker not in Path(artifact_a.payload_path).read_text(encoding="utf-8")
    assert marker not in repr(artifact_a)
    assert marker not in repr(payload)


def test_fitting_minimums_duplicate_ids_and_immutable_payload(tmp_path: Path) -> None:
    config = ChronobiologyConfig(minimum_messages=1, minimum_chars=1)
    first = [_chunk("a", "s1", "c1", ["2026-01-01T00:00:00Z"], [1])]
    second = [_chunk("b", "s2", "c2", ["2026-01-01T12:00:00Z"], [1])]
    with pytest.raises(ContractValidationError) as caught:
        fit_chronobiology_artifact([[], first], config=config, preprocessing_fp=_PREPROCESSING, output_dir=tmp_path)
    assert caught.value.code is ErrorCode.MINIMUMS_NOT_MET
    with pytest.raises(ContractValidationError) as caught:
        fit_chronobiology_artifact([first, first], config=config, preprocessing_fp=_PREPROCESSING, output_dir=tmp_path)
    assert caught.value.code is ErrorCode.INPUT_INVALID
    with pytest.raises(ContractValidationError) as caught:
        fit_chronobiology_artifact([first, second], config=config, preprocessing_fp="not-a-digest", output_dir=tmp_path)
    assert caught.value.code is ErrorCode.INPUT_INVALID

    artifact = fit_chronobiology_artifact(
        [first, second], config=config, preprocessing_fp=_PREPROCESSING, output_dir=tmp_path
    )
    path = Path(artifact.payload_path)
    path.write_bytes(b"corrupt")
    with pytest.raises(ContractValidationError) as caught:
        fit_chronobiology_artifact(
            [first, second], config=config, preprocessing_fp=_PREPROCESSING, output_dir=tmp_path
        )
    assert caught.value.code is ErrorCode.ARTIFACT_INVALID
    assert path.read_bytes() == b"corrupt"


def test_score_calibrates_inclusive_ties_and_checks_profile_ownership(
    fitted: tuple[ChronobiologyConfig, ChronobiologyArtifact, list[TemporalStreamChunk], list[TemporalStreamChunk]],
) -> None:
    config, artifact, first, second = fitted
    raw = compute_chronobiology_raw(first, first, artifact=artifact, config=config)
    assert raw == pytest.approx(1.0)
    baseline = _baseline(artifact, config, [raw, raw])
    profile = _profile(first)
    result = score_chronobiology(profile, profile, first, first, artifact=artifact, baseline=baseline, config=config)
    assert result.engine == "chronobiology"
    assert result.status == "OK"
    assert result.raw_cosine == pytest.approx(raw)
    assert result.calibrated_percentile == calibrate_cosine(raw, baseline) == 100.0
    assert result.error is None
    assert result.n_messages == {"candidate": 1, "suspect": 1}
    assert result.n_chars == {"candidate": 5, "suspect": 5}

    wrong_stream = score_chronobiology(profile, profile, first, second, artifact=artifact, baseline=baseline, config=config)
    assert wrong_stream.status == "ERROR"
    assert wrong_stream.error == ErrorCode.INPUT_INVALID.value
    assert wrong_stream.raw_cosine is None and wrong_stream.calibrated_percentile is None


def test_score_minimums_baseline_and_preprocessing_mismatch(
    fitted: tuple[ChronobiologyConfig, ChronobiologyArtifact, list[TemporalStreamChunk], list[TemporalStreamChunk]],
) -> None:
    config, artifact, first, second = fitted
    left = _profile(first)
    right = _profile(second)
    baseline = _baseline(artifact, config, [-1.0, 0.0])
    insufficient = replace(left, minimum_messages=2, status="INSUFFICIENT_DATA")
    result = score_chronobiology(insufficient, right, first, second, artifact=artifact, baseline=baseline, config=config)
    assert result.status == "INSUFFICIENT_DATA" and result.error == "MINIMUMS_NOT_MET"
    assert result.raw_cosine is None and result.calibrated_percentile is None

    incompatible_profile = replace(left, preprocessing_fingerprint="b" * 64)
    result = score_chronobiology(incompatible_profile, right, first, second, artifact=artifact, baseline=baseline, config=config)
    assert result.status == "ERROR" and result.error == "BASELINE_CONFIG_MISMATCH"
    incompatible_baseline = replace(baseline, minimum_messages=2)
    result = score_chronobiology(left, right, first, second, artifact=artifact, baseline=incompatible_baseline, config=config)
    assert result.status == "ERROR" and result.error == "BASELINE_CONFIG_MISMATCH"
    wrong_engine = replace(baseline, engine="semantic")
    result = score_chronobiology(left, right, first, second, artifact=artifact, baseline=wrong_engine, config=config)
    assert result.status == "ERROR" and result.error == "BASELINE_CONFIG_MISMATCH"

    object.__setattr__(baseline, "sorted_raw_cosines", [0.5, -0.5])
    result = score_chronobiology(left, right, first, second, artifact=artifact, baseline=baseline, config=config)
    assert result.status == "ERROR" and result.error == "BASELINE_INVALID"


def test_zero_variance_artifact_produces_norm_error(tmp_path: Path) -> None:
    config = ChronobiologyConfig(minimum_messages=1, minimum_chars=1)
    first = [_chunk("a", "s1", "c1", ["2026-01-01T00:00:00Z"], [1])]
    second = [_chunk("b", "s2", "c2", ["2026-01-01T00:00:00Z"], [1])]
    artifact = fit_chronobiology_artifact(
        [first, second], config=config, preprocessing_fp=_PREPROCESSING, output_dir=tmp_path
    )
    assert artifact.feature_stds == [0.0] * 122
    with pytest.raises(ContractValidationError) as caught:
        compute_chronobiology_raw(first, second, artifact=artifact, config=config)
    assert str(caught.value) == "chronobiology_norm:ENGINE_FAILURE"
    baseline = _baseline(artifact, config, [0.0])
    result = score_chronobiology(
        _profile(first), _profile(second), first, second,
        artifact=artifact, baseline=baseline, config=config,
    )
    assert result.status == "ERROR" and result.error == "ENGINE_FAILURE"
    assert result.raw_cosine is None and result.calibrated_percentile is None


def test_payload_tampering_and_configuration_mismatch_are_distinguished(
    fitted: tuple[ChronobiologyConfig, ChronobiologyArtifact, list[TemporalStreamChunk], list[TemporalStreamChunk]],
) -> None:
    config, artifact, first, second = fitted
    different_config = replace(
        config,
        block_weights={"circadian_fourier": 0.50, "burst_process": 0.25, "session_kinematics": 0.25},
    )
    with pytest.raises(ContractValidationError) as caught:
        compute_chronobiology_raw(first, second, artifact=artifact, config=different_config)
    assert caught.value.code is ErrorCode.BASELINE_CONFIG_MISMATCH

    path = Path(artifact.payload_path)
    payload = read_json_artifact(path)
    payload["baseline_account_count"] = 3
    write_json_artifact(payload, path)
    with pytest.raises(ContractValidationError) as caught:
        compute_chronobiology_raw(first, second, artifact=artifact, config=config)
    assert caught.value.code is ErrorCode.ARTIFACT_INVALID
    baseline = _baseline(artifact, config, [-1.0, 0.0])
    result = score_chronobiology(
        _profile(first), _profile(second), first, second,
        artifact=artifact, baseline=baseline, config=config,
    )
    assert result.status == "ERROR" and result.error == "ARTIFACT_INVALID"


@pytest.mark.parametrize("case", ["missing_key", "metadata_means", "account_count", "integer_weight"])
def test_rehashed_malformed_payload_is_still_rejected(
    fitted: tuple[ChronobiologyConfig, ChronobiologyArtifact, list[TemporalStreamChunk], list[TemporalStreamChunk]],
    case: str,
) -> None:
    config, artifact, first, second = fitted
    payload = read_json_artifact(Path(artifact.payload_path))
    if case == "missing_key":
        del payload["baseline_account_count"]
    elif case == "metadata_means":
        payload["feature_means"][0] += 1.0
    elif case == "account_count":
        payload["baseline_account_count"] = 1
    elif case == "integer_weight":
        payload["block_weights"]["circadian_fourier"] = 1
    else:
        raise AssertionError("unknown test case")
    digest = sha256_hex(canonical_json(payload))
    artifact_id = "chronobiology-" + digest
    path = Path(artifact.payload_path).parent / (artifact_id + ".json")
    write_json_artifact(payload, path)
    rehashed = replace(
        artifact,
        artifact_id=artifact_id,
        artifact_sha256=digest,
        payload_path=str(path),
        configuration_fingerprint=chronobiology_fingerprint(config, digest, _PREPROCESSING),
    )
    with pytest.raises(ContractValidationError) as caught:
        compute_chronobiology_raw(first, second, artifact=rehashed, config=config)
    assert caught.value.code is ErrorCode.ARTIFACT_INVALID


def test_weighted_block_cosine_uses_squared_scales(tmp_path: Path) -> None:
    config = ChronobiologyConfig(minimum_messages=1, minimum_chars=1)
    first = [_chunk("a", "s1", "c1", ["2026-01-01T00:00:00Z"], [1])]
    second = [_chunk("b", "s2", "c2", ["2026-01-01T06:00:00Z"], [1])]
    payload = {
        "feature_spec_version": config.feature_spec_version,
        "block_weights": dict(config.block_weights),
        "feature_means": [0.0] * 122,
        "feature_stds": [1.0] * 122,
        "preprocessing_fingerprint": _PREPROCESSING,
        "baseline_corpus_identity": "b" * 64,
        "baseline_account_count": 2,
    }
    digest = sha256_hex(canonical_json(payload))
    artifact_id = "chronobiology-" + digest
    path = tmp_path / (artifact_id + ".json")
    write_json_artifact(payload, path)
    artifact = ChronobiologyArtifact(
        artifact_id=artifact_id,
        artifact_sha256=digest,
        feature_spec_version=config.feature_spec_version,
        feature_means=[0.0] * 122,
        feature_stds=[1.0] * 122,
        payload_path=str(path),
        configuration_fingerprint=chronobiology_fingerprint(config, digest, _PREPROCESSING),
    )
    left = extract_chronobiology_features(first)
    right = extract_chronobiology_features(second)
    weighted_cosines: list[float] = []
    squared_weights: list[float] = []
    for name, start, end in (
        ("circadian_fourier", 0, 79),
        ("burst_process", 79, 108),
        ("session_kinematics", 108, 122),
    ):
        a = left[start:end]
        b = right[start:end]
        cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        squared_weight = config.block_weights[name] ** 2
        weighted_cosines.append(squared_weight * cosine)
        squared_weights.append(squared_weight)
    expected = math.fsum(weighted_cosines) / math.fsum(squared_weights)
    assert compute_chronobiology_raw(first, second, artifact=artifact, config=config) == pytest.approx(expected)


def test_engine_never_uses_network_and_does_not_log_private_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    marker = "SYNTHETIC_PRIVATE_NETWORK_MARKER_7460"

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    config = ChronobiologyConfig(minimum_messages=1, minimum_chars=1)
    first = [_chunk(marker, "s1", "c1", ["2026-01-01T00:00:00Z"], [1])]
    second = [_chunk("b", "s2", "c2", ["2026-01-01T12:00:00Z"], [1])]
    with caplog.at_level("DEBUG"):
        artifact = fit_chronobiology_artifact(
            [first, second], config=config, preprocessing_fp=_PREPROCESSING, output_dir=tmp_path
        )
        raw = compute_chronobiology_raw(first, first, artifact=artifact, config=config)
        baseline = _baseline(artifact, config, [raw])
        result = score_chronobiology(
            _profile(first), _profile(first), first, first,
            artifact=artifact, baseline=baseline, config=config,
        )
    assert result.status == "OK"
    assert marker not in caplog.text
    assert marker not in Path(artifact.payload_path).read_text(encoding="utf-8")
