from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from authorship_attribution.calibration import calibrate_cosine
from authorship_attribution.contracts import (
    AccountProfile,
    BaselineDistribution,
    RawStreamChunk,
    StylometryConfig,
)
from authorship_attribution.errors import ErrorCode
from authorship_attribution.stylometry import (
    _structural_features,
    compute_stylometry_raw,
    fit_stylometry_artifact,
    score_stylometry,
)


def _stream(account_id: str, messages: list[str], chunk_id: str = "chunk") -> list[RawStreamChunk]:
    return [
        RawStreamChunk(
            schema_version="1.0",
            chunk_id=chunk_id,
            account_id=account_id,
            session_id=f"session-{account_id}",
            message_ids=[f"{account_id}-{index}" for index in range(len(messages))],
            start_timestamp_utc="2026-10-01T12:00:00Z",
            end_timestamp_utc="2026-10-01T12:00:00Z",
            n_messages=len(messages),
            n_chars=sum(map(len, messages)),
            raw_text=messages,
        )
    ]


def _profile(account_id: str, messages: list[str], *, minimum: int = 1) -> AccountProfile:
    return AccountProfile(
        schema_version="1.0",
        account_id=account_id,
        game_ids=["game"],
        primary_game_id="game",
        primary_channel_id="channel",
        active_months_utc=["2026-10"],
        n_messages=len(messages),
        n_chars=sum(map(len, messages)),
        session_count=1,
        activity_histogram_utc=[len(messages)] + [0] * 23,
        utc_hour_filter=None,
        chunk_ids=["chunk"],
        raw_stream_path="unused-raw-path",
        normalized_stream_path="must-not-be-read",
        status="READY" if len(messages) >= minimum and sum(map(len, messages)) >= minimum else "INSUFFICIENT_DATA",
        minimum_messages=minimum,
        minimum_chars=minimum,
        preprocessing_fingerprint="0" * 64,
    )


@pytest.fixture
def corpus() -> list[list[RawStreamChunk]]:
    return [
        _stream("base-a", ["I think this is useful!", "and then we go"]),
        _stream("base-b", ["well, I think so.", "then we can go"]),
    ]


@pytest.fixture
def config() -> StylometryConfig:
    return StylometryConfig(minimum_messages=1, minimum_chars=1)


def test_exact_copy_cosine_is_one(tmp_path: Path, corpus: list[list[RawStreamChunk]], config: StylometryConfig) -> None:
    artifact = fit_stylometry_artifact(corpus, function_words=["and", "I", "we"], config=config, seed=19, output_dir=tmp_path)
    assert compute_stylometry_raw(corpus[0], corpus[0], artifact=artifact, config=config) == pytest.approx(1.0)


def test_structural_fixture_values_and_case_only_change(tmp_path: Path, corpus: list[list[RawStreamChunk]], config: StylometryConfig) -> None:
    original = ["A  B!!!...🙂 looo"]
    lowered = ["a  b!!!...🙂 looo"]
    features = _structural_features(original, ["a", "b"])
    lower_features = _structural_features(lowered, ["a", "b"])
    assert features[0] == pytest.approx(1 / len(original[0]))
    punctuation_offset = 1
    assert features[punctuation_offset + 0] == pytest.approx(3 / len(original[0]))
    assert features[punctuation_offset + 13] == pytest.approx(3 / len(original[0]))
    assert features[33] == pytest.approx(2 / len(original[0]))
    assert features[34] == pytest.approx(2 / 6)
    assert features[35] == pytest.approx(1 / len(original[0]))
    assert features[36] == pytest.approx(1 / len(original[0]))
    assert features[37] == pytest.approx(len(original[0]))
    assert features[38:] == pytest.approx([1 / 3, 1 / 3])
    assert features != lower_features
    assert original == ["A  B!!!...🙂 looo"]


def test_ngram_documents_do_not_cross_message_boundaries(tmp_path: Path, config: StylometryConfig) -> None:
    corpus = [_stream("only", ["ab", "cd"])]
    artifact = fit_stylometry_artifact(corpus, function_words=[], config=config, seed=5, output_dir=tmp_path)
    payload = json.loads(Path(artifact.payload_path).read_text(encoding="utf-8"))
    from authorship_attribution.stylometry import _hash_bucket

    cross_boundary = str(_hash_bucket("bc", 5, config.hash_space_size))
    assert int(payload["n_documents"]) == 2
    # "bc" occurs only if a boundary is incorrectly introduced between raw messages.
    assert cross_boundary not in payload["idf"]


def test_fit_is_deterministic_and_payload_has_no_raw_text(tmp_path: Path, corpus: list[list[RawStreamChunk]], config: StylometryConfig) -> None:
    marker = "UNIQUE_SYNTHETIC_RAW_MARKER"
    streams = [_stream("fixture-a", [marker, "words here"])]
    first = fit_stylometry_artifact(streams, function_words=["here", "and"], config=config, seed=12, output_dir=tmp_path / "one")
    second = fit_stylometry_artifact(streams, function_words=["here", "and"], config=config, seed=12, output_dir=tmp_path / "two")
    assert first.artifact_sha256 == second.artifact_sha256
    payload_text = Path(first.payload_path).read_text(encoding="utf-8")
    assert marker not in payload_text
    assert "UNIQUE" not in payload_text
    payload = json.loads(payload_text)
    assert set(payload["idf"]).issubset({str(index) for index in range(1 << 18)})
    assert all(isinstance(value, float) for value in payload["idf"].values())
    assert len(payload["structural_means"]) == len(payload["structural_stds"])


def test_minimum_failure_and_baseline_mismatch_have_no_scores(tmp_path: Path, corpus: list[list[RawStreamChunk]], config: StylometryConfig) -> None:
    artifact = fit_stylometry_artifact(corpus, function_words=["and"], config=config, seed=19, output_dir=tmp_path)
    baseline = BaselineDistribution(
        schema_version="1.0", baseline_id="baseline", engine="stylometry",
        created_at_utc="2026-10-01T12:00:00Z", source_snapshot_id="snapshot",
        configuration_fingerprint=artifact.configuration_fingerprint, random_seed=19,
        calibration_population="known_different_account_pairs", n_pairs=2,
        sorted_raw_cosines=[0.1, 0.5], minimum_messages=1, minimum_chars=1,
        strata_counts={}, centering_vector_path=None,
    )
    too_small = _profile("small", ["x"], minimum=2)
    ready = _profile("ready", ["a longer message"])
    failed = score_stylometry(too_small, ready, _stream("small", ["x"]), _stream("ready", ["a longer message"]), artifact=artifact, baseline=baseline, config=config)
    assert failed.status == "INSUFFICIENT_DATA"
    assert failed.error == ErrorCode.MINIMUMS_NOT_MET.value
    assert failed.raw_cosine is failed.calibrated_percentile is None

    incompatible = BaselineDistribution(
        schema_version="1.0", baseline_id="other", engine="stylometry",
        created_at_utc="2026-10-01T12:00:00Z", source_snapshot_id="snapshot",
        configuration_fingerprint="1" * 64, random_seed=19,
        calibration_population="known_different_account_pairs", n_pairs=1,
        sorted_raw_cosines=[0.2], minimum_messages=1, minimum_chars=1,
        strata_counts={}, centering_vector_path=None,
    )
    mismatch = score_stylometry(ready, ready, _stream("ready", ["a longer message"]), _stream("ready", ["a longer message"]), artifact=artifact, baseline=incompatible, config=config)
    assert mismatch.status == "ERROR"
    assert mismatch.error == ErrorCode.BASELINE_CONFIG_MISMATCH.value
    assert mismatch.raw_cosine is mismatch.calibrated_percentile is None


def test_score_calibrates_compatible_baseline_and_logs_no_text(tmp_path: Path, caplog: pytest.LogCaptureFixture, corpus: list[list[RawStreamChunk]], config: StylometryConfig) -> None:
    marker = "DO_NOT_LOG_RAW_MARKER"
    artifact = fit_stylometry_artifact(corpus, function_words=["and"], config=config, seed=19, output_dir=tmp_path)
    stream = _stream("ready", [marker, "another line"])
    profile = _profile("ready", [marker, "another line"])
    baseline = BaselineDistribution(
        schema_version="1.0", baseline_id="baseline", engine="stylometry",
        created_at_utc="2026-10-01T12:00:00Z", source_snapshot_id="snapshot",
        configuration_fingerprint=artifact.configuration_fingerprint, random_seed=19,
        calibration_population="known_different_account_pairs", n_pairs=2,
        sorted_raw_cosines=[0.1, 0.5], minimum_messages=1, minimum_chars=1,
        strata_counts={}, centering_vector_path=None,
    )
    with caplog.at_level(logging.INFO):
        result = score_stylometry(profile, profile, stream, stream, artifact=artifact, baseline=baseline, config=config)
    assert result.status == "OK"
    assert result.raw_cosine == pytest.approx(1.0)
    assert 0 <= result.calibrated_percentile <= 100
    assert result.calibrated_percentile == calibrate_cosine(result.raw_cosine, baseline)
    assert marker not in caplog.text


def test_engine_makes_no_network_calls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, corpus: list[list[RawStreamChunk]], config: StylometryConfig) -> None:
    import socket

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    artifact = fit_stylometry_artifact(corpus, function_words=["and"], config=config, seed=19, output_dir=tmp_path)
    assert compute_stylometry_raw(corpus[0], corpus[0], artifact=artifact, config=config) == pytest.approx(1.0)


def test_similar_styles_rank_above_distinct_synthetic_style(tmp_path: Path, config: StylometryConfig) -> None:
    baseline_streams = [
        _stream("b1", ["I think, and I agree!", "we can do this."]),
        _stream("b2", ["I think and I agree!", "we can do this."]),
    ]
    artifact = fit_stylometry_artifact(baseline_streams, function_words=["I", "and", "we"], config=config, seed=31, output_dir=tmp_path)
    reference = _stream("ref", ["I think, and I agree!", "we can do this."])
    similar = _stream("similar", ["I think and I agree!", "we can do this."])
    different = _stream("different", ["perhaps; perhaps? perhaps.", "maybe: maybe, maybe!"])
    near = compute_stylometry_raw(reference, similar, artifact=artifact, config=config)
    far = compute_stylometry_raw(reference, different, artifact=artifact, config=config)
    assert near > far
