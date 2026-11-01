from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
from typing import Any

import pytest

from authorship_attribution.contracts import (
    AccountProfile,
    BaselineConfig,
    BaselineDistribution,
    Chunk,
    ChronobiologyArtifact,
    ChronobiologyConfig,
    CombinedResult,
    ComparisonAssets,
    ComparisonReport,
    EngineEvaluation,
    EngineResult,
    EvaluationConfig,
    EvaluationReport,
    IngestConfig,
    LabeledPair,
    Message,
    NormalizedStreamChunk,
    PreprocessConfig,
    RawStreamChunk,
    RunConfig,
    SemanticConfig,
    StylometryArtifact,
    StylometryConfig,
    TemporalStreamChunk,
    message_key,
)
from authorship_attribution.calibration import compute_combined_result
from authorship_attribution.errors import ContractValidationError, ErrorCode
from authorship_attribution.fingerprints import canonical_json, sha256_hex


_DIGEST = "a" * 64
_TEXT = "synthetic-private-payload"
_NORMALIZED_TEXT = "normalized-synthetic-payload"


def _assert_not_disclosed(token: str, rendered: str) -> None:
    if token in rendered:
        pytest.fail("text disclosed")


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


def _engine_evaluation(engine: str) -> EngineEvaluation:
    return EngineEvaluation(
        engine=engine,  # type: ignore[arg-type]
        roc_auc=0.7,
        pr_auc=0.6,
        test_base_rate=0.5,
        threshold_percentile=80.0,
        precision_constraint_met=True,
        precision=0.85,
        recall=0.4,
        support_same=4,
        support_different=5,
        excluded_insufficient_data=1,
        ci={
            "roc_auc": [0.6, 0.8],
            "precision": [0.7, 0.9],
            "recall": [0.3, 0.5],
        },
    )


def _samples() -> list[Any]:
    message_ids = [message_key("game/one", "message:one")]
    chunk = Chunk(
        schema_version="1.0",
        chunk_id="chunk-1",
        account_id="account-1",
        session_id="session-1",
        message_ids=message_ids,
        start_timestamp_utc="2026-01-02T03:04:05Z",
        end_timestamp_utc="2026-01-02T03:04:05Z",
        n_messages=1,
        n_chars=len(_TEXT),
        raw_text=[_TEXT],
        normalized_text=_NORMALIZED_TEXT,
        normalized_sha256=_DIGEST,
    )
    stylometry = _engine_result("stylometry")
    semantic = _engine_result("semantic")
    chronobiology = _engine_result("chronobiology")
    combined = compute_combined_result(stylometry, semantic, chronobiology)
    stylometry_evaluation = _engine_evaluation("stylometry")
    semantic_evaluation = _engine_evaluation("semantic")
    chronobiology_evaluation = _engine_evaluation("chronobiology")
    combined_evaluation = _engine_evaluation("combined")
    return [
        Message(
            schema_version="1.0",
            message_id="message:one",
            account_id="account-1",
            game_id="game/one",
            channel_id="channel-1",
            timestamp_utc="2026-01-02T03:04:05Z",
            text=_TEXT,
            is_system=False,
            source_id="source-1",
            source_row=1,
        ),
        chunk,
        chunk.to_raw_stream_chunk(),
        chunk.to_normalized_stream_chunk(),
        AccountProfile(
            schema_version="1.0",
            account_id="account-1",
            game_ids=["game/one"],
            primary_game_id="game/one",
            primary_channel_id="channel-1",
            active_months_utc=["2026-01"],
            n_messages=1,
            n_chars=len(_TEXT),
            session_count=1,
            activity_histogram_utc=[0] * 24,
            utc_hour_filter=[3],
            chunk_ids=["chunk-1"],
            raw_stream_path="streams/raw.ndjson",
            normalized_stream_path="streams/normalized.ndjson",
            status="INSUFFICIENT_DATA",
            minimum_messages=200,
            minimum_chars=5000,
            preprocessing_fingerprint=_DIGEST,
        ),
        TemporalStreamChunk(
            schema_version="1.0",
            chunk_id="temporal-chunk-1",
            account_id="account-1",
            session_id="session-1",
            timestamps_utc=["2026-01-02T03:04:05Z"],
            char_counts=[len(_TEXT)],
            n_messages=1,
            n_chars=len(_TEXT),
        ),
        stylometry,
        chronobiology,
        combined,
        BaselineDistribution(
            schema_version="1.0",
            baseline_id="baseline-1",
            engine="semantic",
            created_at_utc="2026-01-02T03:04:05Z",
            source_snapshot_id="snapshot-1",
            configuration_fingerprint=_DIGEST,
            random_seed=1729,
            calibration_population="known_different_account_pairs",
            n_pairs=3,
            sorted_raw_cosines=[-0.1, 0.0, 0.3],
            minimum_messages=200,
            minimum_chars=5000,
            strata_counts={"game": 3},
            centering_vector_path="means/semantic.bin",
        ),
        BaselineDistribution(
            schema_version="1.0",
            baseline_id="chronobiology-baseline-1",
            engine="chronobiology",
            created_at_utc="2026-01-02T03:04:05Z",
            source_snapshot_id="snapshot-1",
            configuration_fingerprint=_DIGEST,
            random_seed=1729,
            calibration_population="known_different_account_pairs",
            n_pairs=3,
            sorted_raw_cosines=[-0.1, 0.0, 0.3],
            minimum_messages=200,
            minimum_chars=5000,
            strata_counts={"game": 3},
            centering_vector_path=None,
        ),
        ComparisonReport(
            schema_version="1.0",
            flagged_account_id="account-1",
            suspect_account_id="account-2",
            stylometry=stylometry,
            semantic=semantic,
            chronobiology=chronobiology,
            combined=combined,
            banner="Decision support only. Requires human review.",
        ),
        LabeledPair(
            schema_version="1.0",
            account_id_a="z-account",
            account_id_b="a-account",
            label="different",
            source_snapshot_id="snapshot-1",
        ),
        StylometryArtifact(
            artifact_id="artifact-1",
            artifact_sha256=_DIGEST,
            feature_spec_version="1.0",
            hash_space_size=16,
            function_words_sha256=_DIGEST,
            function_words_lang="en",
            structural_means=[0.0, 1.0],
            structural_stds=[1.0, 2.0],
            payload_path="payload.bin",
            seed=1729,
            configuration_fingerprint=_DIGEST,
        ),
        ChronobiologyConfig(),
        ChronobiologyArtifact(
            artifact_id="chronobiology-artifact-1",
            artifact_sha256=_DIGEST,
            feature_spec_version="1.0",
            feature_means=[0.0] * 122,
            feature_stds=[0.0] * 122,
            payload_path="chronobiology-payload.json",
            configuration_fingerprint=_DIGEST,
        ),
        ComparisonAssets(
            stylometry_artifact_path=Path("artifact.json"),
            semantic_model_dir=Path("model"),
            baseline_paths={
                "stylometry": Path("stylometry.json"),
                "semantic": Path("semantic.json"),
                "chronobiology": Path("chronobiology.json"),
            },
            game_terms_path=Path("games.txt"),
            usernames_path=Path("users.txt"),
            function_words_path=Path("function-words.txt"),
            cache_dir=Path("cache"),
            chronobiology_artifact_path=Path("chronobiology-artifact.json"),
        ),
        IngestConfig(field_map={"message_id": "id"}),
        PreprocessConfig(utc_hour_filter=[1, 5]),
        StylometryConfig(),
        SemanticConfig(model_dir=Path("model"), cache_dir=Path("cache")),
        BaselineConfig(),
        RunConfig(),
        EvaluationConfig(),
        stylometry_evaluation,
        chronobiology_evaluation,
        EvaluationReport(
            schema_version="1.0",
            seed=1729,
            partition_sizes={"train": 6, "validation": 2, "test": 2},
            stylometry=stylometry_evaluation,
            semantic=semantic_evaluation,
            chronobiology=chronobiology_evaluation,
            combined=combined_evaluation,
            note="Thresholds are review-prioritization suggestions only.",
        ),
        combined_evaluation,
    ]


@pytest.mark.parametrize("sample", _samples())
def test_contracts_round_trip_byte_identically(sample: Any) -> None:
    serialized = canonical_json(sample.to_dict())
    restored = type(sample).from_dict(json.loads(serialized))

    assert restored.to_dict() == sample.to_dict()
    assert canonical_json(restored.to_dict()) == serialized


def test_message_key_is_unambiguous_for_separator_ids() -> None:
    assert message_key("game/a:b", "message/c:d") == '["game/a:b","message/c:d"]'
    assert message_key("game/a", "b:c") != message_key("game", "a/b:c")


def test_from_dict_rejects_unknown_missing_bad_timestamp_nan_and_cross_fields() -> None:
    message = _samples()[0].to_dict()
    with pytest.raises(ContractValidationError):
        Message.from_dict({**message, "unknown": 1})
    missing = dict(message)
    del missing["source_row"]
    with pytest.raises(ContractValidationError):
        Message.from_dict(missing)
    with pytest.raises(ContractValidationError):
        Message.from_dict({**message, "timestamp_utc": "2026-01-02T03:04:05+00:00"})

    chunk = _samples()[1].to_dict()
    with pytest.raises(ContractValidationError):
        Chunk.from_dict({**chunk, "n_messages": 2})

    profile = next(item for item in _samples() if isinstance(item, AccountProfile)).to_dict()
    with pytest.raises(ContractValidationError):
        AccountProfile.from_dict({**profile, "activity_histogram_utc": [0] * 23})

    baseline = next(item for item in _samples() if isinstance(item, BaselineDistribution)).to_dict()
    with pytest.raises(ContractValidationError):
        BaselineDistribution.from_dict({**baseline, "sorted_raw_cosines": [float("nan")] * 3})

    result = _engine_result("semantic").to_dict()
    with pytest.raises(ContractValidationError):
        EngineResult.from_dict({**result, "error": "ENGINE_FAILURE"})
    with pytest.raises(ContractValidationError):
        EngineResult.from_dict(
            {
                **result,
                "status": "ERROR",
                "raw_cosine": None,
                "calibrated_percentile": None,
                "error": None,
            }
        )


def _empty_post_filter_profile() -> AccountProfile:
    return AccountProfile(
        schema_version="1.0",
        account_id="account-empty",
        game_ids=["game-before-filter"],
        primary_game_id="game-before-filter",
        primary_channel_id="channel-before-filter",
        active_months_utc=[],
        n_messages=0,
        n_chars=0,
        session_count=0,
        activity_histogram_utc=[0] * 24,
        utc_hour_filter=[4],
        chunk_ids=[],
        raw_stream_path="streams/empty.raw.ndjson",
        normalized_stream_path="streams/empty.normalized.ndjson",
        status="INSUFFICIENT_DATA",
        minimum_messages=200,
        minimum_chars=5000,
        preprocessing_fingerprint=_DIGEST,
    )


def test_empty_post_filter_profile_validates_and_round_trips() -> None:
    profile = _empty_post_filter_profile()
    serialized = canonical_json(profile.to_dict())

    restored = AccountProfile.from_dict(json.loads(serialized))

    assert restored == profile
    assert canonical_json(restored.to_dict()) == serialized


@pytest.mark.parametrize(
    "changes",
    [
        {"n_chars": 1},
        {"session_count": 1},
        {"chunk_ids": ["chunk-1"]},
        {"active_months_utc": ["2026-01"]},
    ],
)
def test_empty_post_filter_profile_rejects_each_nonempty_derived_value(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ContractValidationError):
        AccountProfile.from_dict({**_empty_post_filter_profile().to_dict(), **changes})


@pytest.mark.parametrize(
    "changes",
    [
        {"n_messages": 1, "n_chars": 0},
        {"n_messages": 1, "session_count": 0},
        {"n_messages": 1, "chunk_ids": []},
        {"n_messages": 1, "active_months_utc": []},
    ],
)
def test_nonempty_post_filter_profile_rejects_each_empty_derived_value(
    changes: dict[str, object],
) -> None:
    profile = next(item for item in _samples() if isinstance(item, AccountProfile))
    with pytest.raises(ContractValidationError):
        AccountProfile.from_dict({**profile.to_dict(), **changes})


def test_profile_status_must_match_minimums() -> None:
    below_minimums = next(item for item in _samples() if isinstance(item, AccountProfile))
    with pytest.raises(ContractValidationError):
        AccountProfile.from_dict({**below_minimums.to_dict(), "status": "READY"})

    ready = AccountProfile(
        schema_version="1.0",
        account_id="account-ready",
        game_ids=["game-1"],
        primary_game_id="game-1",
        primary_channel_id=None,
        active_months_utc=["2026-01"],
        n_messages=2,
        n_chars=10,
        session_count=1,
        activity_histogram_utc=[0] * 24,
        utc_hour_filter=None,
        chunk_ids=["chunk-1"],
        raw_stream_path="streams/ready.raw.ndjson",
        normalized_stream_path="streams/ready.normalized.ndjson",
        status="READY",
        minimum_messages=2,
        minimum_chars=10,
        preprocessing_fingerprint=_DIGEST,
    )
    with pytest.raises(ContractValidationError):
        AccountProfile.from_dict({**ready.to_dict(), "status": "INSUFFICIENT_DATA"})


def test_text_carrying_repr_and_errors_never_disclose_text() -> None:
    text_types = [
        item
        for item in _samples()
        if isinstance(item, (Message, Chunk, RawStreamChunk, NormalizedStreamChunk))
    ]
    for item in text_types:
        _assert_not_disclosed(_TEXT, repr(item))
        _assert_not_disclosed(_TEXT, str(item))
        _assert_not_disclosed(_NORMALIZED_TEXT, repr(item))
        _assert_not_disclosed(_NORMALIZED_TEXT, str(item))

    with pytest.raises(ContractValidationError) as captured:
        Message.from_dict({**_samples()[0].to_dict(), "text": 123})
    _assert_not_disclosed(_TEXT, str(captured.value))


def test_labeled_pairs_are_canonicalized() -> None:
    pair = LabeledPair(
        schema_version="1.0",
        account_id_a="z",
        account_id_b="a",
        label="same",
        source_snapshot_id="snapshot",
    )

    assert (pair.account_id_a, pair.account_id_b) == ("a", "z")


def test_reports_have_combined_records_but_no_automated_decision_fields() -> None:
    prohibited = {"score", "verdict", "confidence", "probability", "rank"}
    comparison_fields = {item.name for item in fields(ComparisonReport)}
    evaluation_fields = {item.name for item in fields(EvaluationReport)}

    assert {"chronobiology", "combined"} <= comparison_fields
    assert {"chronobiology", "combined"} <= evaluation_fields
    assert prohibited.isdisjoint(comparison_fields)
    assert prohibited.isdisjoint(evaluation_fields)


def test_chunk_digest_input_is_not_used_to_leak_text() -> None:
    digest = sha256_hex(b"synthetic")
    assert len(digest) == 64


def test_account_profile_legacy_migration_and_direct_constructor_default() -> None:
    profile = _empty_post_filter_profile()
    assert profile.temporal_stream_path == "unused-temporal-path"

    legacy = profile.to_dict()
    del legacy["temporal_stream_path"]
    restored = AccountProfile.from_dict(legacy)

    assert restored.temporal_stream_path == "unused-temporal-path"
    assert restored.to_dict()["temporal_stream_path"] == "unused-temporal-path"

    current = profile.to_dict()
    for invalid_path in ("", None, 17):
        with pytest.raises(ContractValidationError):
            AccountProfile.from_dict({**current, "temporal_stream_path": invalid_path})
    with pytest.raises(ContractValidationError):
        AccountProfile.from_dict({**legacy, "unexpected": "value"})
    missing_existing_field = dict(legacy)
    del missing_existing_field["account_id"]
    with pytest.raises(ContractValidationError):
        AccountProfile.from_dict(missing_existing_field)


def _temporal_data() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "chunk_id": "temporal-chunk",
        "account_id": "account-temporal",
        "session_id": "session-temporal",
        "timestamps_utc": [
            "1969-12-31T23:59:59.500000Z",
            "1969-12-31T23:59:59.500000Z",
            "1970-01-01T00:00:00Z",
        ],
        "char_counts": [1, 2, 3],
        "n_messages": 3,
        "n_chars": 6,
    }


def test_temporal_stream_accepts_fractional_equal_and_pre_epoch_timestamps() -> None:
    record = TemporalStreamChunk.from_dict(_temporal_data())
    serialized = canonical_json(record.to_dict())
    restored = TemporalStreamChunk.from_dict(json.loads(serialized))

    assert restored == record
    assert canonical_json(restored.to_dict()) == serialized


@pytest.mark.parametrize(
    "changes",
    [
        {"timestamps_utc": ["malformed-timestamp"]},
        {"timestamps_utc": ["2026-01-02T03:04:05+00:00"]},
        {
            "timestamps_utc": [
                "2026-01-02T03:04:06Z",
                "2026-01-02T03:04:05Z",
                "2026-01-02T03:04:07Z",
            ]
        },
        {"timestamps_utc": [], "char_counts": [], "n_messages": 0, "n_chars": 0},
        {"char_counts": [1, 2], "n_chars": 3},
        {"n_messages": 2},
        {"n_chars": 7},
        {"char_counts": [-1, 2, 5], "n_chars": 6},
        {"char_counts": [True, 2, 3], "n_chars": 6},
    ],
)
def test_temporal_stream_rejects_bad_timestamps_and_count_invariants(
    changes: dict[str, object],
) -> None:
    with pytest.raises(ContractValidationError):
        TemporalStreamChunk.from_dict({**_temporal_data(), **changes})


def test_temporal_stream_from_dict_is_strict_and_content_free() -> None:
    data = _temporal_data()
    with pytest.raises(ContractValidationError):
        TemporalStreamChunk.from_dict({**data, "unknown": "private"})
    missing = dict(data)
    del missing["session_id"]
    with pytest.raises(ContractValidationError):
        TemporalStreamChunk.from_dict(missing)

    marker = "synthetic-private-temporal-text-marker"
    valid = TemporalStreamChunk.from_dict(data)
    _assert_not_disclosed(marker, repr(valid))
    with pytest.raises(ContractValidationError) as captured:
        TemporalStreamChunk.from_dict(
            {**data, "timestamps_utc": [marker], "char_counts": [1], "n_messages": 1, "n_chars": 1}
        )
    _assert_not_disclosed(marker, str(captured.value))


def test_chronobiology_config_validation_and_independent_defaults() -> None:
    first = ChronobiologyConfig()
    second = ChronobiologyConfig()
    assert first.block_weights == {
        "circadian_fourier": 0.45,
        "burst_process": 0.30,
        "session_kinematics": 0.25,
    }
    assert first.block_weights is not second.block_weights

    with pytest.raises(ContractValidationError):
        ChronobiologyConfig(block_weights={"circadian_fourier": 0.45, "burst_process": 0.55})
    with pytest.raises(ContractValidationError):
        ChronobiologyConfig(
            block_weights={
                "circadian_fourier": 0.45,
                "burst_process": 0.30,
                "session_kinematics": 0.25,
                "extra": 0.0,
            }
        )

    invalid_weights = [
        {"circadian_fourier": -0.1, "burst_process": 0.3, "session_kinematics": 0.8},
        {"circadian_fourier": float("nan"), "burst_process": 0.3, "session_kinematics": 0.25},
        {"circadian_fourier": float("inf"), "burst_process": 0.3, "session_kinematics": 0.25},
        {"circadian_fourier": 1, "burst_process": 0.0, "session_kinematics": 0.0},
        {"circadian_fourier": True, "burst_process": 0.0, "session_kinematics": 0.0},
        {"circadian_fourier": 0.45, "burst_process": 0.30, "session_kinematics": 0.25000000001},
    ]
    for weights in invalid_weights:
        with pytest.raises(ContractValidationError):
            ChronobiologyConfig(block_weights=weights)  # type: ignore[arg-type]

    for minimum_field in ("minimum_messages", "minimum_chars"):
        with pytest.raises(ContractValidationError):
            ChronobiologyConfig(**{minimum_field: 0})  # type: ignore[arg-type]

    config_data = first.to_dict()
    del config_data["minimum_chars"]
    with pytest.raises(ContractValidationError):
        ChronobiologyConfig.from_dict(config_data)
    with pytest.raises(ContractValidationError):
        ChronobiologyConfig.from_dict({**first.to_dict(), "unexpected": 1})


def test_chronobiology_artifact_requires_122_finite_features_and_valid_digests() -> None:
    artifact = next(
        item for item in _samples() if isinstance(item, ChronobiologyArtifact)
    )
    assert artifact.feature_stds == [0.0] * 122
    data = artifact.to_dict()

    for field_name in ("feature_means", "feature_stds"):
        for size in (121, 123):
            malformed = {**data, field_name: [0.0] * size}
            with pytest.raises(ContractValidationError):
                ChronobiologyArtifact.from_dict(malformed)
    for field_name in ("feature_means", "feature_stds"):
        malformed_values = [0.0] * 122
        malformed_values[0] = float("nan")
        with pytest.raises(ContractValidationError):
            ChronobiologyArtifact.from_dict({**data, field_name: malformed_values})
    negative_stds = [0.0] * 122
    negative_stds[0] = -0.1
    with pytest.raises(ContractValidationError):
        ChronobiologyArtifact.from_dict({**data, "feature_stds": negative_stds})
    for field_name in ("artifact_sha256", "configuration_fingerprint"):
        with pytest.raises(ContractValidationError):
            ChronobiologyArtifact.from_dict({**data, field_name: "not-a-sha256"})
    missing = dict(data)
    del missing["payload_path"]
    with pytest.raises(ContractValidationError):
        ChronobiologyArtifact.from_dict(missing)
    with pytest.raises(ContractValidationError):
        ChronobiologyArtifact.from_dict({**data, "unexpected": 1})


def _combined_data() -> dict[str, object]:
    return next(item for item in _samples() if isinstance(item, CombinedResult)).to_dict()


def _failed_combined(status: str, error: str) -> dict[str, object]:
    return CombinedResult(
        status=status,  # type: ignore[arg-type]
        combined_score=None,
        weighted_percentile=None,
        fisher_tail_score=None,
        active_engines=[],
        weights_used={},
        error=error,
    ).to_dict()


def test_combined_result_rejects_invalid_status_engines_and_weights() -> None:
    valid = _combined_data()
    invalid_changes = [
        {"status": "UNKNOWN"},
        {"active_engines": ["stylometry", "stylometry", "chronobiology"]},
        {"active_engines": ["semantic", "stylometry", "chronobiology"]},
        {"active_engines": ["stylometry", "unknown", "chronobiology"]},
        {"weights_used": {"stylometry": 0.5, "semantic": 0.5}},
        {"weights_used": {"stylometry": 0.0, "semantic": 0.5, "chronobiology": 0.5}},
        {"weights_used": {"stylometry": -0.1, "semantic": 0.5, "chronobiology": 0.6}},
        {"weights_used": {"stylometry": float("nan"), "semantic": 0.5, "chronobiology": 0.5}},
        {"weights_used": {"stylometry": float("inf"), "semantic": 0.5, "chronobiology": 0.5}},
        {"weights_used": {"stylometry": 1, "semantic": 0.0, "chronobiology": 0.0}},
        {"weights_used": {"stylometry": True, "semantic": 0.0, "chronobiology": 0.0}},
        {"weights_used": {"stylometry": 0.3, "semantic": 0.3, "chronobiology": 0.3}},
        {"combined_score": None},
        {"combined_score": 101.0},
        {"weighted_percentile": -0.1},
        {"fisher_tail_score": float("inf")},
        {"combined_score": 10.0},
        {"error": ErrorCode.ENGINE_FAILURE.value},
    ]
    for changes in invalid_changes:
        with pytest.raises(ContractValidationError):
            CombinedResult.from_dict({**valid, **changes})

    missing = dict(valid)
    del missing["fisher_tail_score"]
    with pytest.raises(ContractValidationError):
        CombinedResult.from_dict(missing)
    with pytest.raises(ContractValidationError):
        CombinedResult.from_dict({**valid, "unexpected": 1})


def test_combined_result_failure_fields_and_error_codes_are_strict() -> None:
    timeout = _failed_combined("TIMEOUT", ErrorCode.ENGINE_TIMEOUT.value)
    insufficient = _failed_combined(
        "INSUFFICIENT_DATA", ErrorCode.MINIMUMS_NOT_MET.value
    )

    for valid, changes in (
        (timeout, {"active_engines": ["stylometry"], "weights_used": {"stylometry": 1.0}}),
        (timeout, {"status": "TIMEOUT", "error": ErrorCode.MINIMUMS_NOT_MET.value}),
        (insufficient, {"error": ErrorCode.ENGINE_TIMEOUT.value}),
    ):
        with pytest.raises(ContractValidationError):
            CombinedResult.from_dict({**valid, **changes})


def _comparison_report_data() -> dict[str, object]:
    return next(
        item for item in _samples() if isinstance(item, ComparisonReport)
    ).to_dict()


def _evaluation_report_data() -> dict[str, object]:
    return next(
        item for item in _samples() if isinstance(item, EvaluationReport)
    ).to_dict()


def test_reports_reject_engine_mismatches_active_mismatch_and_bad_precedence() -> None:
    report = _comparison_report_data()
    for engine_name in ("stylometry", "semantic", "chronobiology"):
        malformed = dict(report)
        malformed_result = dict(report[engine_name])  # type: ignore[arg-type]
        malformed_result["engine"] = "semantic" if engine_name != "semantic" else "stylometry"
        malformed[engine_name] = malformed_result
        with pytest.raises(ContractValidationError):
            ComparisonReport.from_dict(malformed)

    inactive_combined = compute_combined_result(
        _engine_result("stylometry"),
        _engine_result("semantic", status="TIMEOUT"),
        _engine_result("chronobiology", status="ERROR"),
    )
    with pytest.raises(ContractValidationError):
        ComparisonReport.from_dict({**report, "combined": inactive_combined.to_dict()})

    all_insufficient = {
        name: _engine_result(name, status="INSUFFICIENT_DATA").to_dict()
        for name in ("stylometry", "semantic", "chronobiology")
    }
    timeout_combined = _failed_combined("TIMEOUT", ErrorCode.ENGINE_TIMEOUT.value)
    with pytest.raises(ContractValidationError):
        ComparisonReport.from_dict({**report, **all_insufficient, "combined": timeout_combined})

    mixed_failed = {
        "stylometry": _engine_result("stylometry", status="INSUFFICIENT_DATA").to_dict(),
        "semantic": _engine_result("semantic", status="ERROR").to_dict(),
        "chronobiology": _engine_result("chronobiology", status="INSUFFICIENT_DATA").to_dict(),
    }
    wrong_error_combined = _failed_combined("ERROR", ErrorCode.ENGINE_FAILURE.value)
    with pytest.raises(ContractValidationError):
        ComparisonReport.from_dict({**report, **mixed_failed, "combined": wrong_error_combined})


def test_reports_reject_missing_nested_fields_unknown_fields_and_changed_notices() -> None:
    report = _comparison_report_data()
    missing_nested = dict(report)
    combined = dict(report["combined"])  # type: ignore[arg-type]
    del combined["fisher_tail_score"]
    missing_nested["combined"] = combined
    with pytest.raises(ContractValidationError):
        ComparisonReport.from_dict(missing_nested)
    with pytest.raises(ContractValidationError):
        ComparisonReport.from_dict({**report, "unexpected": 1})
    with pytest.raises(ContractValidationError):
        ComparisonReport.from_dict({**report, "banner": "Automated decision"})

    evaluation = _evaluation_report_data()
    for engine_name in ("stylometry", "semantic", "chronobiology", "combined"):
        malformed = dict(evaluation)
        malformed_result = dict(evaluation[engine_name])  # type: ignore[arg-type]
        malformed_result["engine"] = "combined" if engine_name != "combined" else "stylometry"
        malformed[engine_name] = malformed_result
        with pytest.raises(ContractValidationError):
            EvaluationReport.from_dict(malformed)
    missing_evaluation_nested = dict(evaluation)
    del missing_evaluation_nested["chronobiology"]
    with pytest.raises(ContractValidationError):
        EvaluationReport.from_dict(missing_evaluation_nested)
    with pytest.raises(ContractValidationError):
        EvaluationReport.from_dict({**evaluation, "unexpected": 1})
    with pytest.raises(ContractValidationError):
        EvaluationReport.from_dict(
            {**evaluation, "note": "Thresholds are decisions."}
        )


def test_comparison_assets_require_three_baselines_and_chronobiology_artifact() -> None:
    assets = next(item for item in _samples() if isinstance(item, ComparisonAssets))
    serialized = assets.to_dict()
    for baseline_paths in (
        {"stylometry": "stylometry.json", "semantic": "semantic.json"},
        {
            "stylometry": "stylometry.json",
            "semantic": "semantic.json",
            "chronobiology": "chronobiology.json",
            "unknown": "unknown.json",
        },
    ):
        with pytest.raises(ContractValidationError):
            ComparisonAssets.from_dict({**serialized, "baseline_paths": baseline_paths})
    missing_path = dict(serialized)
    del missing_path["chronobiology_artifact_path"]
    with pytest.raises(ContractValidationError):
        ComparisonAssets.from_dict(missing_path)


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("nan"), float("inf")])
def test_run_config_rejects_invalid_chronobiology_timeout(timeout: float) -> None:
    run_config = next(item for item in _samples() if isinstance(item, RunConfig))
    with pytest.raises(ContractValidationError):
        RunConfig.from_dict(
            {**run_config.to_dict(), "chronobiology_timeout_s": timeout}
        )
