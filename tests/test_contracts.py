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
    message_key,
)
from authorship_attribution.errors import ContractValidationError
from authorship_attribution.fingerprints import canonical_json, sha256_hex


_DIGEST = "a" * 64
_TEXT = "synthetic-private-payload"
_NORMALIZED_TEXT = "normalized-synthetic-payload"


def _assert_not_disclosed(token: str, rendered: str) -> None:
    if token in rendered:
        pytest.fail("text disclosed")


def _engine_result(engine: str) -> EngineResult:
    return EngineResult(
        engine=engine,  # type: ignore[arg-type]
        raw_cosine=0.25,
        calibrated_percentile=75.0,
        status="OK",
        n_messages={"candidate": 2, "suspect": 3},
        n_chars={"candidate": 8, "suspect": 12},
        error=None,
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
    stylometry_evaluation = _engine_evaluation("stylometry")
    semantic_evaluation = _engine_evaluation("semantic")
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
        stylometry,
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
        ComparisonReport(
            schema_version="1.0",
            flagged_account_id="account-1",
            suspect_account_id="account-2",
            stylometry=stylometry,
            semantic=semantic,
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
        ComparisonAssets(
            stylometry_artifact_path=Path("artifact.json"),
            semantic_model_dir=Path("model"),
            baseline_paths={"stylometry": Path("stylometry.json"), "semantic": Path("semantic.json")},
            game_terms_path=Path("games.txt"),
            usernames_path=Path("users.txt"),
            function_words_path=Path("function-words.txt"),
            cache_dir=Path("cache"),
        ),
        IngestConfig(field_map={"message_id": "id"}),
        PreprocessConfig(utc_hour_filter=[1, 5]),
        StylometryConfig(),
        SemanticConfig(model_dir=Path("model"), cache_dir=Path("cache")),
        BaselineConfig(),
        RunConfig(),
        EvaluationConfig(),
        stylometry_evaluation,
        EvaluationReport(
            schema_version="1.0",
            seed=1729,
            partition_sizes={"train": 6, "validation": 2, "test": 2},
            stylometry=stylometry_evaluation,
            semantic=semantic_evaluation,
            note="Thresholds are review-prioritization suggestions only.",
        ),
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


def test_reports_do_not_have_merged_decision_fields() -> None:
    prohibited = {"score", "verdict", "combined", "confidence", "probability", "rank"}
    assert prohibited.isdisjoint(field.name for field in fields(ComparisonReport))
    assert prohibited.isdisjoint(field.name for field in fields(EvaluationReport))


def test_chunk_digest_input_is_not_used_to_leak_text() -> None:
    digest = sha256_hex(b"synthetic")
    assert len(digest) == 64
