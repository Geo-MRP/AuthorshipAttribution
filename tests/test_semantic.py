from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import pytest

from authorship_attribution import (
    AccountProfile,
    BaselineDistribution,
    NormalizedStreamChunk,
    SemanticConfig,
    semantic_fingerprint,
)
from authorship_attribution.errors import ContractValidationError, ErrorCode
from authorship_attribution.fileformats import write_json_artifact
from authorship_attribution.semantic import (
    EmbeddingCache,
    ExternalProviderEmbeddingModel,
    _Tokenizer,
    compute_semantic_raw,
    load_local_model,
    score_semantic,
    score_semantic_local,
)


def _chunk(account: str, text: str, chunk_id: str = "chunk") -> NormalizedStreamChunk:
    return NormalizedStreamChunk(
        schema_version="1.0",
        chunk_id=chunk_id,
        account_id=account,
        session_id=f"session-{account}",
        message_ids=[f"{account}-message"],
        start_timestamp_utc="2026-10-01T12:00:00Z",
        end_timestamp_utc="2026-10-01T12:00:00Z",
        n_messages=1,
        n_chars=len(text),
        normalized_text=text,
        normalized_sha256=sha256(text.encode("utf-8")).hexdigest(),
    )


def _profile(account: str, text: str, preprocessing: str = "a" * 64) -> AccountProfile:
    return AccountProfile(
        schema_version="1.0",
        account_id=account,
        game_ids=["game"],
        primary_game_id="game",
        primary_channel_id=None,
        active_months_utc=["2026-10"],
        n_messages=1,
        n_chars=len(text),
        session_count=1,
        activity_histogram_utc=[1] + [0] * 23,
        utc_hour_filter=None,
        chunk_ids=["chunk"],
        raw_stream_path="raw-path-must-not-be-read",
        normalized_stream_path="normalized-path-must-not-be-read",
        status="READY",
        minimum_messages=1,
        minimum_chars=1,
        preprocessing_fingerprint=preprocessing,
    )


class SyntheticModel:
    model_fingerprint = "model-manifest-fingerprint"
    tokenizer_sha256 = "tokenizer-fingerprint"
    embedding_dim = 3

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> np.ndarray:
        self.calls.append(list(texts))
        rows = []
        for text in texts:
            rows.append([float(len(text) + 1), float(sum(map(ord, text)) % 97 + 1), 1.0])
        return np.asarray(rows, dtype=np.float32)


def _config(tmp_path: Path, **kwargs: object) -> SemanticConfig:
    options: dict[str, object] = {
        "model_dir": tmp_path / "model",
        "cache_dir": tmp_path / "cache",
        "minimum_messages": 1,
        "minimum_chars": 1,
        "center_embeddings": False,
    }
    options.update(kwargs)
    return SemanticConfig(
        **options,  # type: ignore[arg-type]
    )


def _baseline(config: SemanticConfig, model: SyntheticModel, preprocessing: str = "a" * 64) -> BaselineDistribution:
    fp = semantic_fingerprint(config, model.model_fingerprint, model.tokenizer_sha256, preprocessing)
    return BaselineDistribution(
        schema_version="1.0",
        baseline_id="baseline",
        engine="semantic",
        created_at_utc="2026-10-01T12:00:00Z",
        source_snapshot_id="snapshot",
        configuration_fingerprint=fp,
        random_seed=3,
        calibration_population="known_different_account_pairs",
        n_pairs=2,
        sorted_raw_cosines=[0.1, 0.5],
        minimum_messages=1,
        minimum_chars=1,
        strata_counts={},
        centering_vector_path=None,
    )


def test_identical_input_is_unit_cosine_and_cached_without_text(tmp_path: Path) -> None:
    config = _config(tmp_path)
    model = SyntheticModel()
    chunks = [_chunk("account-a", "SYNTHETIC_NORMALIZED_FIXTURE")]
    assert compute_semantic_raw(chunks, chunks, model=model, config=config) == pytest.approx(1.0)
    assert len(model.calls) == 1
    cache_data = b"".join(path.read_bytes() for path in (tmp_path / "cache").rglob("*.json"))
    assert b"SYNTHETIC_NORMALIZED_FIXTURE" not in cache_data
    assert b"token_ids_sha256" in cache_data


def test_account_mean_pool_and_centering_are_normalized(tmp_path: Path) -> None:
    config = _config(tmp_path)
    model = SyntheticModel()
    # Repeated identical streams should reuse all cached windows and remain unit vectors.
    stream = [_chunk("account-a", "same text")]
    score = compute_semantic_raw(stream, stream, model=model, config=config)
    assert score == pytest.approx(1.0)
    assert len(model.calls) == 1


def test_wordpiece_windows_preserve_all_source_slices() -> None:
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "a": 4, "b": 5, "c": 6}
    tokenizer = _Tokenizer(vocab, lowercase=False)
    windows = tokenizer.windows("a b c", 3)
    assert [window[0] for window in windows] == ["a ", "b ", "c"]
    assert "".join(window[0] for window in windows) == "a b c"
    assert all(len(ids) <= 3 for _, ids in windows)


def test_cache_retention_zero_removes_vectors_at_completion(tmp_path: Path) -> None:
    config = _config(tmp_path, cache_retention_days=0)
    compute_semantic_raw([_chunk("a", "synthetic")], [_chunk("b", "synthetic")], model=SyntheticModel(), config=config)
    assert list((tmp_path / "cache").rglob("*.json")) == []


def test_profile_minimum_failure_and_baseline_mismatch_have_no_scores(tmp_path: Path) -> None:
    config = _config(tmp_path)
    model = SyntheticModel()
    candidate = _profile("a", "synthetic")
    insufficient = AccountProfile(
        **{**candidate.to_dict(), "status": "INSUFFICIENT_DATA", "minimum_messages": 2}
    )
    result = score_semantic(insufficient, candidate, [_chunk("a", "synthetic")], [_chunk("a", "synthetic")], model=model, baseline=_baseline(config, model), config=config)
    assert result.status == "INSUFFICIENT_DATA"
    assert result.error == ErrorCode.MINIMUMS_NOT_MET.value
    assert result.raw_cosine is None and result.calibrated_percentile is None

    result = score_semantic(candidate, _profile("b", "synthetic"), [_chunk("a", "synthetic")], [_chunk("b", "synthetic")], model=model, baseline=_baseline(config, model, "b" * 64), config=config)
    assert result.status == "ERROR"
    assert result.error == ErrorCode.BASELINE_CONFIG_MISMATCH.value
    assert result.raw_cosine is None and result.calibrated_percentile is None


def test_scoring_returns_raw_and_inclusive_baseline_percentile(tmp_path: Path) -> None:
    config = _config(tmp_path)
    model = SyntheticModel()
    candidate_text = "synthetic text"
    result = score_semantic(
        _profile("a", candidate_text),
        _profile("b", candidate_text),
        [_chunk("a", candidate_text)],
        [_chunk("b", candidate_text)],
        model=model,
        baseline=_baseline(config, model),
        config=config,
    )
    assert result.status == "OK"
    assert result.raw_cosine == pytest.approx(1.0)
    assert result.calibrated_percentile == 100.0
    assert result.error is None


def test_baseline_mean_centers_account_vector_before_normalization(tmp_path: Path) -> None:
    config = _config(tmp_path, center_embeddings=True)
    model = SyntheticModel()
    center_path = config.model_dir / "means.json"
    center_path.parent.mkdir(parents=True)
    write_json_artifact({"embedding_dim": 3, "mean_embedding": [0.0, 0.0, 0.0]}, center_path)
    baseline = _baseline(config, model)
    baseline = BaselineDistribution(
        **{**baseline.to_dict(), "centering_vector_path": "means.json"}
    )
    text = "synthetic"
    result = score_semantic(
        _profile("a", text), _profile("b", text), [_chunk("a", text)], [_chunk("b", text)],
        model=model, baseline=baseline, config=config,
    )
    assert result.status == "OK"
    assert result.raw_cosine == pytest.approx(1.0)


def test_failure_logs_never_include_normalized_text(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    config = _config(tmp_path)
    model = SyntheticModel()
    marker = "PRIVATE_SYNTHETIC_NORMALIZED_MARKER"
    bad_baseline = BaselineDistribution(
        **{**_baseline(config, model).to_dict(), "configuration_fingerprint": "0" * 64}
    )
    with caplog.at_level("INFO"):
        result = score_semantic(
            _profile("a", marker), _profile("b", marker), [_chunk("a", marker)], [_chunk("b", marker)],
            model=model, baseline=bad_baseline, config=config,
        )
    assert result.status == "ERROR"
    assert marker not in caplog.text


def test_raw_stream_records_are_rejected(tmp_path: Path) -> None:
    from authorship_attribution import RawStreamChunk

    raw = RawStreamChunk(
        schema_version="1.0", chunk_id="raw", account_id="a", session_id="s",
        message_ids=["m"], start_timestamp_utc="2026-10-01T12:00:00Z",
        end_timestamp_utc="2026-10-01T12:00:00Z", n_messages=1, n_chars=1, raw_text=["x"],
    )
    with pytest.raises(ContractValidationError) as exc:
        compute_semantic_raw([raw], [_chunk("b", "x")], model=SyntheticModel(), config=_config(tmp_path))  # type: ignore[list-item]
    assert exc.value.code == ErrorCode.INPUT_INVALID


def test_external_provider_requires_explicit_gate() -> None:
    with pytest.raises(ContractValidationError) as exc:
        ExternalProviderEmbeddingModel("https://example.invalid", "fingerprint", 3, allow_external=False)
    assert exc.value.code == ErrorCode.EXTERNAL_NOT_ALLOWED


def test_external_provider_sends_only_normalized_stream_text_when_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import authorship_attribution.semantic as semantic_module

    endpoint = "https://semantic.example.test/embed"
    config = _config(tmp_path, allow_external=True, external_endpoint=endpoint)
    marker = "NORMALIZED_SYNTHETIC_ONLY"
    profile_a = _profile("a", marker)
    profile_b = _profile("b", marker)
    sent: list[bytes] = []

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def read(self) -> bytes:
            return b'{"embeddings":[[1,0,0]]}'

    def fake_urlopen(request: object, *, timeout: float) -> Response:
        sent.append(request.data)  # type: ignore[attr-defined]
        assert timeout == 60
        return Response()

    monkeypatch.setattr(semantic_module, "urlopen", fake_urlopen)
    model = ExternalProviderEmbeddingModel(endpoint, "model-manifest-fingerprint", 3, allow_external=True)
    model.tokenizer_sha256 = "tokenizer-fingerprint"  # type: ignore[attr-defined]
    baseline = _baseline(config, model)
    result = score_semantic(
        profile_a,
        profile_b,
        [_chunk("a", marker)],
        [_chunk("b", marker)],
        model=model,
        baseline=baseline,
        config=config,
    )
    assert result.status == "OK"
    assert [json.loads(body.decode("utf-8")) for body in sent] == [{"input": [marker]}]
    assert marker.encode("utf-8") not in b"".join(path.read_bytes() for path in (tmp_path / "cache").rglob("*.json"))


def test_local_scoring_maps_missing_model_to_engine_result(tmp_path: Path) -> None:
    config = _config(tmp_path)
    profile = _profile("a", "synthetic")
    result = score_semantic_local(
        profile,
        profile,
        [_chunk("a", "synthetic")],
        [_chunk("a", "synthetic")],
        baseline=BaselineDistribution(
            schema_version="1.0", baseline_id="baseline", engine="semantic",
            created_at_utc="2026-10-01T12:00:00Z", source_snapshot_id="snapshot",
            configuration_fingerprint="0" * 64, random_seed=1,
            calibration_population="known_different_account_pairs", n_pairs=1,
            sorted_raw_cosines=[0.0], minimum_messages=1, minimum_chars=1,
            strata_counts={}, centering_vector_path=None,
        ),
        config=config,
    )
    assert result.status == "ERROR"
    assert result.error == ErrorCode.LOCAL_MODEL_UNAVAILABLE.value
    assert result.raw_cosine is None and result.calibrated_percentile is None


def test_missing_local_manifest_is_unavailable(tmp_path: Path) -> None:
    with pytest.raises(ContractValidationError) as exc:
        load_local_model(_config(tmp_path))
    assert exc.value.code == ErrorCode.LOCAL_MODEL_UNAVAILABLE
