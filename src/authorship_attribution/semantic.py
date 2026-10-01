"""Normalized-stream-only semantic embeddings and baseline scoring.

This module accepts only :class:`NormalizedStreamChunk` records. It deliberately
does not import or inspect raw-stream contracts or profile stream paths.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone
import hashlib
import json
import logging
import math
import os
from pathlib import Path, PurePosixPath
import re
import time
import unicodedata
from typing import Literal, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
from numpy.typing import NDArray

from .calibration import calibrate_cosine, check_baseline_compat
from .contracts import (
    AccountProfile,
    BaselineDistribution,
    EmbeddingModel,
    EngineResult,
    NormalizedStreamChunk,
    SemanticConfig,
)
from .errors import ContractValidationError, ErrorCode, error_code_for
from .fileformats import read_json_artifact, write_json_artifact
from .fingerprints import canonical_json, semantic_fingerprint, sha256_hex

_LOGGER = logging.getLogger(__name__)
_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
_DEFAULT_MODEL_DIR = Path("models/all-MiniLM-L6-v2-onnx")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class _Tokenizer:
    """BERT basic-tokenization and greedy WordPiece using a local vocabulary."""

    def __init__(self, vocab: dict[str, int], *, lowercase: bool) -> None:
        self.vocab = vocab
        self.lowercase = lowercase
        self.cls_id = self._required("[CLS]")
        self.sep_id = self._required("[SEP]")
        self.unk_id = self._required("[UNK]")
        self.pad_id = vocab.get("[PAD]", 0)

    def _required(self, token: str) -> int:
        try:
            return self.vocab[token]
        except KeyError:
            raise ContractValidationError("tokenizer_vocab", ErrorCode.LOCAL_MODEL_INVALID) from None

    @staticmethod
    def _is_cjk(char: str) -> bool:
        code = ord(char)
        return (
            0x4E00 <= code <= 0x9FFF
            or 0x3400 <= code <= 0x4DBF
            or 0x20000 <= code <= 0x2A6DF
            or 0x2A700 <= code <= 0x2B73F
            or 0x2B740 <= code <= 0x2B81F
            or 0x2B820 <= code <= 0x2CEAF
            or 0xF900 <= code <= 0xFAFF
            or 0x2F800 <= code <= 0x2FA1F
        )

    @staticmethod
    def _is_punctuation(char: str) -> bool:
        code = ord(char)
        return unicodedata.category(char).startswith("P") or (
            33 <= code <= 47 or 58 <= code <= 64 or 91 <= code <= 96 or 123 <= code <= 126
        )

    def _basic_tokens(self, text: str) -> list[tuple[str, int, int]]:
        tokens: list[tuple[str, int, int]] = []
        start: int | None = None
        for index, char in enumerate(text):
            category = unicodedata.category(char)
            if char.isspace() or category in ("Cc", "Cf"):
                if start is not None:
                    tokens.append((text[start:index], start, index))
                    start = None
                continue
            if self._is_cjk(char) or self._is_punctuation(char):
                if start is not None:
                    tokens.append((text[start:index], start, index))
                    start = None
                tokens.append((char, index, index + 1))
            elif start is None:
                start = index
        if start is not None:
            tokens.append((text[start:], start, len(text)))
        return tokens

    def _wordpieces(self, token: str) -> list[int]:
        value = token.lower() if self.lowercase else token
        if self.lowercase:
            value = unicodedata.normalize("NFD", value)
        cleaned: list[str] = []
        for char in value:
            category = unicodedata.category(char)
            if category in ("Mn", "Mc", "Me"):
                continue
            cleaned.append(char)
        value = "".join(cleaned)
        if not value:
            return []
        if len(value) > 100:
            return [self.unk_id]
        result: list[int] = []
        offset = 0
        while offset < len(value):
            end = len(value)
            match: int | None = None
            while end > offset:
                piece = value[offset:end]
                if offset:
                    piece = "##" + piece
                if piece in self.vocab:
                    match = self.vocab[piece]
                    break
                end -= 1
            if match is None:
                return [self.unk_id]
            result.append(match)
            offset = end
        return result

    def windows(self, text: str, max_tokens: int) -> list[tuple[str, list[int]]]:
        """Return exact source slices and token IDs, never truncating content."""

        if type(text) is not str:
            raise ContractValidationError("normalized_text", ErrorCode.INPUT_INVALID)
        try:
            text.encode("utf-8", errors="strict")
        except UnicodeEncodeError:
            raise ContractValidationError("normalized_text", ErrorCode.INPUT_INVALID) from None
        if max_tokens < 3:
            raise ContractValidationError("max_tokens", ErrorCode.INPUT_INVALID)
        capacity = max_tokens - 2
        basic = self._basic_tokens(text)
        units: list[tuple[int, int, list[int]]] = [
            (start, end, self._wordpieces(token)) for token, start, end in basic
        ]
        units = [unit for unit in units if unit[2]]
        if not units:
            return [(text, [self.cls_id, self.sep_id])]
        windows: list[tuple[str, list[int]]] = []
        index = 0
        text_cursor = 0
        while index < len(units):
            first = index
            count = 0
            while index < len(units) and count + len(units[index][2]) <= capacity:
                count += len(units[index][2])
                index += 1
            if index == first:
                # A single BERT basic token can expand to at most 100 pieces,
                # but configurations with a smaller window still must progress.
                # Encode the complete basic token as [UNK] to avoid truncation.
                start, end, _ = units[index]
                units[index] = (start, end, [self.unk_id])
                continue
            left = text_cursor
            right = units[index][0] if index < len(units) else len(text)
            content_ids = [token_id for _, _, ids in units[first:index] for token_id in ids]
            windows.append((text[left:right], [self.cls_id, *content_ids, self.sep_id]))
            text_cursor = right
        return windows


class EmbeddingCache:
    """Retention-bounded JSON cache containing vectors and non-text metadata."""

    def __init__(self, directory: Path, retention_days: int) -> None:
        self.directory = directory
        self.retention_days = retention_days
        self._created = datetime.now(timezone.utc)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._expire_old()

    def _expire_old(self) -> None:
        cutoff = self._created - timedelta(days=self.retention_days)
        for path in self.directory.rglob("*.json"):
            try:
                value = read_json_artifact(path)
                expiry = value.get("expires_at_utc")
                created = value.get("created_at_utc")
                if not isinstance(expiry, str) or not isinstance(created, str):
                    path.unlink(missing_ok=True)
                    continue
                expiry_dt = datetime.fromisoformat(expiry.replace("Z", "+00:00"))
                created_dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                if expiry_dt <= self._created or created_dt < cutoff:
                    path.unlink(missing_ok=True)
            except (OSError, ValueError, ContractValidationError):
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

    def get(self, key: str, model_fingerprint: str, token_hash: str, dimension: int) -> NDArray[np.float32] | None:
        path = self.directory / key[:2] / f"{key}.json"
        if not path.exists():
            return None
        try:
            record = read_json_artifact(path)
            vector = record.get("vector")
            expiry = record.get("expires_at_utc")
            if (
                record.get("cache_key") != key
                or record.get("model_fingerprint") != model_fingerprint
                or record.get("token_ids_sha256") != token_hash
                or not isinstance(vector, list)
                or len(vector) != dimension
                or not isinstance(expiry, str)
                or datetime.fromisoformat(expiry.replace("Z", "+00:00")) <= self._created
            ):
                path.unlink(missing_ok=True)
                return None
            result = np.asarray(vector, dtype=np.float32)
            if result.shape != (dimension,) or not np.isfinite(result).all():
                path.unlink(missing_ok=True)
                return None
            return result
        except (OSError, ValueError, TypeError, ContractValidationError):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
            return None

    def put(self, key: str, model_fingerprint: str, token_hash: str, vector: NDArray[np.float32]) -> None:
        if self.retention_days == 0:
            return
        expires = self._created + timedelta(days=self.retention_days)
        path = self.directory / key[:2] / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        write_json_artifact(
            {
                "cache_key": key,
                "model_fingerprint": model_fingerprint,
                "token_ids_sha256": token_hash,
                "vector": [float(value) for value in vector],
                "created_at_utc": self._created.isoformat(timespec="seconds").replace("+00:00", "Z"),
                "expires_at_utc": expires.isoformat(timespec="seconds").replace("+00:00", "Z"),
            },
            path,
        )

    def finish(self) -> None:
        if self.retention_days == 0:
            for path in sorted(self.directory.rglob("*"), reverse=True):
                try:
                    if path.is_file():
                        path.unlink(missing_ok=True)
                    elif path.is_dir():
                        path.rmdir()
                except OSError:
                    pass


class LocalOnnxEmbeddingModel:
    """Verified local MiniLM ONNX model with WordPiece tokenization."""

    def __init__(self, session: object, tokenizer: _Tokenizer, model_fingerprint: str, tokenizer_sha256: str, dimension: int, max_tokens: int, batch_size: int) -> None:
        self._session = session
        self._tokenizer = tokenizer
        self.model_fingerprint = model_fingerprint
        self.tokenizer_sha256 = tokenizer_sha256
        self.embedding_dim = dimension
        self.max_tokens = max_tokens
        self.batch_size = batch_size

    def model_windows(self, text: str) -> list[tuple[str, list[int]]]:
        return self._tokenizer.windows(text, self.max_tokens)

    def token_ids_for_window(self, text: str) -> list[int]:
        windows = self.model_windows(text)
        if len(windows) != 1:
            raise ContractValidationError("model_window", ErrorCode.INPUT_INVALID)
        return windows[0][1]

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Embed already-windowed text using masked mean pooling."""

        rows: list[NDArray[np.float32]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            input_ids: list[list[int]] = []
            for text in batch:
                windows = self.model_windows(text)
                if len(windows) != 1:
                    raise ContractValidationError("model_window", ErrorCode.INPUT_INVALID)
                input_ids.append(windows[0][1])
            width = max(map(len, input_ids), default=2)
            ids = np.full((len(batch), width), self._tokenizer.pad_id, dtype=np.int64)
            masks = np.zeros((len(batch), width), dtype=np.int64)
            for row, values in enumerate(input_ids):
                ids[row, : len(values)] = values
                masks[row, : len(values)] = 1
            inputs: dict[str, NDArray[np.int64]] = {}
            for item in self._session.get_inputs():
                if item.name == "input_ids":
                    inputs[item.name] = ids
                elif item.name == "attention_mask":
                    inputs[item.name] = masks
                elif item.name == "token_type_ids":
                    inputs[item.name] = np.zeros_like(ids)
            outputs = self._session.run(None, inputs)
            hidden = np.asarray(outputs[0], dtype=np.float32)
            if hidden.ndim != 3 or hidden.shape[:2] != ids.shape or hidden.shape[2] != self.embedding_dim:
                raise ContractValidationError("model_output", ErrorCode.LOCAL_MODEL_INVALID)
            denominator = np.maximum(masks.sum(axis=1, keepdims=True), 1).astype(np.float32)
            pooled = (hidden * masks[:, :, None]).sum(axis=1) / denominator
            rows.extend(np.asarray(row, dtype=np.float32) for row in pooled)
        return np.stack(rows).astype(np.float32, copy=False) if rows else np.empty((0, self.embedding_dim), dtype=np.float32)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_local_model(config: SemanticConfig) -> LocalOnnxEmbeddingModel:
    """Load and verify local model files only; this function never downloads."""

    if not isinstance(config, SemanticConfig):
        raise ContractValidationError("config", ErrorCode.INPUT_INVALID)
    directory = config.model_dir
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ContractValidationError("manifest", ErrorCode.LOCAL_MODEL_UNAVAILABLE)
    try:
        manifest = read_json_artifact(manifest_path)
    except ContractValidationError:
        raise ContractValidationError("manifest", ErrorCode.LOCAL_MODEL_INVALID) from None
    try:
        if (
            manifest.get("model_id") != _MODEL_ID
            or manifest.get("embedding_dim") != 384
            or manifest.get("maximum_token_length") != 256
            or manifest.get("license") != "Apache-2.0"
        ):
            raise ContractValidationError("manifest", ErrorCode.LOCAL_MODEL_INVALID)
        files = manifest.get("model_files")
        tokenizer_name = manifest.get("tokenizer_vocab")
        tokenizer_hash = manifest.get("tokenizer_vocab_sha256")
        if not isinstance(files, dict) or not isinstance(tokenizer_name, str) or not isinstance(tokenizer_hash, str):
            raise ContractValidationError("manifest", ErrorCode.LOCAL_MODEL_INVALID)
        for filename, expected_hash in files.items():
            relative_file = PurePosixPath(filename) if isinstance(filename, str) else PurePosixPath("/")
            if (
                not isinstance(filename, str)
                or relative_file.is_absolute()
                or ".." in relative_file.parts
                or not isinstance(expected_hash, str)
                or not _SHA256_RE.fullmatch(expected_hash)
            ):
                raise ContractValidationError("manifest", ErrorCode.LOCAL_MODEL_INVALID)
            path = directory.joinpath(*relative_file.parts)
            if not path.is_file():
                raise ContractValidationError("model_file", ErrorCode.LOCAL_MODEL_UNAVAILABLE)
            if _file_sha256(path) != expected_hash:
                raise ContractValidationError("model_file", ErrorCode.LOCAL_MODEL_INVALID)
        vocab_relative = PurePosixPath(tokenizer_name)
        if vocab_relative.is_absolute() or ".." in vocab_relative.parts:
            raise ContractValidationError("tokenizer_vocab", ErrorCode.LOCAL_MODEL_INVALID)
        vocab_path = directory.joinpath(*vocab_relative.parts)
        if not vocab_path.is_file():
            raise ContractValidationError("tokenizer_vocab", ErrorCode.LOCAL_MODEL_UNAVAILABLE)
        vocab_bytes = vocab_path.read_bytes()
        if sha256_hex(vocab_bytes) != tokenizer_hash:
            raise ContractValidationError("tokenizer_vocab", ErrorCode.LOCAL_MODEL_INVALID)
        vocab_text = vocab_bytes.decode("utf-8", errors="strict")
        vocab_lines = vocab_text.splitlines()
        vocab = {token: index for index, token in enumerate(vocab_lines)}
        if len(vocab) != len(vocab_lines):
            raise ContractValidationError("tokenizer_vocab", ErrorCode.LOCAL_MODEL_INVALID)
        tokenizer = _Tokenizer(vocab, lowercase=manifest.get("do_lower_case") is True)
        model_filename = manifest.get("onnx_model")
        model_relative = PurePosixPath(model_filename) if isinstance(model_filename, str) else PurePosixPath("/")
        if not isinstance(model_filename, str) or model_filename not in files or model_relative.is_absolute() or ".." in model_relative.parts:
            raise ContractValidationError("onnx_model", ErrorCode.LOCAL_MODEL_INVALID)
        try:
            import onnxruntime as ort
        except ImportError:
            raise ContractValidationError("onnxruntime", ErrorCode.LOCAL_MODEL_UNAVAILABLE) from None
        providers = ort.get_available_providers()
        if config.device == "cuda" and "CUDAExecutionProvider" not in providers:
            raise ContractValidationError("device", ErrorCode.LOCAL_MODEL_UNAVAILABLE)
        provider = "CUDAExecutionProvider" if config.device == "cuda" or (config.device == "auto" and "CUDAExecutionProvider" in providers) else "CPUExecutionProvider"
        session = ort.InferenceSession(str(directory.joinpath(*model_relative.parts)), providers=[provider])
        manifest_fingerprint = sha256_hex(manifest_path.read_bytes())
        actual_max = min(config.max_tokens, 256)
        if config.max_tokens > 256:
            raise ContractValidationError("max_tokens", ErrorCode.INPUT_INVALID)
        batch_size = config.batch_size_cuda if provider == "CUDAExecutionProvider" else config.batch_size_cpu
        return LocalOnnxEmbeddingModel(session, tokenizer, manifest_fingerprint, tokenizer_hash, 384, actual_max, batch_size)
    except ContractValidationError:
        raise
    except (OSError, UnicodeDecodeError, TypeError, ValueError, KeyError):
        raise ContractValidationError("manifest", ErrorCode.LOCAL_MODEL_INVALID) from None


class ExternalProviderEmbeddingModel:
    """Explicitly gated HTTP embedding adapter using normalized text only.

    Expected response is ``{"embeddings": [[...], ...]}``; credentials are
    read from ``AUTHORSHIP_ATTRIBUTION_API_KEY`` at request time.
    """

    is_external_provider = True

    def __init__(self, endpoint: str, model_fingerprint: str, embedding_dim: int, *, allow_external: bool) -> None:
        if not allow_external:
            raise ContractValidationError("allow_external", ErrorCode.EXTERNAL_NOT_ALLOWED)
        self.endpoint = endpoint
        self.model_fingerprint = model_fingerprint
        self.embedding_dim = embedding_dim
        self._external_allowed = allow_external

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not self._external_allowed:
            raise ContractValidationError("allow_external", ErrorCode.EXTERNAL_NOT_ALLOWED)
        payload = canonical_json({"input": list(texts)})
        headers = {"Content-Type": "application/json"}
        credential = os.environ.get("AUTHORSHIP_ATTRIBUTION_API_KEY")
        if credential:
            headers["Authorization"] = f"Bearer {credential}"
        request = Request(self.endpoint, data=payload, headers=headers, method="POST")
        try:
            with urlopen(request, timeout=60) as response:
                body = json.loads(response.read().decode("utf-8", errors="strict"))
            vectors = body["embeddings"]
            result = np.asarray(vectors, dtype=np.float32)
            if result.shape != (len(texts), self.embedding_dim) or not np.isfinite(result).all():
                raise ValueError
            return result
        except (HTTPError, URLError, OSError, UnicodeDecodeError, ValueError, TypeError, KeyError):
            raise ContractValidationError("external_provider", ErrorCode.ENGINE_FAILURE) from None


def _windows(model: EmbeddingModel, text: str, max_tokens: int) -> list[tuple[str, list[int]]]:
    maker = getattr(model, "model_windows", None)
    if callable(maker):
        result = maker(text)
        if not isinstance(result, list) or any(not isinstance(window, tuple) or len(window) != 2 for window in result):
            raise ContractValidationError("model_window", ErrorCode.INPUT_INVALID)
        return cast(list[tuple[str, list[int]]], result)
    # Generic contract-compatible test/adapter models represent one input as
    # one model window and expose UTF-8 bytes as stable token identifiers.
    try:
        text.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        raise ContractValidationError("normalized_text", ErrorCode.INPUT_INVALID) from None
    return [(text, list(text.encode("utf-8")))]


def _account_vector(stream: Sequence[NormalizedStreamChunk], *, model: EmbeddingModel, config: SemanticConfig, cache: EmbeddingCache, centering_vector: NDArray[np.float32] | None) -> NDArray[np.float32]:
    windows: list[tuple[str, list[int]]] = []
    account_id: str | None = None
    for chunk in stream:
        if not isinstance(chunk, NormalizedStreamChunk):
            raise ContractValidationError("normalized_stream", ErrorCode.INPUT_INVALID)
        chunk._validate()
        if account_id is None:
            account_id = chunk.account_id
        elif chunk.account_id != account_id:
            raise ContractValidationError("account_id", ErrorCode.INPUT_INVALID)
        windows.extend(_windows(model, chunk.normalized_text, config.max_tokens))
    if not windows:
        raise ContractValidationError("normalized_stream", ErrorCode.MINIMUMS_NOT_MET)
    keys: list[tuple[str, str, str]] = []
    vectors: list[NDArray[np.float32] | None] = []
    missing_texts: list[str] = []
    missing_indexes: list[int] = []
    for text, token_ids in windows:
        text_hash = sha256_hex(text.encode("utf-8", errors="strict"))
        token_hash = sha256_hex(canonical_json(token_ids))
        key = sha256_hex(canonical_json({"text_sha256": text_hash, "token_ids_sha256": token_hash, "model_fingerprint": model.model_fingerprint}))
        keys.append((key, token_hash, text_hash))
        hit = cache.get(key, model.model_fingerprint, token_hash, model.embedding_dim)
        vectors.append(hit)
        if hit is None:
            missing_indexes.append(len(vectors) - 1)
            missing_texts.append(text)
    if missing_texts:
        embedded = np.asarray(model.embed(missing_texts), dtype=np.float32)
        if embedded.shape != (len(missing_texts), model.embedding_dim) or not np.isfinite(embedded).all():
            raise ContractValidationError("embedding_output", ErrorCode.ENGINE_FAILURE)
        for index, row in zip(missing_indexes, embedded, strict=True):
            vectors[index] = row.copy()
            key, token_hash, _ = keys[index]
            cache.put(key, model.model_fingerprint, token_hash, row)
    complete = [cast(NDArray[np.float32], item) for item in vectors]
    pooled = np.asarray([math.fsum(float(vector[column]) for vector in complete) / len(complete) for column in range(model.embedding_dim)], dtype=np.float64).astype(np.float32)
    if centering_vector is not None:
        if centering_vector.shape != (model.embedding_dim,):
            raise ContractValidationError("centering_vector", ErrorCode.BASELINE_INVALID)
        pooled = (pooled.astype(np.float64) - centering_vector.astype(np.float64)).astype(np.float32)
    norm = math.sqrt(math.fsum(float(value) * float(value) for value in pooled))
    if norm == 0.0 or not math.isfinite(norm):
        raise ContractValidationError("embedding_norm", ErrorCode.ENGINE_FAILURE)
    return np.asarray([float(value) / norm for value in pooled], dtype=np.float64).astype(np.float32)


def _cosine(left: NDArray[np.float32], right: NDArray[np.float32]) -> float:
    return min(1.0, max(-1.0, math.fsum(float(a) * float(b) for a, b in zip(left, right, strict=True))))


def compute_semantic_raw(candidate_stream: Sequence[NormalizedStreamChunk], suspect_stream: Sequence[NormalizedStreamChunk], *, model: EmbeddingModel, config: SemanticConfig) -> float:
    """Compute cosine from normalized chunks and a supplied embedding model."""

    if getattr(model, "is_external_provider", False) and (
        not config.allow_external
        or not getattr(model, "_external_allowed", False)
        or config.external_endpoint != getattr(model, "endpoint", None)
    ):
        raise ContractValidationError("allow_external", ErrorCode.EXTERNAL_NOT_ALLOWED)
    cache = EmbeddingCache(config.cache_dir, config.cache_retention_days)
    try:
        candidate = _account_vector(candidate_stream, model=model, config=config, cache=cache, centering_vector=None)
        suspect = _account_vector(suspect_stream, model=model, config=config, cache=cache, centering_vector=None)
        return _cosine(candidate, suspect)
    finally:
        cache.finish()


def _load_centering_vector(baseline: BaselineDistribution, config: SemanticConfig, dimension: int) -> NDArray[np.float32] | None:
    if not config.center_embeddings:
        return None
    if baseline.centering_vector_path is None:
        raise ContractValidationError("centering_vector_path", ErrorCode.BASELINE_INVALID)
    relative = Path(baseline.centering_vector_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ContractValidationError("centering_vector_path", ErrorCode.BASELINE_INVALID)
    path = config.model_dir / relative
    try:
        artifact = read_json_artifact(path)
    except ContractValidationError:
        raise ContractValidationError("centering_vector", ErrorCode.BASELINE_INVALID) from None
    values = artifact.get("mean_embedding")
    if artifact.get("embedding_dim") != dimension or not isinstance(values, list) or len(values) != dimension:
        raise ContractValidationError("centering_vector", ErrorCode.BASELINE_INVALID)
    try:
        result = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError):
        raise ContractValidationError("centering_vector", ErrorCode.BASELINE_INVALID) from None
    if not np.isfinite(result).all():
        raise ContractValidationError("centering_vector", ErrorCode.BASELINE_INVALID)
    return result


def score_semantic(candidate: AccountProfile, suspect: AccountProfile, candidate_stream: Sequence[NormalizedStreamChunk], suspect_stream: Sequence[NormalizedStreamChunk], *, model: EmbeddingModel, baseline: BaselineDistribution, config: SemanticConfig) -> EngineResult:
    """Return raw cosine and known-different baseline percentile if compatible."""

    if not isinstance(candidate, AccountProfile) or not isinstance(suspect, AccountProfile):
        raise ContractValidationError("profile", ErrorCode.INPUT_INVALID)
    counts_messages = {"candidate": candidate.n_messages, "suspect": suspect.n_messages}
    counts_chars = {"candidate": candidate.n_chars, "suspect": suspect.n_chars}

    def result(status: str, error: ErrorCode, raw: float | None = None, percentile: float | None = None) -> EngineResult:
        return EngineResult(engine="semantic", raw_cosine=raw, calibrated_percentile=percentile, status=cast(Literal["OK", "INSUFFICIENT_DATA", "TIMEOUT", "ERROR"], status), n_messages=counts_messages, n_chars=counts_chars, error=None if status == "OK" else error.value)

    if any(profile.status == "INSUFFICIENT_DATA" or profile.n_messages < config.minimum_messages or profile.n_chars < config.minimum_chars for profile in (candidate, suspect)):
        return result("INSUFFICIENT_DATA", ErrorCode.MINIMUMS_NOT_MET)
    started = time.monotonic()
    cache: EmbeddingCache | None = None
    try:
        if getattr(model, "is_external_provider", False) and (not config.allow_external or not getattr(model, "_external_allowed", False) or config.external_endpoint != getattr(model, "endpoint", None)):
            return result("ERROR", ErrorCode.EXTERNAL_NOT_ALLOWED)
        if candidate.preprocessing_fingerprint != suspect.preprocessing_fingerprint:
            return result("ERROR", ErrorCode.BASELINE_CONFIG_MISMATCH)
        manifest_hash = model.model_fingerprint
        tokenizer_hash = getattr(model, "tokenizer_sha256", manifest_hash)
        expected = semantic_fingerprint(config, manifest_hash, tokenizer_hash, candidate.preprocessing_fingerprint)
        if baseline.minimum_messages != config.minimum_messages or baseline.minimum_chars != config.minimum_chars:
            return result("ERROR", ErrorCode.BASELINE_CONFIG_MISMATCH)
        incompatibility = check_baseline_compat(baseline, "semantic", expected)
        if incompatibility is not None:
            return result("ERROR", incompatibility)
        for profile, stream in ((candidate, candidate_stream), (suspect, suspect_stream)):
            if any(not isinstance(chunk, NormalizedStreamChunk) for chunk in stream):
                return result("ERROR", ErrorCode.INPUT_INVALID)
            if (stream and any(chunk.account_id != profile.account_id for chunk in stream)) or sum(chunk.n_messages for chunk in stream) != profile.n_messages or sum(chunk.n_chars for chunk in stream) != profile.n_chars:
                return result("ERROR", ErrorCode.INPUT_INVALID)
        center = _load_centering_vector(baseline, config, model.embedding_dim)
        cache = EmbeddingCache(config.cache_dir, config.cache_retention_days)
        left = _account_vector(candidate_stream, model=model, config=config, cache=cache, centering_vector=center)
        right = _account_vector(suspect_stream, model=model, config=config, cache=cache, centering_vector=center)
        raw = _cosine(left, right)
        percentile = calibrate_cosine(raw, baseline)
        _LOGGER.info("semantic_status=OK baseline_id=%s model_fingerprint=%s candidate_messages=%d suspect_messages=%d elapsed_seconds=%.6f", baseline.baseline_id, model.model_fingerprint, candidate.n_messages, suspect.n_messages, time.monotonic() - started)
        return result("OK", ErrorCode.ENGINE_FAILURE, raw, percentile)
    except Exception as exc:
        code = error_code_for(exc)
        _LOGGER.info("semantic_status=ERROR error=%s baseline_id=%s candidate_messages=%d suspect_messages=%d elapsed_seconds=%.6f", code.value, baseline.baseline_id, candidate.n_messages, suspect.n_messages, time.monotonic() - started)
        return result("ERROR", code)
    finally:
        if cache is not None:
            cache.finish()


def score_semantic_local(
    candidate: AccountProfile,
    suspect: AccountProfile,
    candidate_stream: Sequence[NormalizedStreamChunk],
    suspect_stream: Sequence[NormalizedStreamChunk],
    *,
    baseline: BaselineDistribution,
    config: SemanticConfig,
) -> EngineResult:
    """Load the pinned local model and score, mapping asset failures to results."""

    counts_messages = {"candidate": candidate.n_messages, "suspect": suspect.n_messages}
    counts_chars = {"candidate": candidate.n_chars, "suspect": suspect.n_chars}

    def failure(status: Literal["INSUFFICIENT_DATA", "ERROR"], code: ErrorCode) -> EngineResult:
        return EngineResult(
            engine="semantic",
            raw_cosine=None,
            calibrated_percentile=None,
            status=status,
            n_messages=counts_messages,
            n_chars=counts_chars,
            error=code.value,
        )

    if any(
        profile.status == "INSUFFICIENT_DATA"
        or profile.n_messages < config.minimum_messages
        or profile.n_chars < config.minimum_chars
        for profile in (candidate, suspect)
    ):
        return failure("INSUFFICIENT_DATA", ErrorCode.MINIMUMS_NOT_MET)
    try:
        model = load_local_model(config)
    except Exception as exc:
        return failure("ERROR", error_code_for(exc))
    return score_semantic(
        candidate,
        suspect,
        candidate_stream,
        suspect_stream,
        model=model,
        baseline=baseline,
        config=config,
    )


__all__ = [
    "EmbeddingCache",
    "ExternalProviderEmbeddingModel",
    "LocalOnnxEmbeddingModel",
    "SemanticConfig",
    "calibrate_cosine",
    "compute_semantic_raw",
    "load_local_model",
    "score_semantic",
    "score_semantic_local",
]
