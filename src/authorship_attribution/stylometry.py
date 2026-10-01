"""Raw-stream-only classical stylometry and deterministic artifact fitting.

No normalized stream, model, or external service is used in this module.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
import hashlib
import logging
import math
from pathlib import Path
import re
import time
from typing import TypeAlias
import unicodedata

from .calibration import calibrate_cosine, check_baseline_compat
from .contracts import (
    AccountProfile,
    BaselineDistribution,
    EngineResult,
    RawStreamChunk,
    StylometryArtifact,
    StylometryConfig,
)
from .errors import ContractValidationError, ErrorCode, error_code_for
from .fileformats import read_json_artifact, write_json_artifact
from .fingerprints import canonical_json, fingerprint, sha256_hex, stylometry_fingerprint


_LOGGER = logging.getLogger(__name__)
_HASH_SPACE_SIZE = 1 << 18
_PUNCTUATION = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
_FEATURE_SPEC = "raw-stylometry-v1"
_PREPROCESSING_SPEC = "raw-stream-preprocessing-v1"
_REPEATED_PUNCTUATION = re.compile(r"\.{3,}|…|([!\"#$%&'()*+,-./:;<=>?@\[\\\]^_`{|}~])\1+")

SparseVector: TypeAlias = dict[int, float]


def _function_words_data(function_words: Sequence[str]) -> tuple[list[str], str]:
    if isinstance(function_words, (str, bytes)):
        raise ContractValidationError("function_words", ErrorCode.INPUT_INVALID)
    if any(not isinstance(word, str) or not word for word in function_words):
        raise ContractValidationError("function_words", ErrorCode.INPUT_INVALID)
    entries = sorted({unicodedata.normalize("NFC", word.strip()) for word in function_words})
    if any(not entry for entry in entries):
        raise ContractValidationError("function_words", ErrorCode.INPUT_INVALID)
    words = sorted({word.casefold() for word in entries})
    return words, fingerprint({"function_words": entries, "language": "und"})


def _hash_bucket(ngram: str, seed: int, hash_space_size: int) -> int:
    key = hashlib.sha256(f"stylometry-ngram-v1:{seed}".encode("ascii")).digest()
    digest = hashlib.blake2b(
        ngram.encode("utf-8"), digest_size=8, key=key[:64], person=b"styl-ngram-v1"
    ).digest()
    return int.from_bytes(digest, "big", signed=False) % hash_space_size


def _messages(stream: Sequence[RawStreamChunk]) -> list[str]:
    texts: list[str] = []
    account_id: str | None = None
    for chunk in stream:
        if not isinstance(chunk, RawStreamChunk):
            raise ContractValidationError("raw_stream", ErrorCode.INPUT_INVALID)
        chunk._validate()
        if account_id is None:
            account_id = chunk.account_id
        elif account_id != chunk.account_id:
            raise ContractValidationError("account_id", ErrorCode.INPUT_INVALID)
        for text in chunk.raw_text:
            try:
                text.encode("utf-8", errors="strict")
            except UnicodeEncodeError:
                raise ContractValidationError("raw_text", ErrorCode.INPUT_INVALID) from None
            texts.append(text)
    return texts


def _ngram_counts(
    messages: Sequence[str], *, seed: int, hash_space_size: int, ngram_min: int, ngram_max: int
) -> Counter[int]:
    counts: Counter[int] = Counter()
    for message in messages:
        for n in range(ngram_min, ngram_max + 1):
            for offset in range(max(0, len(message) - n + 1)):
                counts[_hash_bucket(message[offset : offset + n], seed, hash_space_size)] += 1
    return counts


def _alpha_tokens(messages: Sequence[str]) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    for message in messages:
        for char in message:
            if char.isalpha():
                current.append(char)
            elif current:
                tokens.append("".join(current).casefold())
                current.clear()
        if current:
            tokens.append("".join(current).casefold())
            current.clear()
    return tokens


def _structural_features(messages: Sequence[str], function_words: Sequence[str]) -> list[float]:
    n_chars = sum(map(len, messages))
    denominator = max(1, n_chars)
    punctuation_counts = Counter(char for message in messages for char in message if char in _PUNCTUATION)
    double_spaces = sum(message[index : index + 2] == "  " for message in messages for index in range(max(0, len(message) - 1)))
    repeated_punctuation = sum(len(_REPEATED_PUNCTUATION.findall(message)) for message in messages)
    alpha_chars = [char for message in messages for char in message if char.isalpha()]
    uppercase = sum(char.isupper() for char in alpha_chars)
    elongations = 0
    emoji_bases = 0
    for message in messages:
        run_length = 0
        previous = ""
        for char in message:
            if char.isalpha() and char == previous:
                run_length += 1
            else:
                if run_length >= 3:
                    elongations += 1
                run_length = 1 if char.isalpha() else 0
                previous = char
            codepoint = ord(char)
            if (
                0x1F300 <= codepoint <= 0x1FAFF
                or 0x2600 <= codepoint <= 0x27BF
                or 0x2300 <= codepoint <= 0x23FF
            ):
                emoji_bases += 1
        if run_length >= 3:
            elongations += 1

    tokens = _alpha_tokens(messages)
    token_counts = Counter(tokens)
    total_tokens = max(1, len(tokens))
    total_messages = max(1, len(messages))
    return [
        double_spaces / denominator,
        *(punctuation_counts[char] / denominator for char in _PUNCTUATION),
        repeated_punctuation / denominator,
        uppercase / max(1, len(alpha_chars)),
        elongations / denominator,
        emoji_bases / denominator,
        n_chars / total_messages,
        *(token_counts[word] / total_tokens for word in function_words),
    ]


def _baseline_identity(accounts: Sequence[tuple[str, Sequence[str]]]) -> str:
    identity = []
    for account_id, messages in accounts:
        digest = hashlib.sha256()
        for message in messages:
            encoded = message.encode("utf-8", errors="strict")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        identity.append((account_id, len(messages), sum(map(len, messages)), digest.hexdigest()))
    return fingerprint({"accounts": identity})


def _configuration_fingerprint(
    artifact: StylometryArtifact,
    payload: dict[str, object],
    config: StylometryConfig,
) -> str:
    words_hash = payload.get("function_words_sha256")
    words_lang = payload.get("function_words_lang")
    preprocessing_fp = payload.get("preprocessing_fingerprint")
    if not all(isinstance(value, str) for value in (words_hash, words_lang, preprocessing_fp)):
        raise ContractValidationError("payload", ErrorCode.ARTIFACT_INVALID)
    return stylometry_fingerprint(
        config,
        words_hash,
        words_lang,
        artifact.artifact_sha256,
        preprocessing_fp,
    )


def _load_payload(artifact: StylometryArtifact, config: StylometryConfig) -> dict[str, object]:
    if not isinstance(artifact, StylometryArtifact) or not isinstance(config, StylometryConfig):
        raise ContractValidationError("artifact", ErrorCode.INPUT_INVALID)
    try:
        artifact._validate()
        payload = read_json_artifact(Path(artifact.payload_path))
        if sha256_hex(canonical_json(payload)) != artifact.artifact_sha256:
            raise ContractValidationError("artifact_sha256", ErrorCode.ARTIFACT_INVALID)
        if artifact.feature_spec_version != config.feature_spec_version:
            raise ContractValidationError("configuration", ErrorCode.BASELINE_CONFIG_MISMATCH)
        if artifact.hash_space_size != _HASH_SPACE_SIZE or config.hash_space_size != _HASH_SPACE_SIZE:
            raise ContractValidationError("hash_space_size", ErrorCode.BASELINE_CONFIG_MISMATCH)
        if payload.get("hash_space_size") != artifact.hash_space_size:
            raise ContractValidationError("hash_space_size", ErrorCode.ARTIFACT_INVALID)
        if payload.get("ngram_min") != config.ngram_min or payload.get("ngram_max") != config.ngram_max:
            raise ContractValidationError("ngram_range", ErrorCode.BASELINE_CONFIG_MISMATCH)
        expected = _configuration_fingerprint(artifact, payload, config)
        if expected != artifact.configuration_fingerprint:
            raise ContractValidationError("configuration", ErrorCode.BASELINE_CONFIG_MISMATCH)
        if payload.get("seed") != artifact.seed:
            raise ContractValidationError("seed", ErrorCode.ARTIFACT_INVALID)
        if payload.get("function_words_sha256") != artifact.function_words_sha256:
            raise ContractValidationError("function_words_sha256", ErrorCode.ARTIFACT_INVALID)
        if payload.get("function_words_lang") != artifact.function_words_lang:
            raise ContractValidationError("function_words_lang", ErrorCode.ARTIFACT_INVALID)
        if payload.get("structural_means") != artifact.structural_means:
            raise ContractValidationError("structural_means", ErrorCode.ARTIFACT_INVALID)
        if payload.get("structural_stds") != artifact.structural_stds:
            raise ContractValidationError("structural_stds", ErrorCode.ARTIFACT_INVALID)
        return payload
    except ContractValidationError:
        raise
    except (OSError, TypeError, ValueError, KeyError):
        raise ContractValidationError("artifact", ErrorCode.ARTIFACT_INVALID) from None


def fit_stylometry_artifact(
    baseline_streams: Iterable[Sequence[RawStreamChunk]],
    *,
    function_words: Sequence[str],
    config: StylometryConfig,
    seed: int,
    output_dir: Path,
) -> StylometryArtifact:
    """Fit hashed n-gram IDF and structural statistics from baseline raw streams."""

    if not isinstance(config, StylometryConfig) or not isinstance(output_dir, Path) or type(seed) is not int:
        raise ContractValidationError("input", ErrorCode.INPUT_INVALID)
    config._validate()
    if config.hash_space_size != _HASH_SPACE_SIZE:
        raise ContractValidationError("hash_space_size", ErrorCode.INPUT_INVALID)
    if config.ngram_min != 2 or config.ngram_max != 4:
        raise ContractValidationError("ngram_range", ErrorCode.INPUT_INVALID)
    words, words_sha256 = _function_words_data(function_words)
    accounts: list[tuple[str, list[str]]] = []
    for stream in baseline_streams:
        messages = _messages(stream)
        account_ids = sorted({chunk.account_id for chunk in stream})
        if len(account_ids) != 1:
            raise ContractValidationError("account_id", ErrorCode.INPUT_INVALID)
        accounts.append((account_ids[0], messages))
    accounts.sort(key=lambda item: item[0])
    if not accounts or len({account_id for account_id, _ in accounts}) != len(accounts):
        raise ContractValidationError("baseline_streams", ErrorCode.INPUT_INVALID)

    eligible = [
        (account_id, messages)
        for account_id, messages in accounts
        if len(messages) >= config.minimum_messages
        and sum(map(len, messages)) >= config.minimum_chars
    ]
    if not eligible:
        raise ContractValidationError("baseline_streams", ErrorCode.MINIMUMS_NOT_MET)

    document_frequency: Counter[int] = Counter()
    for _, messages in eligible:
        for message in messages:
            per_document = _ngram_counts(
                [message], seed=seed, hash_space_size=_HASH_SPACE_SIZE,
                ngram_min=config.ngram_min, ngram_max=config.ngram_max,
            )
            document_frequency.update(per_document.keys())
    n_documents = sum(len(messages) for _, messages in eligible)
    default_idf = math.log((1 + n_documents) / 1) + 1.0
    idf = {
        str(index): math.log((1 + n_documents) / (1 + frequency)) + 1.0
        for index, frequency in sorted(document_frequency.items())
    }

    structural_rows = [
        _structural_features(messages, words)[:38] for _, messages in eligible
    ]
    dimensions = len(structural_rows[0])
    means = [math.fsum(row[column] for row in structural_rows) / len(structural_rows) for column in range(dimensions)]
    stds = [
        math.sqrt(math.fsum((row[column] - means[column]) ** 2 for row in structural_rows) / len(structural_rows))
        for column in range(dimensions)
    ]
    preprocessing_fp = fingerprint(
        {
            "preprocessing_spec": _PREPROCESSING_SPEC,
            "baseline_corpus_identity": _baseline_identity(eligible),
            "minimum_messages": config.minimum_messages,
            "minimum_chars": config.minimum_chars,
            "feature_definitions": _FEATURE_SPEC,
        }
    )
    payload: dict[str, object] = {
        "feature_spec_version": config.feature_spec_version,
        "hash_space_size": _HASH_SPACE_SIZE,
        "ngram_min": config.ngram_min,
        "ngram_max": config.ngram_max,
        "seed": seed,
        "n_documents": n_documents,
        "default_idf": default_idf,
        "idf": idf,
        "function_words": words,
        "function_words_sha256": words_sha256,
        "function_words_lang": "und",
        "structural_means": means,
        "structural_stds": stds,
        "preprocessing_fingerprint": preprocessing_fp,
        "baseline_corpus_identity": _baseline_identity(eligible),
        "baseline_account_count": len(eligible),
    }
    payload_bytes = canonical_json(payload)
    artifact_sha256 = sha256_hex(payload_bytes)
    configuration_fp = stylometry_fingerprint(
        config, words_sha256, "und", artifact_sha256, preprocessing_fp
    )
    artifact_id = f"stylometry-{artifact_sha256}"
    payload_path = output_dir / f"{artifact_id}.json"
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json_artifact(payload, payload_path)
    except OSError:
        raise ContractValidationError("output_dir", ErrorCode.ARTIFACT_INVALID) from None
    except ContractValidationError as exc:
        if exc.code == ErrorCode.ARTIFACT_INVALID:
            raise
        raise ContractValidationError("artifact", ErrorCode.ARTIFACT_INVALID) from None
    return StylometryArtifact(
        artifact_id=artifact_id,
        artifact_sha256=artifact_sha256,
        feature_spec_version=config.feature_spec_version,
        hash_space_size=_HASH_SPACE_SIZE,
        function_words_sha256=words_sha256,
        function_words_lang="und",
        structural_means=means,
        structural_stds=stds,
        payload_path=str(payload_path),
        seed=seed,
        configuration_fingerprint=configuration_fp,
    )


def _sparse_unit(vector: SparseVector) -> SparseVector:
    norm = math.sqrt(math.fsum(value * value for value in vector.values()))
    if norm == 0.0:
        return {}
    return {index: value / norm for index, value in vector.items() if value != 0.0}


def _account_vector(
    stream: Sequence[RawStreamChunk], *, payload: dict[str, object], config: StylometryConfig
) -> SparseVector:
    messages = _messages(stream)
    ngrams = _ngram_counts(
        messages, seed=int(payload["seed"]), hash_space_size=_HASH_SPACE_SIZE,
        ngram_min=config.ngram_min, ngram_max=config.ngram_max,
    )
    idf_data = payload["idf"]
    if not isinstance(idf_data, dict):
        raise ContractValidationError("idf", ErrorCode.ARTIFACT_INVALID)
    char_block: SparseVector = {}
    total_ngrams = sum(ngrams.values())
    if total_ngrams:
        for index, count in sorted(ngrams.items()):
            weight = (count / total_ngrams) * float(idf_data.get(str(index), payload["default_idf"]))
            if weight:
                char_block[index] = weight
    char_block = _sparse_unit(char_block)

    words = payload["function_words"]
    means = payload["structural_means"]
    stds = payload["structural_stds"]
    if not isinstance(words, list) or not isinstance(means, list) or not isinstance(stds, list):
        raise ContractValidationError("statistics", ErrorCode.ARTIFACT_INVALID)
    all_features = _structural_features(messages, words)
    raw_structural = all_features[:38]
    structural_block: SparseVector = {}
    for index, (value, mean, std) in enumerate(zip(raw_structural, means, stds, strict=True)):
        standardized = 0.0 if float(std) == 0.0 else (value - float(mean)) / float(std)
        if standardized:
            structural_block[index] = standardized
    structural_block = _sparse_unit(structural_block)

    tokens = _alpha_tokens(messages)
    token_counts = Counter(tokens)
    denominator = max(1, len(tokens))
    function_block = _sparse_unit(
        {index: token_counts[word] / denominator for index, word in enumerate(words) if token_counts[word]}
    )

    # Blocks occupy disjoint coordinates and are scaled by their configured
    # weight before the final account-vector normalization.
    result: SparseVector = {}
    offset = 0
    for name, block, width in (
        ("char_ngrams", char_block, _HASH_SPACE_SIZE),
        ("structural", structural_block, len(raw_structural)),
        ("function_words", function_block, len(words)),
    ):
        scale = config.block_weights[name]
        for index, value in block.items():
            result[offset + index] = value * scale
        offset += width
    return _sparse_unit(result)


def compute_stylometry_raw(
    candidate_stream: Sequence[RawStreamChunk],
    suspect_stream: Sequence[RawStreamChunk],
    *,
    artifact: StylometryArtifact,
    config: StylometryConfig,
) -> float:
    """Return dot product cosine for sparse raw-only feature vectors."""

    payload = _load_payload(artifact, config)
    candidate = _account_vector(candidate_stream, payload=payload, config=config)
    suspect = _account_vector(suspect_stream, payload=payload, config=config)
    if len(candidate) > len(suspect):
        candidate, suspect = suspect, candidate
    cosine = math.fsum(value * suspect.get(index, 0.0) for index, value in sorted(candidate.items()))
    return min(1.0, max(-1.0, cosine))


def score_stylometry(
    candidate: AccountProfile,
    suspect: AccountProfile,
    candidate_stream: Sequence[RawStreamChunk],
    suspect_stream: Sequence[RawStreamChunk],
    *,
    artifact: StylometryArtifact,
    baseline: BaselineDistribution,
    config: StylometryConfig,
) -> EngineResult:
    """Score a pair and return calibrated output only for a compatible baseline."""

    if not isinstance(candidate, AccountProfile) or not isinstance(suspect, AccountProfile):
        raise ContractValidationError("profile", ErrorCode.INPUT_INVALID)
    counts_messages = {"candidate": candidate.n_messages, "suspect": suspect.n_messages}
    counts_chars = {"candidate": candidate.n_chars, "suspect": suspect.n_chars}
    started = time.monotonic()

    def result(status: str, error: ErrorCode, raw: float | None = None, percentile: float | None = None) -> EngineResult:
        return EngineResult(
            engine="stylometry",
            raw_cosine=raw,
            calibrated_percentile=percentile,
            status=status,  # type: ignore[arg-type]
            n_messages=counts_messages,
            n_chars=counts_chars,
            error=None if status == "OK" else error.value,
        )

    if any(
        profile.status == "INSUFFICIENT_DATA"
        or profile.n_messages < config.minimum_messages
        or profile.n_chars < config.minimum_chars
        for profile in (candidate, suspect)
    ):
        return result("INSUFFICIENT_DATA", ErrorCode.MINIMUMS_NOT_MET)
    try:
        if baseline.minimum_messages != config.minimum_messages or baseline.minimum_chars != config.minimum_chars:
            return result("ERROR", ErrorCode.BASELINE_CONFIG_MISMATCH)
        payload = _load_payload(artifact, config)
        incompatibility = check_baseline_compat(
            baseline, "stylometry", artifact.configuration_fingerprint
        )
        if incompatibility is not None:
            return result("ERROR", incompatibility)
        for profile, stream in ((candidate, candidate_stream), (suspect, suspect_stream)):
            texts = _messages(stream)
            if (
                profile.account_id != (stream[0].account_id if stream else None)
                or len(texts) != profile.n_messages
                or sum(map(len, texts)) != profile.n_chars
            ):
                return result("ERROR", ErrorCode.INPUT_INVALID)
        raw = compute_stylometry_raw(
            candidate_stream, suspect_stream, artifact=artifact, config=config
        )
        percentile = calibrate_cosine(raw, baseline)
        _LOGGER.info(
            "stylometry_status=OK baseline_id=%s candidate_messages=%d suspect_messages=%d elapsed_seconds=%.6f",
            baseline.baseline_id, candidate.n_messages, suspect.n_messages,
            time.monotonic() - started,
        )
        return result("OK", ErrorCode.ENGINE_FAILURE, raw, percentile)
    except Exception as exc:
        error = error_code_for(exc)
        if error == ErrorCode.ENGINE_FAILURE:
            error = ErrorCode.ENGINE_FAILURE
        _LOGGER.info(
            "stylometry_status=ERROR error=%s baseline_id=%s candidate_messages=%d suspect_messages=%d elapsed_seconds=%.6f",
            error.value, getattr(baseline, "baseline_id", "unknown"),
            candidate.n_messages, suspect.n_messages, time.monotonic() - started,
        )
        return result("ERROR", error)


__all__ = [
    "StylometryArtifact",
    "StylometryConfig",
    "calibrate_cosine",
    "compute_stylometry_raw",
    "fit_stylometry_artifact",
    "score_stylometry",
]
