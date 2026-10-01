"""Canonical serialization and reproducibility fingerprints."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import is_dataclass
import hashlib
import json
import math
from pathlib import Path

from .contracts import PreprocessConfig, SemanticConfig, StylometryConfig


NORMALIZATION_SPEC_VERSION = "1.0"


def _json_value(value: object) -> object:
    """Convert supported contract values into JSON values without lossy coercion."""

    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _json_value(value.to_dict())
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite JSON number")
        return value
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, Mapping):
        normalized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("JSON object keys must be strings")
            normalized[key] = _json_value(item)
        return normalized
    if is_dataclass(value):
        raise TypeError("dataclass must provide to_dict")
    raise TypeError("object is not JSON serializable")


def canonical_json(obj: object) -> bytes:
    """Serialize an object as deterministic UTF-8 JSON.

    Keys are sorted, compact separators are used, and non-finite floats are
    rejected before encoding.
    """

    value = _json_value(obj)
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    """Return the lower-case SHA-256 hexadecimal digest for bytes."""

    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    return hashlib.sha256(data).hexdigest()


def fingerprint(components: Mapping[str, object]) -> str:
    """Fingerprint named components independent of input mapping key order."""

    if not isinstance(components, Mapping):
        raise TypeError("components must be a mapping")
    return sha256_hex(canonical_json(components))


def preprocessing_fingerprint(
    config: PreprocessConfig,
    game_terms_sha256: str,
    usernames_sha256: str,
) -> str:
    """Fingerprint preprocessing behavior and its normalization term lists."""

    config._validate()
    return fingerprint(
        {
            "normalization_spec_version": NORMALIZATION_SPEC_VERSION,
            "idle_gap_minutes": config.idle_gap_minutes,
            "utc_hour_filter": config.utc_hour_filter,
            "chunk_target_mode": config.chunk_target_mode,
            "chunk_target_value": config.chunk_target_value,
            "minimum_messages": config.minimum_messages,
            "minimum_chars": config.minimum_chars,
            "game_terms_sha256": game_terms_sha256,
            "usernames_sha256": usernames_sha256,
        }
    )


def stylometry_fingerprint(
    config: StylometryConfig,
    function_words_sha256: str,
    function_words_lang: str,
    artifact_sha256: str,
    preprocessing_fp: str,
) -> str:
    """Fingerprint stylometry behavior, assets, and preprocessing contract."""

    config._validate()
    return fingerprint(
        {
            "feature_spec_version": config.feature_spec_version,
            "hash_space_size": config.hash_space_size,
            "ngram_min": config.ngram_min,
            "ngram_max": config.ngram_max,
            "block_weights": config.block_weights,
            "minimum_messages": config.minimum_messages,
            "minimum_chars": config.minimum_chars,
            "function_words_sha256": function_words_sha256,
            "function_words_lang": function_words_lang,
            "artifact_sha256": artifact_sha256,
            "preprocessing_fingerprint": preprocessing_fp,
        }
    )


def semantic_fingerprint(
    config: SemanticConfig,
    model_manifest_sha256: str,
    tokenizer_sha256: str,
    preprocessing_fp: str,
) -> str:
    """Fingerprint semantic behavior independent of execution hardware."""

    config._validate()
    return fingerprint(
        {
            "max_tokens": config.max_tokens,
            "center_embeddings": config.center_embeddings,
            "minimum_messages": config.minimum_messages,
            "minimum_chars": config.minimum_chars,
            "model_manifest_sha256": model_manifest_sha256,
            "tokenizer_sha256": tokenizer_sha256,
            "preprocessing_fingerprint": preprocessing_fp,
        }
    )
