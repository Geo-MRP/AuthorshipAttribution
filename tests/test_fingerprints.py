from __future__ import annotations

from pathlib import Path

import pytest

from authorship_attribution.contracts import PreprocessConfig, SemanticConfig, StylometryConfig
from authorship_attribution.fingerprints import (
    canonical_json,
    fingerprint,
    preprocessing_fingerprint,
    semantic_fingerprint,
    stylometry_fingerprint,
)


_DIGEST_A = "a" * 64
_DIGEST_B = "b" * 64


def test_canonical_json_and_fingerprint_ignore_mapping_order() -> None:
    first = {"z": [1, 2], "a": {"q": True}}
    second = {"a": {"q": True}, "z": [1, 2]}

    assert canonical_json(first) == b'{"a":{"q":true},"z":[1,2]}'
    assert canonical_json(first) == canonical_json(second)
    assert fingerprint(first) == fingerprint(second)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_canonical_json_rejects_non_finite_values(value: float) -> None:
    with pytest.raises(ValueError):
        canonical_json({"number": value})


def test_preprocessing_fingerprint_includes_behavior_excludes_cache_retention() -> None:
    stable = PreprocessConfig(cache_retention_days=1)
    cache_changed = PreprocessConfig(cache_retention_days=99)
    behavior_changed = PreprocessConfig(chunk_target_value=999)

    initial = preprocessing_fingerprint(stable, _DIGEST_A, _DIGEST_B)
    assert initial == preprocessing_fingerprint(cache_changed, _DIGEST_A, _DIGEST_B)
    assert initial != preprocessing_fingerprint(behavior_changed, _DIGEST_A, _DIGEST_B)
    assert initial != preprocessing_fingerprint(stable, _DIGEST_B, _DIGEST_B)


def test_stylometry_fingerprint_changes_with_included_components() -> None:
    config = StylometryConfig()
    initial = stylometry_fingerprint(config, _DIGEST_A, "en", _DIGEST_B, _DIGEST_A)

    assert initial != stylometry_fingerprint(config, _DIGEST_B, "en", _DIGEST_B, _DIGEST_A)
    assert initial != stylometry_fingerprint(config, _DIGEST_A, "fr", _DIGEST_B, _DIGEST_A)
    assert initial != stylometry_fingerprint(config, _DIGEST_A, "en", _DIGEST_A, _DIGEST_A)
    assert initial != stylometry_fingerprint(
        StylometryConfig(hash_space_size=512), _DIGEST_A, "en", _DIGEST_B, _DIGEST_A
    )


def test_semantic_fingerprint_excludes_hardware_and_cache_settings() -> None:
    base = SemanticConfig(model_dir=Path("model-a"), cache_dir=Path("cache-a"))
    excluded_changed = SemanticConfig(
        model_dir=Path("model-b"),
        cache_dir=Path("cache-b"),
        device="cuda",
        batch_size_cpu=1,
        batch_size_cuda=128,
        cache_retention_days=99,
        allow_external=True,
        external_endpoint="https://example.invalid",
    )
    included_changed = SemanticConfig(
        model_dir=Path("model-a"), cache_dir=Path("cache-a"), max_tokens=512
    )

    initial = semantic_fingerprint(base, _DIGEST_A, _DIGEST_B, _DIGEST_A)
    assert initial == semantic_fingerprint(excluded_changed, _DIGEST_A, _DIGEST_B, _DIGEST_A)
    assert initial != semantic_fingerprint(included_changed, _DIGEST_A, _DIGEST_B, _DIGEST_A)
