from __future__ import annotations

import json
from pathlib import Path

import pytest

from authorship_attribution.errors import ContractValidationError, ErrorCode
from authorship_attribution.fileformats import (
    TermList,
    load_labeled_pairs,
    load_term_list,
    read_json_artifact,
    write_json_artifact,
)
from authorship_attribution.fingerprints import canonical_json


def _pair(account_a: str, account_b: str, label: str, source: str = "snapshot") -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "account_id_a": account_a,
        "account_id_b": account_b,
        "label": label,
        "source_snapshot_id": source,
    }


def test_load_term_list_normalizes_comments_duplicates_and_language(tmp_path: Path) -> None:
    path = tmp_path / "terms.txt"
    path.write_text("# comment\n# lang: en-US\n\nCafe\u0301\nCafé\n beta \n", encoding="utf-8")

    terms = load_term_list(path, require_lang=True)

    assert terms.entries == ["Café", "beta"]
    assert terms.lang == "en-US"
    assert terms == TermList.from_dict(terms.to_dict())


def test_term_list_contract_rejects_unstripped_entries() -> None:
    with pytest.raises(ContractValidationError):
        TermList(entries=[" alpha"], sha256="a" * 64, lang=None)


def test_term_list_requires_language_and_rejects_invalid_utf8(tmp_path: Path) -> None:
    no_language = tmp_path / "plain.txt"
    no_language.write_text("alpha\n", encoding="utf-8")
    with pytest.raises(ContractValidationError) as missing_language:
        load_term_list(no_language, require_lang=True)
    assert missing_language.value.code is ErrorCode.INPUT_INVALID

    invalid = tmp_path / "invalid.txt"
    invalid.write_bytes(b"\xff\xfe")
    with pytest.raises(ContractValidationError) as invalid_utf8:
        load_term_list(invalid, require_lang=False)
    assert invalid_utf8.value.code is ErrorCode.INPUT_INVALID


def test_load_labeled_pairs_deduplicates_canonical_pairs_and_is_order_independent(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.ndjson"
    second = tmp_path / "second.ndjson"
    lines = [
        json.dumps(_pair("z", "a", "same")),
        json.dumps(_pair("b", "c", "different")),
        json.dumps(_pair("a", "z", "same")),
    ]
    first.write_text("\n".join(lines), encoding="utf-8")
    second.write_text("\n".join(reversed(lines)), encoding="utf-8")

    left = load_labeled_pairs(first)
    right = load_labeled_pairs(second)

    assert left == right
    assert [(pair.account_id_a, pair.account_id_b) for pair in left] == [("a", "z"), ("b", "c")]


def test_load_labeled_pairs_rejects_self_pairs_and_conflicting_labels(tmp_path: Path) -> None:
    self_pair = tmp_path / "self.ndjson"
    self_pair.write_text(json.dumps(_pair("a", "a", "same")), encoding="utf-8")
    with pytest.raises(ContractValidationError):
        load_labeled_pairs(self_pair)

    conflict = tmp_path / "conflict.ndjson"
    conflict.write_text(
        "\n".join(
            [json.dumps(_pair("a", "b", "same")), json.dumps(_pair("b", "a", "different"))]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ContractValidationError):
        load_labeled_pairs(conflict)


def test_json_artifact_is_canonical_and_atomic_at_interface_level(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    payload = {"z": [2, 1], "a": "value"}

    write_json_artifact(payload, path)

    assert path.read_bytes() == canonical_json(payload)
    assert read_json_artifact(path) == {"a": "value", "z": [2, 1]}


def test_json_artifact_rejects_non_object_and_nonfinite_constants(tmp_path: Path) -> None:
    array = tmp_path / "array.json"
    array.write_text("[]", encoding="utf-8")
    with pytest.raises(ContractValidationError):
        read_json_artifact(array)

    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"number":NaN}', encoding="utf-8")
    with pytest.raises(ContractValidationError):
        read_json_artifact(nonfinite)
