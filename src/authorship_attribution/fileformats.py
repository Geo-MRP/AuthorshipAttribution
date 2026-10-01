"""Strict readers and writers for shared on-disk contract formats."""

from __future__ import annotations

from dataclasses import dataclass, fields
import json
from pathlib import Path
import os
import re
import tempfile
from typing import Mapping, Self
import unicodedata

from .contracts import LabeledPair
from .errors import ContractValidationError, ErrorCode
from .fingerprints import canonical_json, sha256_hex


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LANG_RE = re.compile(r"^[A-Za-z0-9]{1,8}(?:-[A-Za-z0-9]{1,8})*$")


def _invalid(field_name: str, code: ErrorCode = ErrorCode.INPUT_INVALID) -> None:
    raise ContractValidationError(field_name, code)


def _term_list_payload(entries: list[str], lang: str | None) -> bytes:
    lines: list[str] = []
    if lang is not None:
        lines.append(f"# lang: {lang}")
    lines.extend(entries)
    return "\n".join(lines).encode("utf-8")


@dataclass(frozen=True, slots=True)
class TermList:
    """A normalized term list and the digest of its semantic content."""

    entries: list[str]
    sha256: str
    lang: str | None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if type(self.entries) is not list:
            _invalid("entries")
        if any(not isinstance(entry, str) or not entry for entry in self.entries):
            _invalid("entries")
        normalized = [unicodedata.normalize("NFC", entry) for entry in self.entries]
        stripped = [entry.strip() for entry in normalized]
        if (
            self.entries != normalized
            or self.entries != stripped
            or self.entries != sorted(set(self.entries))
        ):
            _invalid("entries")
        if not isinstance(self.sha256, str) or not _SHA256_RE.fullmatch(self.sha256):
            _invalid("sha256")
        if self.lang is not None and (
            not isinstance(self.lang, str) or not _LANG_RE.fullmatch(self.lang)
        ):
            _invalid("lang")
        if self.sha256 != sha256_hex(_term_list_payload(self.entries, self.lang)):
            _invalid("sha256")

    def to_dict(self) -> dict[str, object]:
        self._validate()
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        if not isinstance(data, Mapping) or set(data) != {"entries", "sha256", "lang"}:
            _invalid("object")
        entries = data["entries"]
        if type(entries) is not list or any(not isinstance(entry, str) for entry in entries):
            _invalid("entries")
        sha256 = data["sha256"]
        if not isinstance(sha256, str):
            _invalid("sha256")
        lang = data["lang"]
        if lang is not None and not isinstance(lang, str):
            _invalid("lang")
        return cls(entries=list(entries), sha256=sha256, lang=lang)


def load_term_list(path: Path, *, require_lang: bool) -> TermList:
    """Load a normalized UTF-8 term list without exposing entries in errors."""

    if not isinstance(path, Path) or type(require_lang) is not bool:
        _invalid("path")
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        _invalid("path")
    entries: list[str] = []
    lang: str | None = None
    for raw_line in source.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("# lang:"):
            if not line.startswith("# lang: "):
                _invalid("lang")
            candidate = line[len("# lang: ") :].strip()
            if lang is not None or not _LANG_RE.fullmatch(candidate):
                _invalid("lang")
            lang = candidate
            continue
        if line.startswith("#"):
            continue
        entries.append(unicodedata.normalize("NFC", line))
    if require_lang and lang is None:
        _invalid("lang")
    entries = sorted(set(entries))
    digest = sha256_hex(_term_list_payload(entries, lang))
    return TermList(entries=entries, sha256=digest, lang=lang)


def load_labeled_pairs(path: Path) -> list[LabeledPair]:
    """Load, validate, deduplicate, and deterministically order NDJSON labels."""

    if not isinstance(path, Path):
        _invalid("path")
    try:
        source = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        _invalid("path")
    unique: set[LabeledPair] = set()
    labels_by_pair: dict[tuple[str, str], str] = {}
    for line in source.splitlines():
        if not line.strip():
            _invalid("line")
        try:
            raw = json.loads(line, parse_constant=lambda _: (_ for _ in ()).throw(ValueError))
        except (TypeError, ValueError, json.JSONDecodeError):
            _invalid("line")
        if not isinstance(raw, dict):
            _invalid("line")
        try:
            pair = LabeledPair.from_dict(raw)
        except ContractValidationError:
            _invalid("line")
        pair_key = (pair.account_id_a, pair.account_id_b)
        previous_label = labels_by_pair.get(pair_key)
        if previous_label is not None and previous_label != pair.label:
            _invalid("label")
        labels_by_pair[pair_key] = pair.label
        unique.add(pair)
    return sorted(
        unique,
        key=lambda pair: (
            pair.account_id_a,
            pair.account_id_b,
            pair.label,
            pair.source_snapshot_id,
        ),
    )


def write_json_artifact(obj: object, path: Path) -> None:
    """Write canonical JSON atomically using a sibling temporary file."""

    if not isinstance(path, Path):
        _invalid("path", ErrorCode.ARTIFACT_INVALID)
    try:
        payload = canonical_json(obj)
    except (TypeError, ValueError):
        _invalid("object", ErrorCode.ARTIFACT_INVALID)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, path)
    except OSError:
        _invalid("path", ErrorCode.ARTIFACT_INVALID)
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            except OSError:
                pass


def read_json_artifact(path: Path) -> dict[str, object]:
    """Read a UTF-8 JSON object artifact and reject non-finite constants."""

    if not isinstance(path, Path):
        _invalid("path", ErrorCode.ARTIFACT_INVALID)
    try:
        source = path.read_text(encoding="utf-8")
        value = json.loads(
            source,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError),
        )
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        _invalid("path", ErrorCode.ARTIFACT_INVALID)
    if type(value) is not dict:
        _invalid("object", ErrorCode.ARTIFACT_INVALID)
    try:
        canonical_json(value)
    except (TypeError, ValueError):
        _invalid("object", ErrorCode.ARTIFACT_INVALID)
    return value
