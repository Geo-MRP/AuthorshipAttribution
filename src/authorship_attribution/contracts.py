"""Shared data contracts and configuration objects.

All concrete contract objects validate themselves at construction and before
serialization. Validation errors intentionally contain only a field name and
stable error code so message content cannot leak through diagnostics.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, fields
from datetime import datetime
from enum import StrEnum
import json
import math
from pathlib import Path, PureWindowsPath
import re
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol, Self, TypedDict

from .errors import ContractValidationError, ErrorCode

if TYPE_CHECKING:
    import numpy


_SCHEMA_VERSION = "1.0"
_TIMESTAMP_RE = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$"
)
_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ENGINE_NAMES = frozenset(("stylometry", "semantic"))


def _invalid(field_name: str) -> None:
    raise ContractValidationError(field_name, ErrorCode.INPUT_INVALID)


def _checked_data(data: object, cls: type[object]) -> Mapping[str, object]:
    if not isinstance(data, Mapping):
        _invalid("object")
    expected = {item.name for item in fields(cls)}
    keys = list(data.keys())
    if any(not isinstance(key, str) for key in keys):
        _invalid("object")
    actual = set(keys)
    if actual - expected:
        _invalid("object")
    missing = expected - actual
    if missing:
        _invalid(sorted(missing)[0])
    return data


def _require_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        _invalid(field_name)
    if any(not isinstance(key, str) for key in value):
        _invalid(field_name)
    return value


def _require_dict(value: object, field_name: str) -> dict[str, object]:
    if type(value) is not dict:
        _invalid(field_name)
    if any(not isinstance(key, str) for key in value):
        _invalid(field_name)
    return value


def _require_str(value: object, field_name: str, *, nonempty: bool = False) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        _invalid(field_name)
    return value


def _require_optional_str(
    value: object, field_name: str, *, nonempty: bool = False
) -> str | None:
    if value is None:
        return None
    return _require_str(value, field_name, nonempty=nonempty)


def _require_bool(value: object, field_name: str) -> bool:
    if type(value) is not bool:
        _invalid(field_name)
    return value


def _require_int(
    value: object,
    field_name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if type(value) is not int:
        _invalid(field_name)
    if minimum is not None and value < minimum:
        _invalid(field_name)
    if maximum is not None and value > maximum:
        _invalid(field_name)
    return value


def _require_float(
    value: object,
    field_name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if type(value) is not float or not math.isfinite(value):
        _invalid(field_name)
    if minimum is not None and value < minimum:
        _invalid(field_name)
    if maximum is not None and value > maximum:
        _invalid(field_name)
    return value


def _require_optional_float(
    value: object,
    field_name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float | None:
    if value is None:
        return None
    return _require_float(value, field_name, minimum=minimum, maximum=maximum)


def _require_list(value: object, field_name: str) -> list[object]:
    if type(value) is not list:
        _invalid(field_name)
    return value


def _require_string_list(
    value: object,
    field_name: str,
    *,
    nonempty_items: bool = False,
) -> list[str]:
    items = _require_list(value, field_name)
    result: list[str] = []
    for item in items:
        result.append(_require_str(item, field_name, nonempty=nonempty_items))
    return result


def _require_float_list(
    value: object,
    field_name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> list[float]:
    items = _require_list(value, field_name)
    result: list[float] = []
    for item in items:
        result.append(
            _require_float(item, field_name, minimum=minimum, maximum=maximum)
        )
    return result


def _require_sorted_unique_strings(
    value: object, field_name: str, *, nonempty: bool = False
) -> list[str]:
    items = _require_string_list(value, field_name, nonempty_items=True)
    if nonempty and not items:
        _invalid(field_name)
    if items != sorted(set(items)):
        _invalid(field_name)
    return items


def _require_hour_list(value: object, field_name: str) -> list[int] | None:
    if value is None:
        return None
    items = _require_list(value, field_name)
    hours = [_require_int(item, field_name, minimum=0, maximum=23) for item in items]
    if hours != sorted(set(hours)):
        _invalid(field_name)
    return hours


def _require_timestamp(value: object, field_name: str) -> str:
    timestamp = _require_str(value, field_name, nonempty=True)
    if not _TIMESTAMP_RE.fullmatch(timestamp):
        _invalid(field_name)
    try:
        datetime.fromisoformat(f"{timestamp[:-1]}+00:00")
    except ValueError:
        _invalid(field_name)
    return timestamp


def _require_month(value: object, field_name: str) -> str:
    month = _require_str(value, field_name, nonempty=True)
    if not _MONTH_RE.fullmatch(month):
        _invalid(field_name)
    try:
        datetime(int(month[:4]), int(month[5:]), 1)
    except ValueError:
        _invalid(field_name)
    return month


def _require_sha256(value: object, field_name: str) -> str:
    digest = _require_str(value, field_name, nonempty=True)
    if not _SHA256_RE.fullmatch(digest):
        _invalid(field_name)
    return digest


def _require_schema_version(value: object) -> Literal["1.0"]:
    if value != _SCHEMA_VERSION or not isinstance(value, str):
        _invalid("schema_version")
    return _SCHEMA_VERSION


def _require_literal(value: object, field_name: str, allowed: frozenset[str]) -> str:
    literal = _require_str(value, field_name, nonempty=True)
    if literal not in allowed:
        _invalid(field_name)
    return literal


def _require_path(value: object, field_name: str) -> Path:
    if not isinstance(value, Path):
        _invalid(field_name)
    return value


def _path_from_dict(value: object, field_name: str) -> Path:
    return Path(_require_str(value, field_name, nonempty=True))


def _require_relative_path(value: object, field_name: str) -> str | None:
    path_string = _require_optional_str(value, field_name, nonempty=True)
    if path_string is None:
        return None
    if Path(path_string).is_absolute() or PureWindowsPath(path_string).is_absolute():
        _invalid(field_name)
    return path_string


def _require_pair_counts(value: object, field_name: str) -> PairCounts:
    counts = _require_dict(value, field_name)
    if set(counts) != {"candidate", "suspect"}:
        _invalid(field_name)
    return PairCounts(
        candidate=_require_int(counts["candidate"], field_name, minimum=0),
        suspect=_require_int(counts["suspect"], field_name, minimum=0),
    )


def _require_error_code(value: object, field_name: str) -> str:
    code = _require_str(value, field_name, nonempty=True)
    try:
        return ErrorCode(code).value
    except ValueError:
        _invalid(field_name)
    raise AssertionError("unreachable")


def _require_counts_dict(value: object, field_name: str) -> dict[str, int]:
    counts = _require_dict(value, field_name)
    validated: dict[str, int] = {}
    for key, count in counts.items():
        _require_str(key, field_name, nonempty=True)
        validated[key] = _require_int(count, field_name, minimum=0)
    return validated


def _require_partition_sizes(value: object, field_name: str) -> dict[str, int]:
    sizes = _require_dict(value, field_name)
    if set(sizes) != {"train", "validation", "test"}:
        _invalid(field_name)
    return {
        "train": _require_int(sizes["train"], field_name, minimum=0),
        "validation": _require_int(sizes["validation"], field_name, minimum=0),
        "test": _require_int(sizes["test"], field_name, minimum=0),
    }


def _serialize(value: object) -> object:
    if isinstance(value, _Contract):
        return value.to_dict()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, StrEnum):
        return value.value
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, Mapping):
        serialized: dict[str, object] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                _invalid("object")
            serialized[key] = _serialize(item)
        return serialized
    return value


class _Contract:
    """Common deterministic conversion for frozen dataclass contracts."""

    __slots__ = ()

    def _validate(self) -> None:
        raise NotImplementedError

    def to_dict(self) -> dict[str, object]:
        self._validate()
        return {item.name: _serialize(getattr(self, item.name)) for item in fields(self)}


def message_key(game_id: str, message_id: str) -> str:
    """Return the unambiguous serialized identity for a game-scoped message."""

    _require_str(game_id, "game_id", nonempty=True)
    _require_str(message_id, "message_id", nonempty=True)
    return json.dumps([game_id, message_id], separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class Message(_Contract):
    schema_version: Literal["1.0"]
    message_id: str
    account_id: str
    game_id: str
    channel_id: str | None
    timestamp_utc: str
    text: str = field(repr=False)
    is_system: bool
    source_id: str
    source_row: int

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        _require_schema_version(self.schema_version)
        _require_str(self.message_id, "message_id", nonempty=True)
        _require_str(self.account_id, "account_id", nonempty=True)
        _require_str(self.game_id, "game_id", nonempty=True)
        _require_optional_str(self.channel_id, "channel_id", nonempty=True)
        _require_timestamp(self.timestamp_utc, "timestamp_utc")
        _require_str(self.text, "text")
        _require_bool(self.is_system, "is_system")
        _require_str(self.source_id, "source_id", nonempty=True)
        _require_int(self.source_row, "source_row", minimum=1)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        values = _checked_data(data, cls)
        return cls(
            schema_version=_require_schema_version(values["schema_version"]),
            message_id=_require_str(values["message_id"], "message_id", nonempty=True),
            account_id=_require_str(values["account_id"], "account_id", nonempty=True),
            game_id=_require_str(values["game_id"], "game_id", nonempty=True),
            channel_id=_require_optional_str(
                values["channel_id"], "channel_id", nonempty=True
            ),
            timestamp_utc=_require_timestamp(values["timestamp_utc"], "timestamp_utc"),
            text=_require_str(values["text"], "text"),
            is_system=_require_bool(values["is_system"], "is_system"),
            source_id=_require_str(values["source_id"], "source_id", nonempty=True),
            source_row=_require_int(values["source_row"], "source_row", minimum=1),
        )


@dataclass(frozen=True, slots=True)
class Chunk(_Contract):
    schema_version: Literal["1.0"]
    chunk_id: str
    account_id: str
    session_id: str
    message_ids: list[str]
    start_timestamp_utc: str
    end_timestamp_utc: str
    n_messages: int
    n_chars: int
    raw_text: list[str] = field(repr=False)
    normalized_text: str = field(repr=False)
    normalized_sha256: str

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        _require_schema_version(self.schema_version)
        _require_str(self.chunk_id, "chunk_id", nonempty=True)
        _require_str(self.account_id, "account_id", nonempty=True)
        _require_str(self.session_id, "session_id", nonempty=True)
        message_ids = _require_string_list(
            self.message_ids, "message_ids", nonempty_items=True
        )
        _require_timestamp(self.start_timestamp_utc, "start_timestamp_utc")
        _require_timestamp(self.end_timestamp_utc, "end_timestamp_utc")
        n_messages = _require_int(self.n_messages, "n_messages", minimum=0)
        n_chars = _require_int(self.n_chars, "n_chars", minimum=0)
        raw_text = _require_string_list(self.raw_text, "raw_text")
        _require_str(self.normalized_text, "normalized_text")
        _require_sha256(self.normalized_sha256, "normalized_sha256")
        if n_messages != len(message_ids) or n_messages != len(raw_text):
            _invalid("n_messages")
        if n_chars != sum(len(text) for text in raw_text):
            _invalid("n_chars")

    def to_raw_stream_chunk(self) -> RawStreamChunk:
        self._validate()
        return RawStreamChunk(
            schema_version=self.schema_version,
            chunk_id=self.chunk_id,
            account_id=self.account_id,
            session_id=self.session_id,
            message_ids=list(self.message_ids),
            start_timestamp_utc=self.start_timestamp_utc,
            end_timestamp_utc=self.end_timestamp_utc,
            n_messages=self.n_messages,
            n_chars=self.n_chars,
            raw_text=list(self.raw_text),
        )

    def to_normalized_stream_chunk(self) -> NormalizedStreamChunk:
        self._validate()
        return NormalizedStreamChunk(
            schema_version=self.schema_version,
            chunk_id=self.chunk_id,
            account_id=self.account_id,
            session_id=self.session_id,
            message_ids=list(self.message_ids),
            start_timestamp_utc=self.start_timestamp_utc,
            end_timestamp_utc=self.end_timestamp_utc,
            n_messages=self.n_messages,
            n_chars=self.n_chars,
            normalized_text=self.normalized_text,
            normalized_sha256=self.normalized_sha256,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        values = _checked_data(data, cls)
        return cls(
            schema_version=_require_schema_version(values["schema_version"]),
            chunk_id=_require_str(values["chunk_id"], "chunk_id", nonempty=True),
            account_id=_require_str(values["account_id"], "account_id", nonempty=True),
            session_id=_require_str(values["session_id"], "session_id", nonempty=True),
            message_ids=_require_string_list(
                values["message_ids"], "message_ids", nonempty_items=True
            ),
            start_timestamp_utc=_require_timestamp(
                values["start_timestamp_utc"], "start_timestamp_utc"
            ),
            end_timestamp_utc=_require_timestamp(
                values["end_timestamp_utc"], "end_timestamp_utc"
            ),
            n_messages=_require_int(values["n_messages"], "n_messages", minimum=0),
            n_chars=_require_int(values["n_chars"], "n_chars", minimum=0),
            raw_text=_require_string_list(values["raw_text"], "raw_text"),
            normalized_text=_require_str(values["normalized_text"], "normalized_text"),
            normalized_sha256=_require_sha256(
                values["normalized_sha256"], "normalized_sha256"
            ),
        )


@dataclass(frozen=True, slots=True)
class RawStreamChunk(_Contract):
    schema_version: Literal["1.0"]
    chunk_id: str
    account_id: str
    session_id: str
    message_ids: list[str]
    start_timestamp_utc: str
    end_timestamp_utc: str
    n_messages: int
    n_chars: int
    raw_text: list[str] = field(repr=False)

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        _require_schema_version(self.schema_version)
        _require_str(self.chunk_id, "chunk_id", nonempty=True)
        _require_str(self.account_id, "account_id", nonempty=True)
        _require_str(self.session_id, "session_id", nonempty=True)
        message_ids = _require_string_list(
            self.message_ids, "message_ids", nonempty_items=True
        )
        _require_timestamp(self.start_timestamp_utc, "start_timestamp_utc")
        _require_timestamp(self.end_timestamp_utc, "end_timestamp_utc")
        n_messages = _require_int(self.n_messages, "n_messages", minimum=0)
        n_chars = _require_int(self.n_chars, "n_chars", minimum=0)
        raw_text = _require_string_list(self.raw_text, "raw_text")
        if n_messages != len(message_ids) or n_messages != len(raw_text):
            _invalid("n_messages")
        if n_chars != sum(len(text) for text in raw_text):
            _invalid("n_chars")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        values = _checked_data(data, cls)
        return cls(
            schema_version=_require_schema_version(values["schema_version"]),
            chunk_id=_require_str(values["chunk_id"], "chunk_id", nonempty=True),
            account_id=_require_str(values["account_id"], "account_id", nonempty=True),
            session_id=_require_str(values["session_id"], "session_id", nonempty=True),
            message_ids=_require_string_list(
                values["message_ids"], "message_ids", nonempty_items=True
            ),
            start_timestamp_utc=_require_timestamp(
                values["start_timestamp_utc"], "start_timestamp_utc"
            ),
            end_timestamp_utc=_require_timestamp(
                values["end_timestamp_utc"], "end_timestamp_utc"
            ),
            n_messages=_require_int(values["n_messages"], "n_messages", minimum=0),
            n_chars=_require_int(values["n_chars"], "n_chars", minimum=0),
            raw_text=_require_string_list(values["raw_text"], "raw_text"),
        )


@dataclass(frozen=True, slots=True)
class NormalizedStreamChunk(_Contract):
    schema_version: Literal["1.0"]
    chunk_id: str
    account_id: str
    session_id: str
    message_ids: list[str]
    start_timestamp_utc: str
    end_timestamp_utc: str
    n_messages: int
    n_chars: int
    normalized_text: str = field(repr=False)
    normalized_sha256: str

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        _require_schema_version(self.schema_version)
        _require_str(self.chunk_id, "chunk_id", nonempty=True)
        _require_str(self.account_id, "account_id", nonempty=True)
        _require_str(self.session_id, "session_id", nonempty=True)
        message_ids = _require_string_list(
            self.message_ids, "message_ids", nonempty_items=True
        )
        _require_timestamp(self.start_timestamp_utc, "start_timestamp_utc")
        _require_timestamp(self.end_timestamp_utc, "end_timestamp_utc")
        n_messages = _require_int(self.n_messages, "n_messages", minimum=0)
        _require_int(self.n_chars, "n_chars", minimum=0)
        _require_str(self.normalized_text, "normalized_text")
        _require_sha256(self.normalized_sha256, "normalized_sha256")
        if n_messages != len(message_ids):
            _invalid("n_messages")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        values = _checked_data(data, cls)
        return cls(
            schema_version=_require_schema_version(values["schema_version"]),
            chunk_id=_require_str(values["chunk_id"], "chunk_id", nonempty=True),
            account_id=_require_str(values["account_id"], "account_id", nonempty=True),
            session_id=_require_str(values["session_id"], "session_id", nonempty=True),
            message_ids=_require_string_list(
                values["message_ids"], "message_ids", nonempty_items=True
            ),
            start_timestamp_utc=_require_timestamp(
                values["start_timestamp_utc"], "start_timestamp_utc"
            ),
            end_timestamp_utc=_require_timestamp(
                values["end_timestamp_utc"], "end_timestamp_utc"
            ),
            n_messages=_require_int(values["n_messages"], "n_messages", minimum=0),
            n_chars=_require_int(values["n_chars"], "n_chars", minimum=0),
            normalized_text=_require_str(values["normalized_text"], "normalized_text"),
            normalized_sha256=_require_sha256(
                values["normalized_sha256"], "normalized_sha256"
            ),
        )


@dataclass(frozen=True, slots=True)
class AccountProfile(_Contract):
    schema_version: Literal["1.0"]
    account_id: str
    game_ids: list[str]
    primary_game_id: str
    primary_channel_id: str | None
    active_months_utc: list[str]
    n_messages: int
    n_chars: int
    session_count: int
    activity_histogram_utc: list[int]
    utc_hour_filter: list[int] | None
    chunk_ids: list[str]
    raw_stream_path: str
    normalized_stream_path: str
    status: Literal["READY", "INSUFFICIENT_DATA"]
    minimum_messages: int
    minimum_chars: int
    preprocessing_fingerprint: str

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        _require_schema_version(self.schema_version)
        _require_str(self.account_id, "account_id", nonempty=True)
        game_ids = _require_sorted_unique_strings(
            self.game_ids, "game_ids", nonempty=True
        )
        primary_game_id = _require_str(
            self.primary_game_id, "primary_game_id", nonempty=True
        )
        if primary_game_id not in game_ids:
            _invalid("primary_game_id")
        _require_optional_str(
            self.primary_channel_id, "primary_channel_id", nonempty=True
        )
        months = _require_string_list(
            self.active_months_utc, "active_months_utc", nonempty_items=True
        )
        for month in months:
            _require_month(month, "active_months_utc")
        if months != sorted(set(months)):
            _invalid("active_months_utc")
        _require_int(self.n_messages, "n_messages", minimum=0)
        _require_int(self.n_chars, "n_chars", minimum=0)
        _require_int(self.session_count, "session_count", minimum=0)
        histogram = _require_list(
            self.activity_histogram_utc, "activity_histogram_utc"
        )
        if len(histogram) != 24:
            _invalid("activity_histogram_utc")
        for value in histogram:
            _require_int(value, "activity_histogram_utc", minimum=0)
        _require_hour_list(self.utc_hour_filter, "utc_hour_filter")
        _require_string_list(self.chunk_ids, "chunk_ids", nonempty_items=True)
        _require_str(self.raw_stream_path, "raw_stream_path", nonempty=True)
        _require_str(
            self.normalized_stream_path, "normalized_stream_path", nonempty=True
        )
        _require_literal(
            self.status,
            "status",
            frozenset(("READY", "INSUFFICIENT_DATA")),
        )
        _require_int(self.minimum_messages, "minimum_messages", minimum=0)
        _require_int(self.minimum_chars, "minimum_chars", minimum=0)
        _require_sha256(self.preprocessing_fingerprint, "preprocessing_fingerprint")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        values = _checked_data(data, cls)
        return cls(
            schema_version=_require_schema_version(values["schema_version"]),
            account_id=_require_str(values["account_id"], "account_id", nonempty=True),
            game_ids=_require_sorted_unique_strings(
                values["game_ids"], "game_ids", nonempty=True
            ),
            primary_game_id=_require_str(
                values["primary_game_id"], "primary_game_id", nonempty=True
            ),
            primary_channel_id=_require_optional_str(
                values["primary_channel_id"], "primary_channel_id", nonempty=True
            ),
            active_months_utc=_require_string_list(
                values["active_months_utc"], "active_months_utc", nonempty_items=True
            ),
            n_messages=_require_int(values["n_messages"], "n_messages", minimum=0),
            n_chars=_require_int(values["n_chars"], "n_chars", minimum=0),
            session_count=_require_int(
                values["session_count"], "session_count", minimum=0
            ),
            activity_histogram_utc=[
                _require_int(item, "activity_histogram_utc", minimum=0)
                for item in _require_list(
                    values["activity_histogram_utc"], "activity_histogram_utc"
                )
            ],
            utc_hour_filter=_require_hour_list(
                values["utc_hour_filter"], "utc_hour_filter"
            ),
            chunk_ids=_require_string_list(
                values["chunk_ids"], "chunk_ids", nonempty_items=True
            ),
            raw_stream_path=_require_str(
                values["raw_stream_path"], "raw_stream_path", nonempty=True
            ),
            normalized_stream_path=_require_str(
                values["normalized_stream_path"],
                "normalized_stream_path",
                nonempty=True,
            ),
            status=_require_literal(
                values["status"],
                "status",
                frozenset(("READY", "INSUFFICIENT_DATA")),
            ),
            minimum_messages=_require_int(
                values["minimum_messages"], "minimum_messages", minimum=0
            ),
            minimum_chars=_require_int(
                values["minimum_chars"], "minimum_chars", minimum=0
            ),
            preprocessing_fingerprint=_require_sha256(
                values["preprocessing_fingerprint"], "preprocessing_fingerprint"
            ),
        )


class PairCounts(TypedDict):
    """Counts for a comparison pair; candidate always means flagged account."""

    candidate: int
    suspect: int


@dataclass(frozen=True, slots=True)
class EngineResult(_Contract):
    engine: Literal["stylometry", "semantic"]
    raw_cosine: float | None
    calibrated_percentile: float | None
    status: Literal["OK", "INSUFFICIENT_DATA", "TIMEOUT", "ERROR"]
    n_messages: PairCounts
    n_chars: PairCounts
    error: str | None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        _require_literal(self.engine, "engine", _ENGINE_NAMES)
        raw_cosine = _require_optional_float(
            self.raw_cosine, "raw_cosine", minimum=-1.0, maximum=1.0
        )
        percentile = _require_optional_float(
            self.calibrated_percentile,
            "calibrated_percentile",
            minimum=0.0,
            maximum=100.0,
        )
        status = _require_literal(
            self.status,
            "status",
            frozenset(("OK", "INSUFFICIENT_DATA", "TIMEOUT", "ERROR")),
        )
        _require_pair_counts(self.n_messages, "n_messages")
        _require_pair_counts(self.n_chars, "n_chars")
        if status == "OK":
            if raw_cosine is None or percentile is None or self.error is not None:
                _invalid("status")
        else:
            if raw_cosine is not None or percentile is not None:
                _invalid("status")
            _require_error_code(self.error, "error")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        values = _checked_data(data, cls)
        return cls(
            engine=_require_literal(values["engine"], "engine", _ENGINE_NAMES),
            raw_cosine=_require_optional_float(
                values["raw_cosine"], "raw_cosine", minimum=-1.0, maximum=1.0
            ),
            calibrated_percentile=_require_optional_float(
                values["calibrated_percentile"],
                "calibrated_percentile",
                minimum=0.0,
                maximum=100.0,
            ),
            status=_require_literal(
                values["status"],
                "status",
                frozenset(("OK", "INSUFFICIENT_DATA", "TIMEOUT", "ERROR")),
            ),
            n_messages=_require_pair_counts(values["n_messages"], "n_messages"),
            n_chars=_require_pair_counts(values["n_chars"], "n_chars"),
            error=(
                None
                if values["error"] is None
                else _require_error_code(values["error"], "error")
            ),
        )


@dataclass(frozen=True, slots=True)
class BaselineDistribution(_Contract):
    schema_version: Literal["1.0"]
    baseline_id: str
    engine: Literal["stylometry", "semantic"]
    created_at_utc: str
    source_snapshot_id: str
    configuration_fingerprint: str
    random_seed: int
    calibration_population: Literal["known_different_account_pairs"]
    n_pairs: int
    sorted_raw_cosines: list[float]
    minimum_messages: int
    minimum_chars: int
    strata_counts: dict[str, int]
    centering_vector_path: str | None

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        _require_schema_version(self.schema_version)
        _require_str(self.baseline_id, "baseline_id", nonempty=True)
        engine = _require_literal(self.engine, "engine", _ENGINE_NAMES)
        _require_timestamp(self.created_at_utc, "created_at_utc")
        _require_str(self.source_snapshot_id, "source_snapshot_id", nonempty=True)
        _require_sha256(self.configuration_fingerprint, "configuration_fingerprint")
        _require_int(self.random_seed, "random_seed")
        _require_literal(
            self.calibration_population,
            "calibration_population",
            frozenset(("known_different_account_pairs",)),
        )
        n_pairs = _require_int(self.n_pairs, "n_pairs", minimum=1)
        cosines = _require_float_list(
            self.sorted_raw_cosines,
            "sorted_raw_cosines",
            minimum=-1.0,
            maximum=1.0,
        )
        if n_pairs != len(cosines):
            _invalid("n_pairs")
        if any(left > right for left, right in zip(cosines, cosines[1:])):
            _invalid("sorted_raw_cosines")
        _require_int(self.minimum_messages, "minimum_messages", minimum=0)
        _require_int(self.minimum_chars, "minimum_chars", minimum=0)
        _require_counts_dict(self.strata_counts, "strata_counts")
        path = _require_relative_path(
            self.centering_vector_path, "centering_vector_path"
        )
        if engine != "semantic" and path is not None:
            _invalid("centering_vector_path")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        values = _checked_data(data, cls)
        return cls(
            schema_version=_require_schema_version(values["schema_version"]),
            baseline_id=_require_str(values["baseline_id"], "baseline_id", nonempty=True),
            engine=_require_literal(values["engine"], "engine", _ENGINE_NAMES),
            created_at_utc=_require_timestamp(
                values["created_at_utc"], "created_at_utc"
            ),
            source_snapshot_id=_require_str(
                values["source_snapshot_id"], "source_snapshot_id", nonempty=True
            ),
            configuration_fingerprint=_require_sha256(
                values["configuration_fingerprint"], "configuration_fingerprint"
            ),
            random_seed=_require_int(values["random_seed"], "random_seed"),
            calibration_population=_require_literal(
                values["calibration_population"],
                "calibration_population",
                frozenset(("known_different_account_pairs",)),
            ),
            n_pairs=_require_int(values["n_pairs"], "n_pairs", minimum=1),
            sorted_raw_cosines=_require_float_list(
                values["sorted_raw_cosines"],
                "sorted_raw_cosines",
                minimum=-1.0,
                maximum=1.0,
            ),
            minimum_messages=_require_int(
                values["minimum_messages"], "minimum_messages", minimum=0
            ),
            minimum_chars=_require_int(
                values["minimum_chars"], "minimum_chars", minimum=0
            ),
            strata_counts=_require_counts_dict(values["strata_counts"], "strata_counts"),
            centering_vector_path=_require_relative_path(
                values["centering_vector_path"], "centering_vector_path"
            ),
        )


@dataclass(frozen=True, slots=True)
class ComparisonReport(_Contract):
    schema_version: Literal["1.0"]
    flagged_account_id: str
    suspect_account_id: str
    stylometry: EngineResult
    semantic: EngineResult
    banner: Literal["Decision support only. Requires human review."]

    BANNER: ClassVar[str] = "Decision support only. Requires human review."

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        _require_schema_version(self.schema_version)
        _require_str(self.flagged_account_id, "flagged_account_id", nonempty=True)
        _require_str(self.suspect_account_id, "suspect_account_id", nonempty=True)
        if not isinstance(self.stylometry, EngineResult):
            _invalid("stylometry")
        if not isinstance(self.semantic, EngineResult):
            _invalid("semantic")
        self.stylometry._validate()
        self.semantic._validate()
        if self.stylometry.engine != "stylometry":
            _invalid("stylometry")
        if self.semantic.engine != "semantic":
            _invalid("semantic")
        if self.banner != self.BANNER or not isinstance(self.banner, str):
            _invalid("banner")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        values = _checked_data(data, cls)
        return cls(
            schema_version=_require_schema_version(values["schema_version"]),
            flagged_account_id=_require_str(
                values["flagged_account_id"], "flagged_account_id", nonempty=True
            ),
            suspect_account_id=_require_str(
                values["suspect_account_id"], "suspect_account_id", nonempty=True
            ),
            stylometry=EngineResult.from_dict(
                _require_mapping(values["stylometry"], "stylometry")
            ),
            semantic=EngineResult.from_dict(_require_mapping(values["semantic"], "semantic")),
            banner=_require_str(values["banner"], "banner", nonempty=True),
        )


@dataclass(frozen=True, slots=True)
class LabeledPair(_Contract):
    schema_version: Literal["1.0"]
    account_id_a: str
    account_id_b: str
    label: Literal["same", "different"]
    source_snapshot_id: str

    def __post_init__(self) -> None:
        _require_schema_version(self.schema_version)
        account_id_a = _require_str(self.account_id_a, "account_id_a", nonempty=True)
        account_id_b = _require_str(self.account_id_b, "account_id_b", nonempty=True)
        if account_id_a == account_id_b:
            _invalid("account_id_b")
        if account_id_b < account_id_a:
            object.__setattr__(self, "account_id_a", account_id_b)
            object.__setattr__(self, "account_id_b", account_id_a)
        self._validate()

    def _validate(self) -> None:
        _require_schema_version(self.schema_version)
        account_id_a = _require_str(self.account_id_a, "account_id_a", nonempty=True)
        account_id_b = _require_str(self.account_id_b, "account_id_b", nonempty=True)
        if account_id_a >= account_id_b:
            _invalid("account_id_b")
        _require_literal(self.label, "label", frozenset(("same", "different")))
        _require_str(self.source_snapshot_id, "source_snapshot_id", nonempty=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        values = _checked_data(data, cls)
        return cls(
            schema_version=_require_schema_version(values["schema_version"]),
            account_id_a=_require_str(
                values["account_id_a"], "account_id_a", nonempty=True
            ),
            account_id_b=_require_str(
                values["account_id_b"], "account_id_b", nonempty=True
            ),
            label=_require_literal(
                values["label"], "label", frozenset(("same", "different"))
            ),
            source_snapshot_id=_require_str(
                values["source_snapshot_id"], "source_snapshot_id", nonempty=True
            ),
        )


@dataclass(frozen=True, slots=True)
class StylometryArtifact(_Contract):
    artifact_id: str
    artifact_sha256: str
    feature_spec_version: str
    hash_space_size: int
    function_words_sha256: str
    function_words_lang: str
    structural_means: list[float]
    structural_stds: list[float]
    payload_path: str
    seed: int
    configuration_fingerprint: str

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        _require_str(self.artifact_id, "artifact_id", nonempty=True)
        _require_sha256(self.artifact_sha256, "artifact_sha256")
        _require_str(self.feature_spec_version, "feature_spec_version", nonempty=True)
        _require_int(self.hash_space_size, "hash_space_size", minimum=1)
        _require_sha256(self.function_words_sha256, "function_words_sha256")
        _require_str(self.function_words_lang, "function_words_lang", nonempty=True)
        means = _require_float_list(self.structural_means, "structural_means")
        stds = _require_float_list(self.structural_stds, "structural_stds")
        if len(means) != len(stds):
            _invalid("structural_stds")
        _require_str(self.payload_path, "payload_path", nonempty=True)
        _require_int(self.seed, "seed")
        _require_sha256(self.configuration_fingerprint, "configuration_fingerprint")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        values = _checked_data(data, cls)
        return cls(
            artifact_id=_require_str(values["artifact_id"], "artifact_id", nonempty=True),
            artifact_sha256=_require_sha256(values["artifact_sha256"], "artifact_sha256"),
            feature_spec_version=_require_str(
                values["feature_spec_version"], "feature_spec_version", nonempty=True
            ),
            hash_space_size=_require_int(
                values["hash_space_size"], "hash_space_size", minimum=1
            ),
            function_words_sha256=_require_sha256(
                values["function_words_sha256"], "function_words_sha256"
            ),
            function_words_lang=_require_str(
                values["function_words_lang"], "function_words_lang", nonempty=True
            ),
            structural_means=_require_float_list(
                values["structural_means"], "structural_means"
            ),
            structural_stds=_require_float_list(
                values["structural_stds"], "structural_stds"
            ),
            payload_path=_require_str(values["payload_path"], "payload_path", nonempty=True),
            seed=_require_int(values["seed"], "seed"),
            configuration_fingerprint=_require_sha256(
                values["configuration_fingerprint"], "configuration_fingerprint"
            ),
        )


class EmbeddingModel(Protocol):
    """Interface supplied by the semantic-engine implementation."""

    model_fingerprint: str
    embedding_dim: int

    def embed(
        self, texts: Sequence[str]
    ) -> "numpy.typing.NDArray[numpy.float32]": ...


@dataclass(frozen=True, slots=True)
class ComparisonAssets(_Contract):
    stylometry_artifact_path: Path
    semantic_model_dir: Path
    baseline_paths: Mapping[Literal["stylometry", "semantic"], Path]
    game_terms_path: Path
    usernames_path: Path
    function_words_path: Path
    cache_dir: Path

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        _require_path(self.stylometry_artifact_path, "stylometry_artifact_path")
        _require_path(self.semantic_model_dir, "semantic_model_dir")
        baseline_paths = _require_mapping(self.baseline_paths, "baseline_paths")
        if set(baseline_paths) != _ENGINE_NAMES:
            _invalid("baseline_paths")
        for engine in _ENGINE_NAMES:
            _require_path(baseline_paths[engine], "baseline_paths")
        _require_path(self.game_terms_path, "game_terms_path")
        _require_path(self.usernames_path, "usernames_path")
        _require_path(self.function_words_path, "function_words_path")
        _require_path(self.cache_dir, "cache_dir")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        values = _checked_data(data, cls)
        baseline_paths = _require_dict(values["baseline_paths"], "baseline_paths")
        if set(baseline_paths) != _ENGINE_NAMES:
            _invalid("baseline_paths")
        return cls(
            stylometry_artifact_path=_path_from_dict(
                values["stylometry_artifact_path"], "stylometry_artifact_path"
            ),
            semantic_model_dir=_path_from_dict(
                values["semantic_model_dir"], "semantic_model_dir"
            ),
            baseline_paths={
                "stylometry": _path_from_dict(
                    baseline_paths["stylometry"], "baseline_paths"
                ),
                "semantic": _path_from_dict(
                    baseline_paths["semantic"], "baseline_paths"
                ),
            },
            game_terms_path=_path_from_dict(values["game_terms_path"], "game_terms_path"),
            usernames_path=_path_from_dict(values["usernames_path"], "usernames_path"),
            function_words_path=_path_from_dict(
                values["function_words_path"], "function_words_path"
            ),
            cache_dir=_path_from_dict(values["cache_dir"], "cache_dir"),
        )


@dataclass(frozen=True, slots=True)
class IngestConfig(_Contract):
    input_format: Literal["json", "csv", "auto"] = "auto"
    field_map: dict[str, str] = field(default_factory=dict)
    system_flag_field: str | None = None
    system_flag_true_values: list[str] = field(default_factory=lambda: ["true", "1"])
    dedupe: bool = True

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        _require_literal(
            self.input_format, "input_format", frozenset(("json", "csv", "auto"))
        )
        field_map = _require_dict(self.field_map, "field_map")
        for key, value in field_map.items():
            _require_str(key, "field_map", nonempty=True)
            _require_str(value, "field_map", nonempty=True)
        _require_optional_str(self.system_flag_field, "system_flag_field", nonempty=True)
        _require_string_list(
            self.system_flag_true_values,
            "system_flag_true_values",
            nonempty_items=True,
        )
        _require_bool(self.dedupe, "dedupe")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        values = _checked_data(data, cls)
        field_map = _require_dict(values["field_map"], "field_map")
        return cls(
            input_format=_require_literal(
                values["input_format"],
                "input_format",
                frozenset(("json", "csv", "auto")),
            ),
            field_map={
                _require_str(key, "field_map", nonempty=True): _require_str(
                    value, "field_map", nonempty=True
                )
                for key, value in field_map.items()
            },
            system_flag_field=_require_optional_str(
                values["system_flag_field"], "system_flag_field", nonempty=True
            ),
            system_flag_true_values=_require_string_list(
                values["system_flag_true_values"],
                "system_flag_true_values",
                nonempty_items=True,
            ),
            dedupe=_require_bool(values["dedupe"], "dedupe"),
        )


@dataclass(frozen=True, slots=True)
class PreprocessConfig(_Contract):
    minimum_messages: int = 200
    minimum_chars: int = 5000
    idle_gap_minutes: int = 30
    utc_hour_filter: list[int] | None = None
    chunk_target_mode: Literal["messages", "characters"] = "characters"
    chunk_target_value: int = 1000
    cache_retention_days: int = 7

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        _require_int(self.minimum_messages, "minimum_messages", minimum=1)
        _require_int(self.minimum_chars, "minimum_chars", minimum=1)
        _require_int(self.idle_gap_minutes, "idle_gap_minutes", minimum=1)
        _require_hour_list(self.utc_hour_filter, "utc_hour_filter")
        _require_literal(
            self.chunk_target_mode,
            "chunk_target_mode",
            frozenset(("messages", "characters")),
        )
        _require_int(self.chunk_target_value, "chunk_target_value", minimum=1)
        _require_int(self.cache_retention_days, "cache_retention_days", minimum=0)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        values = _checked_data(data, cls)
        return cls(
            minimum_messages=_require_int(
                values["minimum_messages"], "minimum_messages", minimum=1
            ),
            minimum_chars=_require_int(values["minimum_chars"], "minimum_chars", minimum=1),
            idle_gap_minutes=_require_int(
                values["idle_gap_minutes"], "idle_gap_minutes", minimum=1
            ),
            utc_hour_filter=_require_hour_list(
                values["utc_hour_filter"], "utc_hour_filter"
            ),
            chunk_target_mode=_require_literal(
                values["chunk_target_mode"],
                "chunk_target_mode",
                frozenset(("messages", "characters")),
            ),
            chunk_target_value=_require_int(
                values["chunk_target_value"], "chunk_target_value", minimum=1
            ),
            cache_retention_days=_require_int(
                values["cache_retention_days"], "cache_retention_days", minimum=0
            ),
        )


@dataclass(frozen=True, slots=True)
class StylometryConfig(_Contract):
    feature_spec_version: str = "1.0"
    hash_space_size: int = 262144
    ngram_min: int = 2
    ngram_max: int = 4
    block_weights: dict[str, float] = field(
        default_factory=lambda: {
            "char_ngrams": 0.5,
            "structural": 0.25,
            "function_words": 0.25,
        }
    )
    minimum_messages: int = 200
    minimum_chars: int = 5000

    _BLOCK_NAMES: ClassVar[frozenset[str]] = frozenset(
        ("char_ngrams", "structural", "function_words")
    )

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        _require_str(self.feature_spec_version, "feature_spec_version", nonempty=True)
        _require_int(self.hash_space_size, "hash_space_size", minimum=1)
        ngram_min = _require_int(self.ngram_min, "ngram_min", minimum=1)
        ngram_max = _require_int(self.ngram_max, "ngram_max", minimum=1)
        if ngram_max < ngram_min:
            _invalid("ngram_max")
        weights = _require_dict(self.block_weights, "block_weights")
        if set(weights) != self._BLOCK_NAMES:
            _invalid("block_weights")
        values = [
            _require_float(weights[name], "block_weights", minimum=0.0)
            for name in sorted(self._BLOCK_NAMES)
        ]
        if not math.isclose(math.fsum(values), 1.0, rel_tol=0.0, abs_tol=1e-12):
            _invalid("block_weights")
        _require_int(self.minimum_messages, "minimum_messages", minimum=1)
        _require_int(self.minimum_chars, "minimum_chars", minimum=1)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        values = _checked_data(data, cls)
        weights = _require_dict(values["block_weights"], "block_weights")
        if set(weights) != cls._BLOCK_NAMES:
            _invalid("block_weights")
        return cls(
            feature_spec_version=_require_str(
                values["feature_spec_version"], "feature_spec_version", nonempty=True
            ),
            hash_space_size=_require_int(
                values["hash_space_size"], "hash_space_size", minimum=1
            ),
            ngram_min=_require_int(values["ngram_min"], "ngram_min", minimum=1),
            ngram_max=_require_int(values["ngram_max"], "ngram_max", minimum=1),
            block_weights={
                name: _require_float(weights[name], "block_weights", minimum=0.0)
                for name in cls._BLOCK_NAMES
            },
            minimum_messages=_require_int(
                values["minimum_messages"], "minimum_messages", minimum=1
            ),
            minimum_chars=_require_int(values["minimum_chars"], "minimum_chars", minimum=1),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SemanticConfig(_Contract):
    model_dir: Path
    device: Literal["auto", "cpu", "cuda"] = "auto"
    batch_size_cpu: int = 16
    batch_size_cuda: int = 64
    max_tokens: int = 256
    center_embeddings: bool = True
    allow_external: bool = False
    external_endpoint: str | None = None
    cache_dir: Path
    cache_retention_days: int = 7
    minimum_messages: int = 200
    minimum_chars: int = 5000

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        _require_path(self.model_dir, "model_dir")
        _require_literal(self.device, "device", frozenset(("auto", "cpu", "cuda")))
        _require_int(self.batch_size_cpu, "batch_size_cpu", minimum=1)
        _require_int(self.batch_size_cuda, "batch_size_cuda", minimum=1)
        _require_int(self.max_tokens, "max_tokens", minimum=1)
        _require_bool(self.center_embeddings, "center_embeddings")
        _require_bool(self.allow_external, "allow_external")
        _require_optional_str(self.external_endpoint, "external_endpoint", nonempty=True)
        _require_path(self.cache_dir, "cache_dir")
        _require_int(self.cache_retention_days, "cache_retention_days", minimum=0)
        _require_int(self.minimum_messages, "minimum_messages", minimum=1)
        _require_int(self.minimum_chars, "minimum_chars", minimum=1)

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        values = _checked_data(data, cls)
        return cls(
            model_dir=_path_from_dict(values["model_dir"], "model_dir"),
            cache_dir=_path_from_dict(values["cache_dir"], "cache_dir"),
            device=_require_literal(
                values["device"], "device", frozenset(("auto", "cpu", "cuda"))
            ),
            batch_size_cpu=_require_int(
                values["batch_size_cpu"], "batch_size_cpu", minimum=1
            ),
            batch_size_cuda=_require_int(
                values["batch_size_cuda"], "batch_size_cuda", minimum=1
            ),
            max_tokens=_require_int(values["max_tokens"], "max_tokens", minimum=1),
            center_embeddings=_require_bool(
                values["center_embeddings"], "center_embeddings"
            ),
            allow_external=_require_bool(values["allow_external"], "allow_external"),
            external_endpoint=_require_optional_str(
                values["external_endpoint"], "external_endpoint", nonempty=True
            ),
            cache_retention_days=_require_int(
                values["cache_retention_days"], "cache_retention_days", minimum=0
            ),
            minimum_messages=_require_int(
                values["minimum_messages"], "minimum_messages", minimum=1
            ),
            minimum_chars=_require_int(values["minimum_chars"], "minimum_chars", minimum=1),
        )


@dataclass(frozen=True, slots=True)
class BaselineConfig(_Contract):
    min_pairs: int = 1000
    max_pairs: int = 50000
    seed: int = 1729
    strata: list[str] = field(
        default_factory=lambda: ["game", "channel", "month", "volume_bucket"]
    )
    volume_bucket_edges: list[int] = field(
        default_factory=lambda: [200, 500, 1000, 5000]
    )
    refresh_max_age_days: int = 31
    refresh_new_pair_fraction: float = 0.10

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        min_pairs = _require_int(self.min_pairs, "min_pairs", minimum=1)
        max_pairs = _require_int(self.max_pairs, "max_pairs", minimum=1)
        if max_pairs < min_pairs:
            _invalid("max_pairs")
        _require_int(self.seed, "seed")
        _require_string_list(self.strata, "strata", nonempty_items=True)
        edges = [
            _require_int(edge, "volume_bucket_edges", minimum=0)
            for edge in _require_list(self.volume_bucket_edges, "volume_bucket_edges")
        ]
        if any(left >= right for left, right in zip(edges, edges[1:])):
            _invalid("volume_bucket_edges")
        _require_int(self.refresh_max_age_days, "refresh_max_age_days", minimum=1)
        _require_float(
            self.refresh_new_pair_fraction,
            "refresh_new_pair_fraction",
            minimum=0.0,
            maximum=1.0,
        )

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        values = _checked_data(data, cls)
        return cls(
            min_pairs=_require_int(values["min_pairs"], "min_pairs", minimum=1),
            max_pairs=_require_int(values["max_pairs"], "max_pairs", minimum=1),
            seed=_require_int(values["seed"], "seed"),
            strata=_require_string_list(
                values["strata"], "strata", nonempty_items=True
            ),
            volume_bucket_edges=[
                _require_int(edge, "volume_bucket_edges", minimum=0)
                for edge in _require_list(
                    values["volume_bucket_edges"], "volume_bucket_edges"
                )
            ],
            refresh_max_age_days=_require_int(
                values["refresh_max_age_days"], "refresh_max_age_days", minimum=1
            ),
            refresh_new_pair_fraction=_require_float(
                values["refresh_new_pair_fraction"],
                "refresh_new_pair_fraction",
                minimum=0.0,
                maximum=1.0,
            ),
        )


@dataclass(frozen=True, slots=True)
class RunConfig(_Contract):
    stylometry_timeout_s: float = 30.0
    semantic_timeout_s: float = 90.0
    terminate_timed_out_workers: bool = True
    output_format: Literal["table", "json"] = "table"
    allow_external: bool = False

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        _require_float(self.stylometry_timeout_s, "stylometry_timeout_s", minimum=0.0)
        if self.stylometry_timeout_s == 0.0:
            _invalid("stylometry_timeout_s")
        _require_float(self.semantic_timeout_s, "semantic_timeout_s", minimum=0.0)
        if self.semantic_timeout_s == 0.0:
            _invalid("semantic_timeout_s")
        _require_bool(
            self.terminate_timed_out_workers, "terminate_timed_out_workers"
        )
        _require_literal(
            self.output_format, "output_format", frozenset(("table", "json"))
        )
        _require_bool(self.allow_external, "allow_external")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        values = _checked_data(data, cls)
        return cls(
            stylometry_timeout_s=_require_float(
                values["stylometry_timeout_s"],
                "stylometry_timeout_s",
                minimum=0.0,
            ),
            semantic_timeout_s=_require_float(
                values["semantic_timeout_s"], "semantic_timeout_s", minimum=0.0
            ),
            terminate_timed_out_workers=_require_bool(
                values["terminate_timed_out_workers"], "terminate_timed_out_workers"
            ),
            output_format=_require_literal(
                values["output_format"], "output_format", frozenset(("table", "json"))
            ),
            allow_external=_require_bool(values["allow_external"], "allow_external"),
        )


@dataclass(frozen=True, slots=True)
class EvaluationConfig(_Contract):
    seed: int = 1729
    split_fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)
    min_precision: float = 0.80
    n_bootstrap: int = 1000
    ci_level: float = 0.95

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        _require_int(self.seed, "seed")
        if type(self.split_fractions) is not tuple or len(self.split_fractions) != 3:
            _invalid("split_fractions")
        fractions = [
            _require_float(value, "split_fractions", minimum=0.0)
            for value in self.split_fractions
        ]
        if not math.isclose(math.fsum(fractions), 1.0, rel_tol=0.0, abs_tol=1e-12):
            _invalid("split_fractions")
        _require_float(self.min_precision, "min_precision", minimum=0.0, maximum=1.0)
        _require_int(self.n_bootstrap, "n_bootstrap", minimum=1)
        ci_level = _require_float(self.ci_level, "ci_level", minimum=0.0, maximum=1.0)
        if ci_level == 0.0 or ci_level == 1.0:
            _invalid("ci_level")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        values = _checked_data(data, cls)
        fractions = _require_float_list(
            values["split_fractions"], "split_fractions", minimum=0.0
        )
        if len(fractions) != 3:
            _invalid("split_fractions")
        return cls(
            seed=_require_int(values["seed"], "seed"),
            split_fractions=(fractions[0], fractions[1], fractions[2]),
            min_precision=_require_float(
                values["min_precision"], "min_precision", minimum=0.0, maximum=1.0
            ),
            n_bootstrap=_require_int(values["n_bootstrap"], "n_bootstrap", minimum=1),
            ci_level=_require_float(
                values["ci_level"], "ci_level", minimum=0.0, maximum=1.0
            ),
        )


def _require_ci(value: object, field_name: str) -> dict[str, list[float]]:
    ci = _require_dict(value, field_name)
    expected = {"roc_auc", "precision", "recall"}
    if set(ci) != expected:
        _invalid(field_name)
    validated: dict[str, list[float]] = {}
    for metric in sorted(expected):
        interval = _require_float_list(ci[metric], field_name, minimum=0.0, maximum=1.0)
        if len(interval) != 2 or interval[0] > interval[1]:
            _invalid(field_name)
        validated[metric] = interval
    return validated


@dataclass(frozen=True, slots=True)
class EngineEvaluation(_Contract):
    engine: Literal["stylometry", "semantic"]
    roc_auc: float | None
    pr_auc: float | None
    test_base_rate: float
    threshold_percentile: float
    precision_constraint_met: bool
    precision: float
    recall: float
    support_same: int
    support_different: int
    excluded_insufficient_data: int
    ci: dict[str, list[float]]

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        _require_literal(self.engine, "engine", _ENGINE_NAMES)
        _require_optional_float(self.roc_auc, "roc_auc", minimum=0.0, maximum=1.0)
        _require_optional_float(self.pr_auc, "pr_auc", minimum=0.0, maximum=1.0)
        _require_float(self.test_base_rate, "test_base_rate", minimum=0.0, maximum=1.0)
        _require_float(
            self.threshold_percentile,
            "threshold_percentile",
            minimum=0.0,
            maximum=100.0,
        )
        _require_bool(self.precision_constraint_met, "precision_constraint_met")
        _require_float(self.precision, "precision", minimum=0.0, maximum=1.0)
        _require_float(self.recall, "recall", minimum=0.0, maximum=1.0)
        _require_int(self.support_same, "support_same", minimum=0)
        _require_int(self.support_different, "support_different", minimum=0)
        _require_int(
            self.excluded_insufficient_data, "excluded_insufficient_data", minimum=0
        )
        _require_ci(self.ci, "ci")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        values = _checked_data(data, cls)
        return cls(
            engine=_require_literal(values["engine"], "engine", _ENGINE_NAMES),
            roc_auc=_require_optional_float(
                values["roc_auc"], "roc_auc", minimum=0.0, maximum=1.0
            ),
            pr_auc=_require_optional_float(
                values["pr_auc"], "pr_auc", minimum=0.0, maximum=1.0
            ),
            test_base_rate=_require_float(
                values["test_base_rate"], "test_base_rate", minimum=0.0, maximum=1.0
            ),
            threshold_percentile=_require_float(
                values["threshold_percentile"],
                "threshold_percentile",
                minimum=0.0,
                maximum=100.0,
            ),
            precision_constraint_met=_require_bool(
                values["precision_constraint_met"], "precision_constraint_met"
            ),
            precision=_require_float(
                values["precision"], "precision", minimum=0.0, maximum=1.0
            ),
            recall=_require_float(values["recall"], "recall", minimum=0.0, maximum=1.0),
            support_same=_require_int(values["support_same"], "support_same", minimum=0),
            support_different=_require_int(
                values["support_different"], "support_different", minimum=0
            ),
            excluded_insufficient_data=_require_int(
                values["excluded_insufficient_data"],
                "excluded_insufficient_data",
                minimum=0,
            ),
            ci=_require_ci(values["ci"], "ci"),
        )


@dataclass(frozen=True, slots=True)
class EvaluationReport(_Contract):
    schema_version: Literal["1.0"]
    seed: int
    partition_sizes: dict[str, int]
    stylometry: EngineEvaluation
    semantic: EngineEvaluation
    note: Literal["Thresholds are review-prioritization suggestions only."]

    NOTE: ClassVar[str] = "Thresholds are review-prioritization suggestions only."

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        _require_schema_version(self.schema_version)
        _require_int(self.seed, "seed")
        _require_partition_sizes(self.partition_sizes, "partition_sizes")
        if not isinstance(self.stylometry, EngineEvaluation):
            _invalid("stylometry")
        if not isinstance(self.semantic, EngineEvaluation):
            _invalid("semantic")
        self.stylometry._validate()
        self.semantic._validate()
        if self.stylometry.engine != "stylometry":
            _invalid("stylometry")
        if self.semantic.engine != "semantic":
            _invalid("semantic")
        if self.note != self.NOTE or not isinstance(self.note, str):
            _invalid("note")

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Self:
        values = _checked_data(data, cls)
        return cls(
            schema_version=_require_schema_version(values["schema_version"]),
            seed=_require_int(values["seed"], "seed"),
            partition_sizes=_require_partition_sizes(
                values["partition_sizes"], "partition_sizes"
            ),
            stylometry=EngineEvaluation.from_dict(
                _require_mapping(values["stylometry"], "stylometry")
            ),
            semantic=EngineEvaluation.from_dict(_require_mapping(values["semantic"], "semantic")),
            note=_require_str(values["note"], "note", nonempty=True),
        )
