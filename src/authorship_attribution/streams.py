"""Strict typed readers for isolated raw and normalized NDJSON streams."""

from __future__ import annotations

import json
from pathlib import Path
from collections.abc import Sequence

from .contracts import AccountProfile, NormalizedStreamChunk, RawStreamChunk
from .errors import ContractValidationError, ErrorCode


def _invalid(field_name: str) -> None:
    raise ContractValidationError(field_name, ErrorCode.ARTIFACT_INVALID)


def _load_stream(path: Path, record_type: type[RawStreamChunk] | type[NormalizedStreamChunk]) -> list[object]:
    try:
        with path.open("r", encoding="utf-8", errors="strict", newline="") as source:
            records: list[object] = []
            for line in source:
                if not line.strip():
                    _invalid("line")
                value = json.loads(
                    line,
                    parse_constant=lambda _: (_ for _ in ()).throw(ValueError),
                )
                if not isinstance(value, dict):
                    _invalid("object")
                records.append(record_type.from_dict(value))
            return records
    except ContractValidationError:
        raise
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        _invalid("path")
    raise AssertionError("unreachable")


def load_raw_stream(profile: AccountProfile) -> Sequence[RawStreamChunk]:
    """Load only raw stream projections referenced by an account profile."""

    if not isinstance(profile, AccountProfile):
        _invalid("profile")
    loaded = _load_stream(Path(profile.raw_stream_path), RawStreamChunk)
    records = [record for record in loaded if isinstance(record, RawStreamChunk)]
    if [record.chunk_id for record in records] != profile.chunk_ids:
        _invalid("chunk_ids")
    if any(record.account_id != profile.account_id for record in records):
        _invalid("account_id")
    return records


def load_normalized_stream(profile: AccountProfile) -> Sequence[NormalizedStreamChunk]:
    """Load only normalized stream projections referenced by an account profile."""

    if not isinstance(profile, AccountProfile):
        _invalid("profile")
    loaded = _load_stream(Path(profile.normalized_stream_path), NormalizedStreamChunk)
    records = [record for record in loaded if isinstance(record, NormalizedStreamChunk)]
    if [record.chunk_id for record in records] != profile.chunk_ids:
        _invalid("chunk_ids")
    if any(record.account_id != profile.account_id for record in records):
        _invalid("account_id")
    return records
