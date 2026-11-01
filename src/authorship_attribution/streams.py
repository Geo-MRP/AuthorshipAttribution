"""Strict typed readers for isolated raw, normalized, and temporal streams."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
import json
from pathlib import Path

from .contracts import (
    AccountProfile,
    NormalizedStreamChunk,
    RawStreamChunk,
    TemporalStreamChunk,
)
from .errors import ContractValidationError, ErrorCode


def _invalid(field_name: str) -> None:
    raise ContractValidationError(field_name, ErrorCode.ARTIFACT_INVALID)


def _load_stream(
    path: Path,
    record_type: type[RawStreamChunk] | type[NormalizedStreamChunk] | type[TemporalStreamChunk],
) -> list[object]:
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


def load_temporal_stream(profile: AccountProfile) -> Sequence[TemporalStreamChunk]:
    """Load and validate the content-free projection for one profile."""

    if not isinstance(profile, AccountProfile):
        _invalid("profile")
    profile._validate()
    loaded = _load_stream(Path(profile.temporal_stream_path), TemporalStreamChunk)
    records = [record for record in loaded if isinstance(record, TemporalStreamChunk)]
    chunk_ids = [record.chunk_id for record in records]
    if chunk_ids != profile.chunk_ids or len(set(chunk_ids)) != len(chunk_ids):
        _invalid("chunk_ids")
    if any(record.account_id != profile.account_id for record in records):
        _invalid("account_id")
    if sum(record.n_messages for record in records) != profile.n_messages:
        _invalid("n_messages")
    if sum(record.n_chars for record in records) != profile.n_chars:
        _invalid("n_chars")

    current_session: str | None = None
    completed_sessions: set[str] = set()
    observed_sessions: set[str] = set()
    previous_epoch: float | None = None
    for record in records:
        if record.session_id != current_session:
            if current_session is not None:
                completed_sessions.add(current_session)
            if record.session_id in completed_sessions:
                _invalid("session_id")
            current_session = record.session_id
            observed_sessions.add(record.session_id)
        for timestamp in record.timestamps_utc:
            epoch = datetime.fromisoformat(f"{timestamp[:-1]}+00:00").timestamp()
            if previous_epoch is not None and epoch < previous_epoch:
                _invalid("timestamps_utc")
            previous_epoch = epoch
    if len(observed_sessions) != profile.session_count:
        _invalid("session_count")
    return records
