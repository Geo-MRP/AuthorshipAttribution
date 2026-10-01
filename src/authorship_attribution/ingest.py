"""Strict UTF-8 JSON and CSV ingestion for chat messages."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
import csv
import io
import json
from pathlib import Path
from typing import cast

from .contracts import IngestConfig, Message, message_key
from .errors import ContractValidationError, ErrorCode


_REQUIRED_SOURCE_FIELDS = ("message_id", "account_id", "game_id", "timestamp_utc", "text")


def _invalid(field_name: str, code: ErrorCode = ErrorCode.INPUT_INVALID) -> None:
    raise ContractValidationError(field_name, code)


def _field_value(
    record: Mapping[str, object], field_name: str, *, required: bool
) -> object:
    if field_name not in record:
        if required:
            _invalid(field_name)
        return None
    return record[field_name]


def validate_message(
    record: Mapping[str, object], *, source_id: str, source_row: int
) -> Message:
    """Validate canonical message fields while keeping diagnostics content-free."""

    if not isinstance(record, Mapping):
        _invalid("object")
    allowed_fields = {
        "schema_version",
        "message_id",
        "account_id",
        "game_id",
        "channel_id",
        "timestamp_utc",
        "text",
        "is_system",
    }
    if any(not isinstance(key, str) for key in record) or set(record) - allowed_fields:
        _invalid("object")
    if not isinstance(source_id, str) or not source_id:
        _invalid("source_id")
    if type(source_row) is not int or source_row < 1:
        _invalid("source_row")
    values: dict[str, object] = {}
    values["schema_version"] = record.get("schema_version", "1.0")
    for name in _REQUIRED_SOURCE_FIELDS:
        values[name] = _field_value(record, name, required=True)
    values["channel_id"] = _field_value(record, "channel_id", required=False)
    values["is_system"] = _field_value(record, "is_system", required=False)
    if values["is_system"] is None:
        values["is_system"] = False
    values["source_id"] = source_id
    values["source_row"] = source_row
    try:
        return Message.from_dict(values)
    except ContractValidationError:
        raise


def _detect_format(path: Path, config: IngestConfig) -> str:
    if config.input_format != "auto":
        return config.input_format
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        return "csv"
    if suffix in {".json", ".jsonl", ".ndjson"}:
        return "json"
    _invalid("input_format")
    raise AssertionError("unreachable")


def _read_utf8(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8", errors="strict", newline="") as source:
            return source.read()
    except (OSError, UnicodeDecodeError):
        _invalid("path")
    raise AssertionError("unreachable")


def _json_records(source: str) -> list[Mapping[str, object]]:
    try:
        parsed: object = json.loads(source, parse_constant=lambda _: (_ for _ in ()).throw(ValueError))
    except (ValueError, json.JSONDecodeError):
        parsed = None
        lines: list[Mapping[str, object]] = []
        try:
            for line in source.splitlines():
                if not line.strip():
                    continue
                value = json.loads(
                    line,
                    parse_constant=lambda _: (_ for _ in ()).throw(ValueError),
                )
                if not isinstance(value, Mapping):
                    _invalid("record")
                lines.append(cast(Mapping[str, object], value))
        except (ValueError, json.JSONDecodeError):
            _invalid("json")
        if lines:
            return lines
        _invalid("json")
    if isinstance(parsed, Mapping):
        records_value: object = parsed.get("messages", parsed)
        if isinstance(records_value, list):
            parsed = records_value
        else:
            return [cast(Mapping[str, object], parsed)]
    if not isinstance(parsed, list) or any(not isinstance(item, Mapping) for item in parsed):
        _invalid("json")
    return [cast(Mapping[str, object], item) for item in parsed]


def _csv_records(source: str) -> list[Mapping[str, object]]:
    try:
        reader = csv.DictReader(io.StringIO(source, newline=""))
        if reader.fieldnames is None:
            _invalid("csv_header")
        rows: list[Mapping[str, object]] = []
        for row in reader:
            if None in row:
                _invalid("csv_row")
            rows.append(cast(Mapping[str, object], row))
        return rows
    except (csv.Error, UnicodeError):
        _invalid("csv")
    raise AssertionError("unreachable")


def _mapped_record(
    source_record: Mapping[str, object], config: IngestConfig
) -> dict[str, object]:
    mapped: dict[str, object] = {}
    for canonical_name in _REQUIRED_SOURCE_FIELDS:
        source_name = config.field_map.get(canonical_name, canonical_name)
        if source_name not in source_record:
            _invalid(canonical_name)
        mapped[canonical_name] = source_record[source_name]
    channel_name = config.field_map.get("channel_id", "channel_id")
    mapped["channel_id"] = source_record.get(channel_name)

    if config.system_flag_field is not None:
        raw_system = source_record.get(config.system_flag_field)
        if isinstance(raw_system, bool):
            mapped["is_system"] = raw_system
        elif isinstance(raw_system, str):
            normalized = raw_system.strip().casefold()
            accepted = {value.casefold() for value in config.system_flag_true_values}
            if normalized in accepted:
                mapped["is_system"] = True
            elif normalized in {"", "false", "0"}:
                mapped["is_system"] = False
            else:
                _invalid("is_system")
        elif type(raw_system) is int and raw_system in (0, 1):
            mapped["is_system"] = bool(raw_system)
        elif raw_system is None:
            mapped["is_system"] = False
        else:
            _invalid("is_system")
    else:
        system_name = config.field_map.get("is_system", "is_system")
        if system_name in source_record:
            mapped["is_system"] = source_record[system_name]
        else:
            mapped["is_system"] = False
    return mapped


def ingest_chat_logs(
    paths: Sequence[Path], *, config: IngestConfig
) -> Iterator[Message]:
    """Yield validated messages, rejecting ambiguous duplicate identities."""

    if not isinstance(config, IngestConfig):
        _invalid("config")
    seen: dict[str, tuple[str, Message]] = {}
    for path in paths:
        if not isinstance(path, Path):
            _invalid("path")
        source_id = path.stem or "source"
        source_key = str(path.resolve())
        file_format = _detect_format(path, config)
        source = _read_utf8(path)
        records = _json_records(source) if file_format == "json" else _csv_records(source)
        within_source: set[str] = set()
        for source_row, source_record in enumerate(records, start=1):
            canonical = _mapped_record(source_record, config)
            message = validate_message(canonical, source_id=source_id, source_row=source_row)
            key = message_key(message.game_id, message.message_id)
            if key in within_source:
                _invalid("message_id")
            within_source.add(key)
            prior = seen.get(key)
            if prior is not None:
                prior_source, prior_message = prior
                equivalent = (
                    prior_message.account_id == message.account_id
                    and prior_message.game_id == message.game_id
                    and prior_message.channel_id == message.channel_id
                    and prior_message.timestamp_utc == message.timestamp_utc
                    and prior_message.text == message.text
                    and prior_message.is_system == message.is_system
                )
                if config.dedupe and prior_source != source_key and equivalent:
                    continue
                _invalid("message_id")
            seen[key] = (source_key, message)
            yield message
