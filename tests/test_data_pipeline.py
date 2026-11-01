from __future__ import annotations

import csv
import json
import logging
import os
from pathlib import Path
from dataclasses import fields, replace
from datetime import datetime
from hashlib import sha256
import time

import authorship_attribution.preprocess as preprocess_module
import pytest

from authorship_attribution.contracts import (
    AccountProfile,
    IngestConfig,
    Message,
    PreprocessConfig,
    TemporalStreamChunk,
    message_key,
)
from authorship_attribution.errors import ContractValidationError, ErrorCode
from authorship_attribution.fingerprints import canonical_json
from authorship_attribution.ingest import ingest_chat_logs, validate_message
from authorship_attribution.preprocess import build_account_profiles
from authorship_attribution.streams import (
    load_normalized_stream,
    load_raw_stream,
    load_temporal_stream,
)


def _message(
    message_id: str,
    text: str,
    timestamp: str = "2026-10-01T12:00:00Z",
    *,
    system: bool = False,
    account_id: str = "acct-a",
) -> Message:
    return Message(
        schema_version="1.0",
        message_id=message_id,
        account_id=account_id,
        game_id="game-a",
        channel_id="channel-a",
        timestamp_utc=timestamp,
        text=text,
        is_system=system,
        source_id="synthetic",
        source_row=1,
    )


def test_json_and_csv_ingest_logically_equivalent(tmp_path: Path) -> None:
    record = {
        "id": "m-1",
        "user": "acct-a",
        "game": "game-a",
        "time": "2026-10-01T12:00:00Z",
        "body": "  Hi! 🙂  ",
        "channel": "channel-a",
        "system": "false",
    }
    json_path = tmp_path / "messages.json"
    json_path.write_text(json.dumps([record]), encoding="utf-8")
    csv_path = tmp_path / "messages.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=list(record))
        writer.writeheader()
        writer.writerow(record)
    config = IngestConfig(
        field_map={
            "message_id": "id",
            "account_id": "user",
            "game_id": "game",
            "timestamp_utc": "time",
            "text": "body",
            "channel_id": "channel",
        },
        system_flag_field="system",
    )
    from_json = list(ingest_chat_logs([json_path], config=config))[0]
    from_csv = list(ingest_chat_logs([csv_path], config=config))[0]
    assert from_json == from_csv
    assert from_json.text == "  Hi! 🙂  "


@pytest.mark.parametrize(
    "record",
    [
        {"message_id": "m", "account_id": "a", "game_id": "g", "timestamp_utc": "yesterday", "text": "SYNTHETIC_PRIVATE_VALUE"},
        {"message_id": "m", "account_id": "a", "game_id": "g", "timestamp_utc": "2026-10-01T12:00:00Z"},
    ],
)
def test_invalid_records_have_sanitized_errors(record: dict[str, object]) -> None:
    with pytest.raises(ContractValidationError) as caught:
        validate_message(record, source_id="synthetic", source_row=1)
    assert "SYNTHETIC_PRIVATE_VALUE" not in str(caught.value)
    assert "yesterday" not in str(caught.value)


def test_invalid_utf8_and_duplicate_ids_rejected(tmp_path: Path) -> None:
    invalid = tmp_path / "bad.json"
    invalid.write_bytes(b'[{"message_id":"m"}\xff]')
    with pytest.raises(ContractValidationError):
        list(ingest_chat_logs([invalid], config=IngestConfig()))

    duplicate = tmp_path / "duplicate.json"
    item = {
        "message_id": "same",
        "account_id": "acct-a",
        "game_id": "game-a",
        "timestamp_utc": "2026-10-01T12:00:00Z",
        "text": "synthetic text",
    }
    duplicate.write_text(json.dumps([item, item]), encoding="utf-8")
    with pytest.raises(ContractValidationError) as caught:
        list(ingest_chat_logs([duplicate], config=IngestConfig()))
    assert str(caught.value) == "message_id:INPUT_INVALID"


def test_deduplication_uses_shared_message_key_and_rejects_conflicts(
    tmp_path: Path,
) -> None:
    base = _message("m/1", "same synthetic", account_id="acct-a")
    equivalent = Message(
        schema_version="1.0",
        message_id=base.message_id,
        account_id=base.account_id,
        game_id=base.game_id,
        channel_id=base.channel_id,
        timestamp_utc=base.timestamp_utc,
        text=base.text,
        is_system=base.is_system,
        source_id="second-source",
        source_row=10,
    )
    result = build_account_profiles(
        [base, equivalent],
        game_terms=[],
        usernames=[],
        config=PreprocessConfig(minimum_messages=1, minimum_chars=1),
        output_dir=tmp_path / "dedupe",
    )
    assert message_key(base.game_id, base.message_id) == '["game-a","m/1"]'
    assert result["acct-a"].n_messages == 1

    conflicting = _message("m/1", "different synthetic")
    with pytest.raises(ContractValidationError):
        build_account_profiles(
            [base, conflicting],
            game_terms=[],
            usernames=[],
            config=PreprocessConfig(),
            output_dir=tmp_path / "conflict",
        )


def test_raw_preservation_masking_and_projection_isolation(tmp_path: Path) -> None:
    exact = "  MiXeD 😺 !!!!!!!! sooooo   "
    messages = [
        _message("empty", ""),
        _message("system", "system synthetic", system=True),
        _message("one", exact + " @Player https://example.test/a 123 Dragon"),
        _message("space", " \t  ", "2026-10-01T12:01:00Z"),
    ]
    profiles = build_account_profiles(
        messages,
        game_terms=["Dragon"],
        usernames=["Player"],
        config=PreprocessConfig(minimum_messages=1, minimum_chars=1, chunk_target_mode="messages", chunk_target_value=50),
        output_dir=tmp_path,
    )
    profile = profiles["acct-a"]
    raw = load_raw_stream(profile)
    normalized = load_normalized_stream(profile)
    assert raw[0].raw_text == [exact + " @Player https://example.test/a 123 Dragon", " \t  "]
    assert "normalized_text" not in raw[0].to_dict()
    assert "raw_text" not in normalized[0].to_dict()
    assert "[USER]" in normalized[0].normalized_text
    assert "[URL]" in normalized[0].normalized_text
    assert "[NUM]" in normalized[0].normalized_text
    assert "[GAME_TERM]" in normalized[0].normalized_text
    assert raw[0].raw_text[0].encode("utf-8").decode("utf-8") == raw[0].raw_text[0]


def test_session_boundary_filter_histogram_and_chunk_modes(tmp_path: Path) -> None:
    entries = [
        _message("a", "aa", "2026-10-01T00:00:00Z"),
        _message("b", "bb", "2026-10-01T00:29:00Z"),
        _message("c", "cc", "2026-10-01T00:59:00Z"),
        _message("d", "dd", "2026-10-01T01:00:00Z"),
    ]
    config = PreprocessConfig(
        minimum_messages=1,
        minimum_chars=1,
        utc_hour_filter=[0],
        chunk_target_mode="messages",
        chunk_target_value=1,
    )
    profile = build_account_profiles(
        entries,
        game_terms=[],
        usernames=[],
        config=config,
        output_dir=tmp_path,
    )["acct-a"]
    assert profile.activity_histogram_utc[0] == 3
    assert profile.activity_histogram_utc[1] == 1
    assert profile.n_messages == 3
    assert profile.session_count == 2
    chunks = load_raw_stream(profile)
    assert [chunk.n_messages for chunk in chunks] == [1, 1, 1]
    assert len({chunk.session_id for chunk in chunks[:2]}) == 1
    assert chunks[2].session_id != chunks[1].session_id

    char_profile = build_account_profiles(
        [
            _message("long", "abcdefghij"),
            _message("short-1", "a", "2026-10-01T12:01:00Z"),
            _message("short-2", "b", "2026-10-01T12:02:00Z"),
        ],
        game_terms=[],
        usernames=[],
        config=PreprocessConfig(minimum_messages=1, minimum_chars=1, chunk_target_mode="characters", chunk_target_value=5),
        output_dir=tmp_path / "characters",
    )["acct-a"]
    assert [chunk.raw_text for chunk in load_raw_stream(char_profile)] == [
        ["abcdefghij"],
        ["a", "b"],
    ]
    assert [chunk.n_messages for chunk in load_raw_stream(char_profile)] == [1, 2]


def test_empty_post_filter_profile_and_no_accepted_omission(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    accepted_but_filtered = _message("filtered", "kept before filter", "2026-10-01T12:00:00Z")
    no_accepted_a = _message("empty", "", account_id="acct-empty")
    no_accepted_b = _message(
        "system", "system synthetic", system=True, account_id="acct-system"
    )
    with caplog.at_level(logging.INFO):
        profiles = build_account_profiles(
            [accepted_but_filtered, no_accepted_a, no_accepted_b],
            game_terms=[],
            usernames=[],
            config=PreprocessConfig(utc_hour_filter=[1]),
            output_dir=tmp_path,
        )

    assert list(profiles) == ["acct-a"]
    profile = profiles["acct-a"]
    assert profile.status == "INSUFFICIENT_DATA"
    assert profile.game_ids == ["game-a"]
    assert profile.active_months_utc == []
    assert profile.n_messages == profile.n_chars == profile.session_count == 0
    assert profile.chunk_ids == []
    assert load_raw_stream(profile) == []
    assert load_normalized_stream(profile) == []
    assert load_temporal_stream(profile) == []
    assert Path(profile.temporal_stream_path).read_bytes() == b""
    assert tuple(field.name for field in fields(AccountProfile)) == (
        "schema_version",
        "account_id",
        "game_ids",
        "primary_game_id",
        "primary_channel_id",
        "active_months_utc",
        "n_messages",
        "n_chars",
        "session_count",
        "activity_histogram_utc",
        "utc_hour_filter",
        "chunk_ids",
        "raw_stream_path",
        "normalized_stream_path",
        "status",
        "minimum_messages",
        "minimum_chars",
        "preprocessing_fingerprint",
        "temporal_stream_path",
    )
    assert "omitted_accounts=2" in caplog.text
    assert "kept before filter" not in caplog.text


def test_minimum_status_determinism_and_no_text_in_logs(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    marker = "SYNTHETIC_SECRET_PHRASE"
    config = PreprocessConfig(minimum_messages=2, minimum_chars=50)
    with caplog.at_level(logging.DEBUG):
        first = build_account_profiles(
            [_message("m", marker)], game_terms=[], usernames=[], config=config, output_dir=tmp_path
        )
        second = build_account_profiles(
            [_message("m", marker)], game_terms=[], usernames=[], config=config, output_dir=tmp_path
        )
    assert first["acct-a"].status == "INSUFFICIENT_DATA"
    assert first["acct-a"].to_dict() == second["acct-a"].to_dict()
    assert marker not in caplog.text


def _phase2_message(
    message_id: str,
    text: str,
    timestamp: str,
    *,
    game_id: str = "game-a",
    account_id: str = "acct-a",
    system: bool = False,
) -> Message:
    return Message(
        schema_version="1.0",
        message_id=message_id,
        account_id=account_id,
        game_id=game_id,
        channel_id="channel-a",
        timestamp_utc=timestamp,
        text=text,
        is_system=system,
        source_id="synthetic",
        source_row=1,
    )


@pytest.fixture
def phase2_profile(tmp_path: Path) -> AccountProfile:
    messages = [
        _phase2_message("a", "aa", "2026-10-01T12:00:00Z"),
        _phase2_message("b", "bbb", "2026-10-01T12:29:00Z"),
        _phase2_message("c", "cccc", "2026-10-01T12:59:00Z"),
    ]
    return build_account_profiles(
        messages,
        game_terms=[],
        usernames=[],
        config=PreprocessConfig(
            minimum_messages=1,
            minimum_chars=1,
            chunk_target_mode="messages",
            chunk_target_value=1,
        ),
        output_dir=tmp_path,
    )["acct-a"]


def test_temporal_projection_content_counts_and_canonical_bytes(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    marker = "SYNTHETIC_PRIVATE_TEXT_8520"
    messages = [
        _phase2_message("a", "a😺", "2026-10-01T12:00:00Z"),
        _phase2_message("b", " \t ", "2026-10-01T12:00:30Z"),
        _phase2_message("c", marker, "2026-10-01T12:01:00Z"),
        _phase2_message("empty", "", "2026-10-01T12:02:00Z"),
        _phase2_message("system", marker, "2026-10-01T12:03:00Z", system=True),
    ]
    with caplog.at_level(logging.DEBUG):
        profile = build_account_profiles(
            messages,
            game_terms=[],
            usernames=[],
            config=PreprocessConfig(
                minimum_messages=1,
                minimum_chars=1,
                chunk_target_mode="messages",
                chunk_target_value=1,
            ),
            output_dir=tmp_path,
        )["acct-a"]

    expected_name = sha256(b"acct-a").hexdigest() + ".ndjson"
    temporal_path = Path(profile.temporal_stream_path)
    assert temporal_path == tmp_path.resolve() / "temporal" / expected_name
    assert temporal_path.is_absolute()
    temporal = load_temporal_stream(profile)
    raw = load_raw_stream(profile)
    normalized = load_normalized_stream(profile)
    assert all(isinstance(item, TemporalStreamChunk) for item in temporal)
    assert [item.chunk_id for item in temporal] == profile.chunk_ids
    assert [item.chunk_id for item in temporal] == [item.chunk_id for item in raw]
    assert [item.chunk_id for item in temporal] == [item.chunk_id for item in normalized]
    assert [item.char_counts for item in temporal] == [[2], [3], [len(marker)]]
    assert [item.timestamps_utc for item in temporal] == [
        ["2026-10-01T12:00:00Z"],
        ["2026-10-01T12:00:30Z"],
        ["2026-10-01T12:01:00Z"],
    ]
    assert profile.n_messages == 3
    assert profile.n_chars == 5 + len(marker)
    assert profile.session_count == 1
    assert len({item.session_id for item in temporal}) == 1
    assert temporal_path.read_bytes() == b"".join(
        canonical_json(item) + b"\n" for item in temporal
    )
    assert set(temporal[0].to_dict()) == {
        "schema_version",
        "chunk_id",
        "account_id",
        "session_id",
        "timestamps_utc",
        "char_counts",
        "n_messages",
        "n_chars",
    }
    assert marker.encode("utf-8") not in temporal_path.read_bytes()
    assert marker not in repr(temporal)
    assert marker not in caplog.text
    assert "raw_text" not in temporal_path.read_text(encoding="utf-8")
    assert "normalized_text" not in temporal_path.read_text(encoding="utf-8")


def test_temporal_session_boundaries_and_split_chunks(
    phase2_profile: AccountProfile,
) -> None:
    temporal = load_temporal_stream(phase2_profile)
    assert len(temporal) == 3
    assert temporal[0].session_id == temporal[1].session_id
    assert temporal[2].session_id != temporal[1].session_id
    assert phase2_profile.session_count == 2
    assert [item.n_messages for item in temporal] == [1, 1, 1]
    assert [item.n_chars for item in temporal] == [2, 3, 4]


def test_temporal_character_chunks_match_raw_chunks(tmp_path: Path) -> None:
    messages = [
        _phase2_message("long", "abcdefghij", "2026-10-01T12:00:00Z"),
        _phase2_message("short-a", "a", "2026-10-01T12:01:00Z"),
        _phase2_message("short-b", "b", "2026-10-01T12:02:00Z"),
    ]
    profile = build_account_profiles(
        messages,
        game_terms=[],
        usernames=[],
        config=PreprocessConfig(
            minimum_messages=1,
            minimum_chars=1,
            chunk_target_mode="characters",
            chunk_target_value=5,
        ),
        output_dir=tmp_path,
    )["acct-a"]
    raw = load_raw_stream(profile)
    temporal = load_temporal_stream(profile)
    assert [item.raw_text for item in raw] == [["abcdefghij"], ["a", "b"]]
    assert [item.char_counts for item in temporal] == [[10], [1, 1]]
    assert [item.n_messages for item in temporal] == [1, 2]
    assert [item.chunk_id for item in temporal] == [item.chunk_id for item in raw]
    assert temporal[0].session_id == temporal[1].session_id


def test_temporal_empty_filtered_profile(tmp_path: Path) -> None:
    profile = build_account_profiles(
        [_phase2_message("m", "kept before filter", "2026-10-01T12:00:00Z")],
        game_terms=[],
        usernames=[],
        config=PreprocessConfig(utc_hour_filter=[1]),
        output_dir=tmp_path,
    )["acct-a"]
    assert profile.status == "INSUFFICIENT_DATA"
    assert profile.n_messages == profile.n_chars == profile.session_count == 0
    assert profile.chunk_ids == []
    assert load_raw_stream(profile) == []
    assert load_normalized_stream(profile) == []
    assert load_temporal_stream(profile) == []
    assert Path(profile.raw_stream_path).read_bytes() == b""
    assert Path(profile.normalized_stream_path).read_bytes() == b""
    assert Path(profile.temporal_stream_path).read_bytes() == b""


def test_temporal_partial_utc_filter_retains_prefilter_histogram(tmp_path: Path) -> None:
    messages = [
        _phase2_message("before", "morning", "2026-10-01T12:00:00Z"),
        _phase2_message("after", "kept", "2026-10-01T13:00:00Z"),
    ]
    profile = build_account_profiles(
        messages,
        game_terms=[],
        usernames=[],
        config=PreprocessConfig(
            minimum_messages=1,
            minimum_chars=1,
            utc_hour_filter=[13],
        ),
        output_dir=tmp_path,
    )["acct-a"]
    assert profile.activity_histogram_utc[12] == 1
    assert profile.activity_histogram_utc[13] == 1
    assert profile.n_messages == 1
    assert profile.n_chars == 4
    temporal = load_temporal_stream(profile)
    assert len(temporal) == 1
    assert temporal[0].timestamps_utc == ["2026-10-01T13:00:00Z"]
    assert temporal[0].char_counts == [4]
    assert load_raw_stream(profile)[0].raw_text == ["kept"]


def test_temporal_preserves_fractional_and_pre_epoch_timestamps(tmp_path: Path) -> None:
    timestamps = [
        "1969-12-31T23:59:59.999999Z",
        "1970-01-01T00:00:00.123456789Z",
    ]
    profile = build_account_profiles(
        [
            _phase2_message("a", "x", timestamps[0]),
            _phase2_message("b", "y", timestamps[1]),
        ],
        game_terms=[],
        usernames=[],
        config=PreprocessConfig(minimum_messages=1, minimum_chars=1),
        output_dir=tmp_path,
    )["acct-a"]
    temporal = load_temporal_stream(profile)
    assert len(temporal) == 1
    assert temporal[0].timestamps_utc == timestamps
    epochs = [
        datetime.fromisoformat(f"{timestamp[:-1]}+00:00").timestamp()
        for timestamp in temporal[0].timestamps_utc
    ]
    assert epochs[0] < 0.0 < epochs[1]
    assert epochs == sorted(epochs)


def test_temporal_tie_order_and_repeat_are_deterministic(tmp_path: Path) -> None:
    messages = [
        _phase2_message("same", "zeta text", "2026-10-01T12:00:00Z", game_id="zeta"),
        _phase2_message("same", "alpha text", "2026-10-01T12:00:00Z", game_id="alpha"),
    ]
    config = PreprocessConfig(
        minimum_messages=1,
        minimum_chars=1,
        chunk_target_mode="messages",
        chunk_target_value=1,
    )
    first = build_account_profiles(
        messages, game_terms=[], usernames=[], config=config, output_dir=tmp_path
    )["acct-a"]
    paths = [
        Path(first.raw_stream_path),
        Path(first.normalized_stream_path),
        Path(first.temporal_stream_path),
        tmp_path.resolve() / "profiles" / (sha256(b"acct-a").hexdigest() + ".json"),
    ]
    first_bytes = [path.read_bytes() for path in paths]
    assert [item.raw_text for item in load_raw_stream(first)] == [
        ["alpha text"],
        ["zeta text"],
    ]
    second = build_account_profiles(
        list(reversed(messages)),
        game_terms=[],
        usernames=[],
        config=config,
        output_dir=tmp_path,
    )["acct-a"]
    assert second.to_dict() == first.to_dict()
    assert [path.read_bytes() for path in paths] == first_bytes


def test_temporal_reader_never_opens_text_streams(
    phase2_profile: AccountProfile, monkeypatch: pytest.MonkeyPatch
) -> None:
    blocked = {
        Path(phase2_profile.raw_stream_path),
        Path(phase2_profile.normalized_stream_path),
    }
    original_open = Path.open

    def guarded_open(self: Path, *args: object, **kwargs: object) -> object:
        if self in blocked:
            pytest.fail("temporal reader opened a text-bearing stream")
        return original_open(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    assert len(load_temporal_stream(phase2_profile)) == 3


def test_temporal_reader_legacy_path_is_not_fabricated(
    phase2_profile: AccountProfile,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    legacy = replace(phase2_profile, temporal_stream_path="unused-temporal-path")
    with pytest.raises(ContractValidationError) as caught:
        load_temporal_stream(legacy)
    assert str(caught.value) == "path:ARTIFACT_INVALID"


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("blank", "line:ARTIFACT_INVALID"),
        ("invalid_utf8", "path:ARTIFACT_INVALID"),
        ("invalid_json", "path:ARTIFACT_INVALID"),
        ("nonfinite", "path:ARTIFACT_INVALID"),
        ("non_object", "object:ARTIFACT_INVALID"),
        ("unknown_field", "object:INPUT_INVALID"),
        ("missing_field", "session_id:INPUT_INVALID"),
        ("bad_schema", "schema_version:INPUT_INVALID"),
        ("bad_timestamp", "timestamps_utc:INPUT_INVALID"),
        ("bad_char_type", "char_counts:INPUT_INVALID"),
        ("negative_char_count", "char_counts:INPUT_INVALID"),
        ("wrong_account", "account_id:ARTIFACT_INVALID"),
        ("wrong_chunk", "chunk_ids:ARTIFACT_INVALID"),
        ("reordered_chunks", "chunk_ids:ARTIFACT_INVALID"),
        ("duplicate_chunk", "chunk_ids:ARTIFACT_INVALID"),
        ("bad_record_count", "n_messages:INPUT_INVALID"),
        ("bad_record_chars", "n_chars:INPUT_INVALID"),
        ("profile_message_total", "n_messages:ARTIFACT_INVALID"),
        ("profile_char_total", "n_chars:ARTIFACT_INVALID"),
        ("cross_chunk_time", "timestamps_utc:ARTIFACT_INVALID"),
        ("session_reappears", "session_id:ARTIFACT_INVALID"),
        ("session_total", "session_count:ARTIFACT_INVALID"),
    ],
)
def test_temporal_reader_rejects_malformed_and_inconsistent_streams(
    phase2_profile: AccountProfile,
    case: str,
    expected: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    marker = "SYNTHETIC_PRIVATE_ERROR_MARKER_2851"
    path = Path(phase2_profile.temporal_stream_path)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    if case == "blank":
        path.write_bytes(b"\n")
    elif case == "invalid_utf8":
        path.write_bytes(b"\xff")
    elif case == "invalid_json":
        path.write_bytes(b"{\n")
    elif case == "nonfinite":
        path.write_bytes(b'{"value":NaN}\n')
    elif case == "non_object":
        path.write_bytes(b"[]\n")
    else:
        if case == "unknown_field":
            records[0][marker] = marker
        elif case == "missing_field":
            del records[0]["session_id"]
        elif case == "bad_schema":
            records[0]["schema_version"] = "2.0"
        elif case == "bad_timestamp":
            records[0]["timestamps_utc"] = [marker]
        elif case == "bad_char_type":
            records[0]["char_counts"] = [True]
        elif case == "negative_char_count":
            records[0]["char_counts"] = [-1]
        elif case == "wrong_account":
            records[0]["account_id"] = marker
        elif case == "wrong_chunk":
            records[0]["chunk_id"] = marker
        elif case == "reordered_chunks":
            records.reverse()
        elif case == "duplicate_chunk":
            records[1] = dict(records[0])
        elif case == "bad_record_count":
            records[0]["n_messages"] = 2
        elif case == "bad_record_chars":
            records[0]["n_chars"] += 1
        elif case == "profile_message_total":
            records[0]["timestamps_utc"].append(records[0]["timestamps_utc"][0])
            records[0]["char_counts"].append(0)
            records[0]["n_messages"] += 1
        elif case == "profile_char_total":
            records[0]["char_counts"][0] += 1
            records[0]["n_chars"] += 1
        elif case == "cross_chunk_time":
            records[1]["timestamps_utc"] = ["2026-10-01T11:59:00Z"]
        elif case == "session_reappears":
            records[1]["session_id"] = "other-session"
            records[2]["session_id"] = records[0]["session_id"]
        elif case == "session_total":
            records[2]["session_id"] = records[0]["session_id"]
        else:
            raise AssertionError("unknown test case")
        path.write_bytes(b"".join(canonical_json(item) + b"\n" for item in records))

    with caplog.at_level(logging.DEBUG):
        with pytest.raises(ContractValidationError) as caught:
            load_temporal_stream(phase2_profile)
    assert str(caught.value) == expected
    assert marker not in str(caught.value)
    assert marker not in caplog.text


def test_temporal_reader_rejects_non_profile() -> None:
    with pytest.raises(ContractValidationError) as caught:
        load_temporal_stream(None)  # type: ignore[arg-type]
    assert str(caught.value) == "profile:ARTIFACT_INVALID"


def test_expiration_includes_temporal_and_preserves_fresh_files(tmp_path: Path) -> None:
    old_paths = []
    fresh_paths = []
    for folder_name in ("raw", "normalized", "temporal"):
        folder = tmp_path / folder_name
        folder.mkdir(parents=True)
        old_path = folder / "expired.ndjson"
        fresh_path = folder / "fresh.ndjson"
        old_path.write_bytes(b"old\n")
        fresh_path.write_bytes(b"fresh\n")
        old_time = time.time() - 8 * 86400
        os.utime(old_path, (old_time, old_time))
        old_paths.append(old_path)
        fresh_paths.append(fresh_path)

    build_account_profiles(
        [_phase2_message("m", "x", "2026-10-01T12:00:00Z")],
        game_terms=[],
        usernames=[],
        config=PreprocessConfig(minimum_messages=1, minimum_chars=1, cache_retention_days=7),
        output_dir=tmp_path,
    )
    assert all(not path.exists() for path in old_paths)
    assert all(path.read_bytes() == b"fresh\n" for path in fresh_paths)


def test_zero_retention_cleans_temporal_without_removing_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registered: set[Path] = set()
    monkeypatch.setattr(preprocess_module, "_REGISTERED_CLEANUP", registered)
    monkeypatch.setattr(preprocess_module.atexit, "register", lambda callback: None)
    monkeypatch.setattr(preprocess_module.atexit, "unregister", lambda callback: None)
    profile = build_account_profiles(
        [_phase2_message("m", "x", "2026-10-01T12:00:00Z")],
        game_terms=[],
        usernames=[],
        config=PreprocessConfig(minimum_messages=1, minimum_chars=1, cache_retention_days=0),
        output_dir=tmp_path,
    )["acct-a"]
    stream_paths = {
        Path(profile.raw_stream_path),
        Path(profile.normalized_stream_path),
        Path(profile.temporal_stream_path),
    }
    profile_path = tmp_path.resolve() / "profiles" / (sha256(b"acct-a").hexdigest() + ".json")
    assert registered == stream_paths
    assert all(path.exists() for path in stream_paths)
    preprocess_module._cleanup_at_exit()
    assert all(not path.exists() for path in stream_paths)
    assert profile_path.exists()


def test_positive_retention_rewrite_unmarks_prior_zero_retention_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registered: set[Path] = set()
    monkeypatch.setattr(preprocess_module, "_REGISTERED_CLEANUP", registered)
    monkeypatch.setattr(preprocess_module.atexit, "register", lambda callback: None)
    monkeypatch.setattr(preprocess_module.atexit, "unregister", lambda callback: None)
    messages = [_phase2_message("m", "x", "2026-10-01T12:00:00Z")]
    first = build_account_profiles(
        messages,
        game_terms=[],
        usernames=[],
        config=PreprocessConfig(minimum_messages=1, minimum_chars=1, cache_retention_days=0),
        output_dir=tmp_path,
    )["acct-a"]
    assert Path(first.temporal_stream_path) in registered
    second = build_account_profiles(
        messages,
        game_terms=[],
        usernames=[],
        config=PreprocessConfig(minimum_messages=1, minimum_chars=1, cache_retention_days=7),
        output_dir=tmp_path,
    )["acct-a"]
    assert registered == set()
    preprocess_module._cleanup_at_exit()
    assert Path(second.temporal_stream_path).exists()


def test_temporal_preprocessing_and_reader_make_no_network_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import socket

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    profile = build_account_profiles(
        [_phase2_message("m", "x", "2026-10-01T12:00:00Z")],
        game_terms=[],
        usernames=[],
        config=PreprocessConfig(minimum_messages=1, minimum_chars=1),
        output_dir=tmp_path,
    )["acct-a"]
    assert len(load_temporal_stream(profile)) == 1


def test_failed_temporal_replace_is_sanitized_and_does_not_publish_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = "SYNTHETIC_PRIVATE_IO_MARKER_6381"
    original_replace = preprocess_module.os.replace

    def guarded_replace(source: object, destination: object) -> None:
        if Path(destination).parent.name == "temporal":
            raise OSError(marker)
        original_replace(source, destination)

    monkeypatch.setattr(preprocess_module.os, "replace", guarded_replace)
    with pytest.raises(ContractValidationError) as caught:
        build_account_profiles(
            [_phase2_message("m", marker, "2026-10-01T12:00:00Z")],
            game_terms=[],
            usernames=[],
            config=PreprocessConfig(minimum_messages=1, minimum_chars=1),
            output_dir=tmp_path,
        )
    assert caught.value.code is ErrorCode.ARTIFACT_INVALID
    assert str(caught.value) == "path:ARTIFACT_INVALID"
    assert marker not in str(caught.value)
    assert not (tmp_path / "profiles" / (sha256(b"acct-a").hexdigest() + ".json")).exists()
    assert list((tmp_path / "temporal").glob(".*")) == []
