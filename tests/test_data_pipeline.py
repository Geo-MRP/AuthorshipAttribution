from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import pytest

from authorship_attribution.contracts import IngestConfig, Message, PreprocessConfig
from authorship_attribution.errors import ContractValidationError
from authorship_attribution.ingest import ingest_chat_logs, validate_message
from authorship_attribution.preprocess import build_account_profiles
from authorship_attribution.streams import load_normalized_stream, load_raw_stream


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
        [_message(str(i), "abc") for i in range(3)],
        game_terms=[],
        usernames=[],
        config=PreprocessConfig(minimum_messages=1, minimum_chars=1, chunk_target_mode="characters", chunk_target_value=5),
        output_dir=tmp_path / "characters",
    )["acct-a"]
    assert [chunk.n_messages for chunk in load_raw_stream(char_profile)] == [1, 1, 1]


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
