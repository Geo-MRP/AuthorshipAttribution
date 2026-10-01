"""Privacy-preserving preprocessing, sessionization, and account artifacts."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta
import atexit
import logging
import os
from pathlib import Path
import re
import tempfile
import time
import unicodedata

from .contracts import (
    AccountProfile,
    Chunk,
    Message,
    PreprocessConfig,
    message_key,
)
from .errors import ContractValidationError, ErrorCode
from .fileformats import write_json_artifact
from .fingerprints import (
    canonical_json,
    fingerprint,
    preprocessing_fingerprint,
    sha256_hex,
)


_REGISTERED_CLEANUP: set[Path] = set()
_LOGGER = logging.getLogger(__name__)


def _invalid(field_name: str, code: ErrorCode = ErrorCode.INPUT_INVALID) -> None:
    raise ContractValidationError(field_name, code)


def _timestamp(value: str) -> datetime:
    try:
        return datetime.fromisoformat(f"{value[:-1]}+00:00")
    except (TypeError, ValueError):
        _invalid("timestamp_utc")
    raise AssertionError("unreachable")


def _cleanup_at_exit() -> None:
    for path in sorted(_REGISTERED_CLEANUP, key=str):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _write_ndjson(path: Path, records: Sequence[object]) -> None:
    temporary_name: str | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix=f".{path.name}.", delete=False
        ) as output:
            temporary_name = output.name
            for record in records:
                output.write(canonical_json(record))
                output.write(b"\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, path)
    except (OSError, TypeError, ValueError):
        _invalid("path", ErrorCode.ARTIFACT_INVALID)
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink(missing_ok=True)
            except OSError:
                pass


def _compile_mask(terms: Sequence[str]) -> re.Pattern[str] | None:
    normalized = {unicodedata.normalize("NFC", term.lstrip("@")) for term in terms if term}
    if not normalized:
        return None
    alternatives = sorted(
        normalized, key=lambda term: (-len(term), term.casefold(), term)
    )
    body = "|".join(re.escape(term) for term in alternatives)
    return re.compile(rf"(?<![\w@])@?(?:{body})(?!\w)", re.IGNORECASE)


def _compile_game_terms(terms: Sequence[str]) -> re.Pattern[str] | None:
    normalized = {unicodedata.normalize("NFC", term) for term in terms if term}
    if not normalized:
        return None
    alternatives = sorted(
        normalized, key=lambda term: (-len(term), term.casefold(), term)
    )
    body = "|".join(re.escape(term) for term in alternatives)
    return re.compile(rf"(?<!\w)(?:{body})(?!\w)", re.IGNORECASE)


def _normalize_text(
    text: str,
    *,
    username_pattern: re.Pattern[str] | None,
    game_term_pattern: re.Pattern[str] | None,
) -> str:
    normalized = unicodedata.normalize("NFC", text)
    if username_pattern is not None:
        normalized = username_pattern.sub("[USER]", normalized)
    normalized = re.sub(r"(?<!\w)@[\w.-]+", "[USER]", normalized, flags=re.UNICODE)
    normalized = re.sub(r"(?i)\b(?:https?://|www\.)[^\s]+", "[URL]", normalized)
    normalized = re.sub(r"(?<!\w)\d+(?:[.,]\d+)*(?!\w)", "[NUM]", normalized)
    if game_term_pattern is not None:
        normalized = game_term_pattern.sub("[GAME_TERM]", normalized)
    return re.sub(r"\s+", " ", normalized, flags=re.UNICODE).strip()


def _atomic_id(kind: str, components: object) -> str:
    return f"{kind}-{sha256_hex(canonical_json(components))}"


def _stream_paths(output_dir: Path, account_id: str) -> tuple[Path, Path, Path]:
    account_key = sha256_hex(account_id.encode("utf-8"))
    root = output_dir.resolve()
    return (
        root / "profiles" / f"{account_key}.json",
        root / "raw" / f"{account_key}.ndjson",
        root / "normalized" / f"{account_key}.ndjson",
    )


def _expire_old_streams(output_dir: Path, retention_days: int) -> None:
    if retention_days == 0:
        return
    cutoff = time.time() - timedelta(days=retention_days).total_seconds()
    for folder_name in ("raw", "normalized"):
        folder = output_dir / folder_name
        if not folder.is_dir():
            continue
        try:
            paths = sorted(folder.glob("*.ndjson"), key=str)
        except OSError:
            _invalid("path", ErrorCode.ARTIFACT_INVALID)
        for path in paths:
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
            except OSError:
                _invalid("path", ErrorCode.ARTIFACT_INVALID)


def build_account_profiles(
    messages: Iterable[Message],
    *,
    game_terms: Sequence[str],
    usernames: Sequence[str],
    config: PreprocessConfig,
    output_dir: Path,
) -> Mapping[str, AccountProfile]:
    """Create deterministic, physically isolated streams and profile artifacts."""

    if not isinstance(config, PreprocessConfig):
        _invalid("config")
    if not isinstance(output_dir, Path):
        _invalid("output_dir")
    if any(not isinstance(term, str) for term in (*game_terms, *usernames)):
        _invalid("terms")

    source_by_account: dict[str, list[Message]] = defaultdict(list)
    by_message_key: dict[str, Message] = {}
    for message in messages:
        if not isinstance(message, Message):
            _invalid("message")
        try:
            message.text.encode("utf-8", errors="strict")
        except UnicodeEncodeError:
            _invalid("text")
        key = message_key(message.game_id, message.message_id)
        previous = by_message_key.get(key)
        if previous is not None:
            same_payload = (
                previous.account_id == message.account_id
                and previous.channel_id == message.channel_id
                and previous.timestamp_utc == message.timestamp_utc
                and previous.text == message.text
                and previous.is_system == message.is_system
            )
            if not same_payload:
                _invalid("message_id")
            if (message.source_id, message.source_row) < (
                previous.source_id,
                previous.source_row,
            ):
                by_message_key[key] = message
        else:
            by_message_key[key] = message

    for message in by_message_key.values():
        source_by_account[message.account_id].append(message)

    normalized_config = PreprocessConfig.from_dict(
        {
            **config.to_dict(),
            "utc_hour_filter": (
                None if config.utc_hour_filter is None else sorted(config.utc_hour_filter)
            ),
        }
    )
    terms_digest = fingerprint({"game_terms": sorted(set(game_terms))})
    usernames_digest = fingerprint({"usernames": sorted(set(usernames))})
    preprocess_fp = preprocessing_fingerprint(
        normalized_config, terms_digest, usernames_digest
    )
    username_pattern = _compile_mask(usernames)
    game_term_pattern = _compile_game_terms(game_terms)

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        _invalid("output_dir", ErrorCode.ARTIFACT_INVALID)
    _expire_old_streams(output_dir, config.cache_retention_days)

    profiles: dict[str, AccountProfile] = {}
    omitted_accounts = 0
    for account_id in sorted(source_by_account):
        pre_filter = sorted(
            (
                message
                for message in source_by_account[account_id]
                if not message.is_system and len(message.text) > 0
            ),
            key=lambda item: (_timestamp(item.timestamp_utc), item.message_id),
        )
        if not pre_filter:
            omitted_accounts += 1
            continue
        histogram = [0] * 24
        for message in pre_filter:
            histogram[_timestamp(message.timestamp_utc).hour] += 1
        included = [
            message
            for message in pre_filter
            if config.utc_hour_filter is None
            or _timestamp(message.timestamp_utc).hour in config.utc_hour_filter
        ]

        sessions: list[list[Message]] = []
        for message in included:
            if not sessions:
                sessions.append([message])
                continue
            previous = sessions[-1][-1]
            gap = _timestamp(message.timestamp_utc) - _timestamp(previous.timestamp_utc)
            if gap < timedelta(minutes=config.idle_gap_minutes):
                sessions[-1].append(message)
            else:
                sessions.append([message])

        chunks: list[Chunk] = []
        for session_index, session in enumerate(sessions):
            session_id = _atomic_id(
                "session",
                [account_id, session_index, [message_key(m.game_id, m.message_id) for m in session]],
            )
            current: list[Message] = []
            current_chars = 0

            def flush_chunk() -> None:
                nonlocal current, current_chars
                if not current:
                    return
                ids = [message_key(item.game_id, item.message_id) for item in current]
                raw_text = [item.text for item in current]
                normalized_text = "\n".join(
                    _normalize_text(
                        item.text,
                        username_pattern=username_pattern,
                        game_term_pattern=game_term_pattern,
                    )
                    for item in current
                )
                chunk_id = _atomic_id("chunk", [account_id, session_id, ids])
                chunks.append(
                    Chunk(
                        schema_version="1.0",
                        chunk_id=chunk_id,
                        account_id=account_id,
                        session_id=session_id,
                        message_ids=ids,
                        start_timestamp_utc=current[0].timestamp_utc,
                        end_timestamp_utc=current[-1].timestamp_utc,
                        n_messages=len(current),
                        n_chars=sum(len(text) for text in raw_text),
                        raw_text=raw_text,
                        normalized_text=normalized_text,
                        normalized_sha256=sha256_hex(normalized_text.encode("utf-8")),
                    )
                )
                current = []
                current_chars = 0

            for message in session:
                message_chars = len(message.text)
                if current and config.chunk_target_mode == "messages" and len(current) >= config.chunk_target_value:
                    flush_chunk()
                elif (
                    current
                    and config.chunk_target_mode == "characters"
                    and current_chars + message_chars > config.chunk_target_value
                ):
                    flush_chunk()
                current.append(message)
                current_chars += message_chars
                if config.chunk_target_mode == "messages" and len(current) >= config.chunk_target_value:
                    flush_chunk()
                elif config.chunk_target_mode == "characters" and current_chars >= config.chunk_target_value:
                    flush_chunk()
            flush_chunk()

        raw_records = [chunk.to_raw_stream_chunk() for chunk in chunks]
        normalized_records = [chunk.to_normalized_stream_chunk() for chunk in chunks]
        profile_path, raw_path, normalized_path = _stream_paths(output_dir, account_id)
        _REGISTERED_CLEANUP.discard(raw_path)
        _REGISTERED_CLEANUP.discard(normalized_path)
        _write_ndjson(raw_path, raw_records)
        _write_ndjson(normalized_path, normalized_records)
        if config.cache_retention_days == 0:
            _REGISTERED_CLEANUP.update((raw_path, normalized_path))
            atexit.unregister(_cleanup_at_exit)
            atexit.register(_cleanup_at_exit)

        game_counts = Counter(message.game_id for message in pre_filter)
        game_ids = sorted({message.game_id for message in pre_filter})
        primary_game_id = min(game_counts, key=lambda game: (-game_counts[game], game))
        channel_counts = Counter(
            message.channel_id
            for message in pre_filter
            if message.channel_id is not None
        )
        primary_channel_id = (
            min(channel_counts, key=lambda channel: (-channel_counts[channel], channel))
            if channel_counts
            else None
        )
        months = sorted(
            {
                _timestamp(message.timestamp_utc).strftime("%Y-%m")
                for message in included
            }
        )
        n_chars = sum(len(message.text) for message in included)
        profile = AccountProfile(
            schema_version="1.0",
            account_id=account_id,
            game_ids=game_ids,
            primary_game_id=primary_game_id,
            primary_channel_id=primary_channel_id,
            active_months_utc=months,
            n_messages=len(included),
            n_chars=n_chars,
            session_count=len(sessions),
            activity_histogram_utc=histogram,
            utc_hour_filter=(
                None if config.utc_hour_filter is None else sorted(config.utc_hour_filter)
            ),
            chunk_ids=[chunk.chunk_id for chunk in chunks],
            raw_stream_path=str(raw_path),
            normalized_stream_path=str(normalized_path),
            status=(
                "INSUFFICIENT_DATA"
                if len(included) < config.minimum_messages or n_chars < config.minimum_chars
                else "READY"
            ),
            minimum_messages=config.minimum_messages,
            minimum_chars=config.minimum_chars,
            preprocessing_fingerprint=preprocess_fp,
        )
        try:
            profile_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            _invalid("output_dir", ErrorCode.ARTIFACT_INVALID)
        write_json_artifact(profile, profile_path)
        profiles[account_id] = profile
    if omitted_accounts:
        _LOGGER.info("omitted_accounts=%d", omitted_accounts)
    return profiles
