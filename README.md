# AuthorshipAttribution

Local data contracts, ingestion/preprocessing, and semantic scoring foundations
for a moderator decision-support system. Scoring outputs are review-support
measurements and require human review.

## Documentation

Full documentation lives in [`docs/`](docs/):

- [CONTRACTS.md](docs/CONTRACTS.md) — shared records, artifacts, errors, and fingerprints.
- [INTEGRATION.md](docs/INTEGRATION.md) — CLI workflows, baselines, evaluation, and privacy.
- [STYLOMETRY.md](docs/STYLOMETRY.md) — stylometry engine and artifact format.
- [SEMANTIC.md](docs/SEMANTIC.md) — semantic engine, model manifest, and caching.
- [CHRONOBIOLOGY.md](docs/CHRONOBIOLOGY.md) — temporal features, fitting, and scoring.

## Data ingestion

`ingest_chat_logs(paths, config=...)` accepts UTF-8 JSON (a list of records, a
single record, a `{ "messages": [...] }` object, or newline-delimited JSON) and
UTF-8 CSV with a header. Files are decoded strictly and CSV is read with
newline-aware parsing. The canonical field names are `message_id`, `account_id`,
`game_id`, `timestamp_utc`, and `text`; `channel_id` is optional. Configure
`IngestConfig.field_map` as canonical-name to source-column-name mappings. Set
`system_flag_field` to the source system indicator column and optionally adjust
`system_flag_true_values` (default `true` and `1`). Fields not mapped are read
from their canonical names. Message IDs are unique within a game; duplicate
records within one source are rejected, while identical repeats from another
source can be deduplicated when `dedupe=True`.

Invalid inputs raise `ContractValidationError` with stable field/code strings;
chat content is never copied into those messages. Source message text remains
the decoded UTF-8 payload, unchanged.

## Preprocessing and defaults

`PreprocessConfig` defaults are minimum 200 messages and 5,000 raw characters,
a 30-minute idle gap, no UTC-hour filter, character-based chunking with target
1,000, and seven days of stream retention. To configure message-count chunks,
set `chunk_target_mode="messages"` and `chunk_target_value` (for example, 50).
Chunking never splits messages or crosses sessions. A gap exactly equal to the
idle-gap limit starts a new session.

Input messages are sorted by UTC timestamp, message ID, then game ID, making
same-ID messages from different games deterministic. This ordering change
updates the preprocessing fingerprint; regenerate references built under the
old preprocessing behavior. Empty and system messages are excluded;
whitespace-only messages remain. The UTC activity histogram is computed from
accepted messages before the optional hour filter. Filtering and minimum-count
checks use post-filter messages, as do account counts, sessions, active months,
and chunks. Normalization uses Unicode NFC, username and @handle masking, URL
masking, numeric masking, game-term masking, whitespace collapse, and trim, in
that order. The original text is kept only in the raw stream.

## Artifacts and retention

For output directory `OUT`, deterministic account-keyed artifacts are written
under:

* `OUT/profiles/<sha256-account-id>.json` — one validated `AccountProfile`.
* `OUT/raw/<sha256-account-id>.ndjson` — raw-only `RawStreamChunk` records.
* `OUT/normalized/<sha256-account-id>.ndjson` — normalized-only
  `NormalizedStreamChunk` records.
* `OUT/temporal/<sha256-account-id>.ndjson` — content-free
  `TemporalStreamChunk` records with exactly `schema_version`, `chunk_id`,
  `account_id`, `session_id`, `timestamps_utc`, `char_counts`, `n_messages`,
  and `n_chars`. There is one temporal record per raw/normalized chunk;
  timestamp and character-count arrays align with that chunk's messages.
  Character counts are Python Unicode code-point lengths. Chunks from the same
  session repeat the same `session_id`, so a chunk boundary is not a session
  boundary. Timestamps and lengths are sensitive behavioral metadata and need
  the same retention protections as the other streams.

Stream profile paths are absolute local paths. Artifacts use UTF-8; profile
JSON is a single canonical object and each NDJSON line is a single canonical
object. `load_raw_stream` and `load_normalized_stream` validate projections
and ensure account/chunk ordering matches the profile. `load_temporal_stream`
validates account and chunk identity, profile message and character totals,
session continuity/count, and chronological ordering across records without
opening either text-bearing stream. Each stream file is atomically replaced
individually; writing the raw, normalized, and temporal files is not a
multi-file transaction. Runs remove expired raw, normalized, and temporal
stream files based on configured retention. A retention of zero schedules all
three generated streams for deletion when the Python process exits; the profile
metadata remains available. A zero-retention CLI ingest therefore deletes its
generated streams at process exit, so later CLI processes cannot use them.

Accounts with no accepted non-system, non-empty messages are omitted and the
omitted account count is logged without message content. Accounts with accepted
messages but none remaining after UTC-hour filtering retain their pre-filter
game/channel metadata and receive an `INSUFFICIENT_DATA` profile with empty
post-filter counts, month/chunk lists, and three zero-line stream files.

Preprocessing and temporal readers make no network calls and do not access IP,
geolocation, or timezone databases. Existing semantic inference may use the
approved pinned ONNX Runtime offline. The external semantic adapter remains
explicitly opt-in and network-capable, and `setup_minilm_onnx.py` remains a
manual network-capable helper.

## Semantic engine

The normalized-stream-only semantic engine, local ONNX model manifest format,
device selection, cache behavior, external-provider gate, and stable failure
codes are documented in [SEMANTIC.md](docs/SEMANTIC.md). NumPy is the only required
runtime dependency; ONNX Runtime CPU or GPU may be installed through the
corresponding optional dependency extra. Models are never downloaded at
runtime.

## Integration CLI

The `authorship-attribution` commands provide `ingest`, `compare`,
`build-baseline`, and `evaluate` workflows. See [INTEGRATION.md](docs/INTEGRATION.md)
for baseline refresh cadence, evaluation threshold selection, timeout and
partial-failure behavior, privacy, and limits on use. Comparisons are
moderator decision support only and require human review.
