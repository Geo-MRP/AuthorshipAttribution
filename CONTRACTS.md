# Authorship Attribution Contracts

This document is the Phase 0 source of truth for shared Python objects,
serialized artifacts, error codes, fingerprints, calibration, and module
ownership. The package code uses only the Python standard library. All later
agents import these contracts and must not duplicate their validation,
fingerprinting, calibration, or file-reader behavior.

## Global Rules

- Runtime support is Python 3.11 or newer.
- Concrete contract records are frozen, slotted dataclasses with `to_dict()`
  and `from_dict()`.
- JSON is UTF-8, deterministic, uses sorted keys and compact separators
  `(',', ':')`, and uses `ensure_ascii=False`.
- JSON arrays deserialize to lists. The only tuple contract,
  `EvaluationConfig.split_fractions`, serializes as a JSON array.
- IDs are nonempty opaque strings. Callers must not infer structure from IDs.
- Timestamps are RFC 3339 UTC strings ending in `Z`, for example
  `2026-10-01T12:34:56Z`.
- A SHA-256 field is a 64-character lower-case hexadecimal digest.
- `from_dict()` rejects missing fields, unknown fields, wrong types,
  non-finite floats, invalid enum values, and contract-specific range or
  cross-field violations with `ContractValidationError`.
- The message-text fields are redacted from dataclass reprs: `Message.text`,
  `Chunk.raw_text`, `Chunk.normalized_text`, `RawStreamChunk.raw_text`, and
  `NormalizedStreamChunk.normalized_text`. Contract validation errors contain
  a field name and stable code only, never a supplied value.

## Errors

`ErrorCode` is a `StrEnum` with exactly these values:

| Value | Meaning |
| --- | --- |
| `MINIMUMS_NOT_MET` | Account or pair does not meet required data minimums. |
| `ENGINE_TIMEOUT` | A per-pair engine deadline elapsed. |
| `ENGINE_FAILURE` | An otherwise uncategorized engine failure. |
| `BASELINE_CONFIG_MISMATCH` | Baseline engine or configuration fingerprint differs. |
| `BASELINE_INVALID` | Baseline content violates its contract. |
| `LOCAL_MODEL_UNAVAILABLE` | Required local model files are unavailable. |
| `LOCAL_MODEL_INVALID` | Required local model files are invalid. |
| `EXTERNAL_NOT_ALLOWED` | A requested external service is not permitted. |
| `INPUT_INVALID` | Input or contract content is invalid. |
| `ARTIFACT_INVALID` | A persisted artifact cannot be read or written safely. |

`ContractValidationError(field_name, code)` subclasses `ValueError` and
exposes `field_name: str` and `code: ErrorCode`. Its string is exactly
`<field_name>:<ERROR_CODE>`; it never includes an offending value.

`error_code_for(exc)` returns an `ErrorCode` carried by the exception when
present, otherwise `ENGINE_FAILURE`. It does not inspect exception message
text.

## Core Records

### `Message`

| Field | Type and rule |
| --- | --- |
| `schema_version` | Exactly `"1.0"`. |
| `message_id` | Nonempty source message ID. Unique within its `game_id` across all sources. |
| `account_id` | Nonempty opaque account ID. |
| `game_id` | Nonempty opaque game ID. |
| `channel_id` | Nonempty opaque channel ID or `None`. |
| `timestamp_utc` | RFC 3339 UTC timestamp. |
| `text` | String, redacted from repr. |
| `is_system` | Boolean. |
| `source_id` | Nonempty opaque source ID. |
| `source_row` | Integer at least 1. |

Message identity is `message_key(game_id, message_id)`, not `message_id`
alone. `message_key` is exactly:

```python
json.dumps([game_id, message_id], separators=(",", ":"), ensure_ascii=False)
```

This remains unambiguous when IDs contain separators such as `/` or `:`.

### `Chunk`

| Field | Type and rule |
| --- | --- |
| `schema_version` | Exactly `"1.0"`. |
| `chunk_id`, `account_id`, `session_id` | Nonempty opaque strings. |
| `message_ids` | List of nonempty `message_key` values. |
| `start_timestamp_utc`, `end_timestamp_utc` | RFC 3339 UTC timestamps. |
| `n_messages` | Nonnegative integer equal to both `len(message_ids)` and `len(raw_text)`. |
| `n_chars` | Nonnegative integer equal to the total Unicode code-point count of `raw_text`. |
| `raw_text` | List of strings, redacted from repr. |
| `normalized_text` | String, redacted from repr. |
| `normalized_sha256` | SHA-256 of normalized text under the preprocessing contract. |

`Chunk.to_raw_stream_chunk()` creates the raw-only stream representation.
`Chunk.to_normalized_stream_chunk()` creates the normalized-only stream
representation.

### Stream Chunk Records

`RawStreamChunk` has exactly:

```text
schema_version, chunk_id, account_id, session_id, message_ids,
start_timestamp_utc, end_timestamp_utc, n_messages, n_chars, raw_text
```

It applies the raw `Chunk` count invariants and redacts `raw_text` from repr.

`NormalizedStreamChunk` has exactly:

```text
schema_version, chunk_id, account_id, session_id, message_ids,
start_timestamp_utc, end_timestamp_utc, n_messages, n_chars,
normalized_text, normalized_sha256
```

It requires `n_messages == len(message_ids)`, validates the timestamp and
digest fields, and redacts `normalized_text` from repr.

### `AccountProfile`

| Field | Type and rule |
| --- | --- |
| `schema_version` | Exactly `"1.0"`. |
| `account_id` | Nonempty opaque ID. |
| `game_ids` | Nonempty sorted, unique list of game IDs. |
| `primary_game_id` | One of `game_ids`; chosen upstream by highest message count, then lexicographic tie break. |
| `primary_channel_id` | Channel ID or `None`; chosen upstream by the same count/tie rule. |
| `active_months_utc` | Sorted, unique `YYYY-MM` list derived from post-filter messages. |
| `n_messages`, `n_chars`, `session_count` | Nonnegative integers. |
| `activity_histogram_utc` | Exactly 24 nonnegative integers. |
| `utc_hour_filter` | `None` or sorted, unique integers from 0 through 23. |
| `chunk_ids` | List of nonempty chunk IDs. |
| `raw_stream_path`, `normalized_stream_path` | Nonempty stream path strings. |
| `status` | `READY` or `INSUFFICIENT_DATA`. |
| `minimum_messages`, `minimum_chars` | Applied minimum thresholds. |
| `preprocessing_fingerprint` | SHA-256 preprocessing fingerprint. |

### Pair and Engine Records

`PairCounts` is a `TypedDict` with exactly `candidate: int` and
`suspect: int`. Candidate always means the flagged account.

`EngineResult` fields are:

| Field | Type and rule |
| --- | --- |
| `engine` | `stylometry` or `semantic`. |
| `raw_cosine` | Float in `[-1, 1]` or `None`. |
| `calibrated_percentile` | Float in `[0, 100]` or `None`. |
| `status` | `OK`, `INSUFFICIENT_DATA`, `TIMEOUT`, or `ERROR`. |
| `n_messages`, `n_chars` | `PairCounts` with nonnegative integer counts. |
| `error` | `None` or an `ErrorCode` value string. |

For `OK`, both numeric values are present and `error` is `None`. For all
other statuses, both numeric values are `None` and `error` is a nonempty
`ErrorCode` value.

### `BaselineDistribution`

| Field | Type and rule |
| --- | --- |
| `schema_version` | Exactly `"1.0"`. |
| `baseline_id`, `source_snapshot_id` | Nonempty opaque strings. |
| `engine` | `stylometry` or `semantic`. |
| `created_at_utc` | RFC 3339 UTC timestamp. |
| `configuration_fingerprint` | SHA-256 engine configuration fingerprint. |
| `random_seed` | Integer. |
| `calibration_population` | Exactly `known_different_account_pairs`. |
| `n_pairs` | Integer at least 1 and equal to `len(sorted_raw_cosines)`. |
| `sorted_raw_cosines` | Nondecreasing floats in `[-1, 1]`. |
| `minimum_messages`, `minimum_chars` | Applied baseline minimums. |
| `strata_counts` | Mapping of stratum names to nonnegative counts only; no IDs. |
| `centering_vector_path` | `None` or a relative mean-embedding path; may only be populated for semantic baselines. |

### Reports

`ComparisonReport` has exactly:

```text
schema_version, flagged_account_id, suspect_account_id, stylometry, semantic,
banner
```

The schema version is `"1.0"`; account IDs are nonempty; `stylometry` must
be an `EngineResult` for the stylometry engine; `semantic` must be one for the
semantic engine; and `banner` is exactly:

```text
Decision support only. Requires human review.
```

There is deliberately no combined field, verdict field, ranking field, or
other merged decision in this contract.

`LabeledPair` fields are `schema_version`, `account_id_a`, `account_id_b`,
`label`, and `source_snapshot_id`. Its schema version is `"1.0"`; accounts
must differ and are stored lexicographically as `account_id_a < account_id_b`;
`label` is `same` or `different`; and source snapshot ID is nonempty.

### Model and Asset Records

`StylometryArtifact` is metadata only. It has exactly:

```text
artifact_id, artifact_sha256, feature_spec_version, hash_space_size,
function_words_sha256, function_words_lang, structural_means, structural_stds,
payload_path, seed, configuration_fingerprint
```

IDs and path are nonempty; digest fields are SHA-256; hash-space size is at
least 1; structural arrays contain finite floats and have equal lengths.
Payload ownership remains with the stylometry engine.

`EmbeddingModel` is a protocol supplied by the semantic engine:

```python
model_fingerprint: str
embedding_dim: int
def embed(self, texts: Sequence[str]) -> numpy.typing.NDArray[numpy.float32]: ...
```

NumPy is imported only under `TYPE_CHECKING` by the contracts package.

`ComparisonAssets` contains paths and specifications only, never live model
or engine instances. Its fields are:

```text
stylometry_artifact_path, semantic_model_dir,
baseline_paths["stylometry" | "semantic"], game_terms_path, usernames_path,
function_words_path, cache_dir
```

All values are `Path` values in memory and path strings in JSON. Both baseline
engine keys are required. Workers rebuild assets after process spawn.

## Configuration Records

All configuration records are frozen, slotted dataclasses and validate ranges
in `__post_init__`.

| Record | Fields and defaults |
| --- | --- |
| `IngestConfig` | `input_format="auto"` (`json`, `csv`, or `auto`); `field_map={}`; `system_flag_field=None`; `system_flag_true_values=["true", "1"]`; `dedupe=True`. Field-map keys and values are nonempty strings. |
| `PreprocessConfig` | `minimum_messages=200`; `minimum_chars=5000`; `idle_gap_minutes=30`; `utc_hour_filter=None`; `chunk_target_mode="characters"` (`messages` or `characters`); `chunk_target_value=1000`; `cache_retention_days=7`. Minimums, idle gap, and target are positive; retention is nonnegative. |
| `StylometryConfig` | `feature_spec_version="1.0"`; `hash_space_size=262144`; `ngram_min=2`; `ngram_max=4`; `block_weights={"char_ngrams": 0.5, "structural": 0.25, "function_words": 0.25}`; `minimum_messages=200`; `minimum_chars=5000`. N-gram max is at least min; the three exact finite nonnegative weights sum to 1.0. |
| `SemanticConfig` | Required `model_dir` and `cache_dir` paths; `device="auto"` (`auto`, `cpu`, `cuda`); `batch_size_cpu=16`; `batch_size_cuda=64`; `max_tokens=256`; `center_embeddings=True`; `allow_external=False`; `external_endpoint=None`; `cache_retention_days=7`; `minimum_messages=200`; `minimum_chars=5000`. Batch sizes, tokens, and minimums are positive; retention is nonnegative. |
| `BaselineConfig` | `min_pairs=1000`; `max_pairs=50000`; `seed=1729`; `strata=["game", "channel", "month", "volume_bucket"]`; `volume_bucket_edges=[200, 500, 1000, 5000]`; `refresh_max_age_days=31`; `refresh_new_pair_fraction=0.10`. Pair limits are positive and ordered; edges strictly increase; age is positive; fraction is in `[0, 1]`. |
| `RunConfig` | `stylometry_timeout_s=30.0`; `semantic_timeout_s=90.0`; `terminate_timed_out_workers=True`; `output_format="table"` (`table` or `json`); `allow_external=False`. Timeouts are positive floats and apply per pair from job execution start, never batch submission. |
| `EvaluationConfig` | `seed=1729`; `split_fractions=(0.6, 0.2, 0.2)`; `min_precision=0.80`; `n_bootstrap=1000`; `ci_level=0.95`. Fractions are finite nonnegative floats summing to 1.0; precision is in `[0, 1]`; bootstrap count is positive; CI level is strictly between 0 and 1. |

### Evaluation Records

`EngineEvaluation` fields are:

```text
engine, roc_auc, pr_auc, test_base_rate, threshold_percentile,
precision_constraint_met, precision, recall, support_same, support_different,
excluded_insufficient_data, ci
```

Engine is `stylometry` or `semantic`; AUC values are `None` or floats in
`[0,1]`; base rate, precision, and recall are floats in `[0,1]`; threshold is
a percentile in `[0,100]`; support and exclusion values are nonnegative
integers; and `ci` has exactly `roc_auc`, `precision`, and `recall`, each a
nondecreasing two-float interval in `[0,1]`.

`EvaluationReport` has exactly:

```text
schema_version, seed, partition_sizes, stylometry, semantic, note
```

Its schema version is `"1.0"`; `partition_sizes` has exactly nonnegative
`train`, `validation`, and `test` integers; engine records match their named
engines; and `note` is exactly:

```text
Thresholds are review-prioritization suggestions only.
```

It has no merged metric or merged threshold.

## Fingerprints

`canonical_json(obj) -> bytes` is the only canonical JSON encoder. It sorts
object keys, emits compact UTF-8 JSON with `ensure_ascii=False`, and rejects
NaN and infinity.

`sha256_hex(data) -> str` produces a lower-case SHA-256 hex digest.

`fingerprint(components) -> str` hashes `canonical_json(components)`, so
mapping insertion order never changes a fingerprint.

`NORMALIZATION_SPEC_VERSION` is exactly `"1.0"`.

| Function | Included components | Explicitly excluded components |
| --- | --- | --- |
| `preprocessing_fingerprint(config, game_terms_sha256, usernames_sha256)` | Normalization spec version, idle gap, UTC filter, chunk target mode/value, minimum messages/chars, game-term digest, username digest. | Cache retention days. |
| `stylometry_fingerprint(config, function_words_sha256, function_words_lang, artifact_sha256, preprocessing_fp)` | Feature spec version, hash-space size, n-gram min/max, block weights, minimums, function-word digest/language, artifact digest, preprocessing fingerprint. | Nothing from the listed behavior inputs. |
| `semantic_fingerprint(config, model_manifest_sha256, tokenizer_sha256, preprocessing_fp)` | Max tokens, centering flag, minimums, model-manifest digest, tokenizer digest, preprocessing fingerprint. | Device, CPU/CUDA batch sizes, cache directory/retention, external permission, external endpoint, and model directory. |

## Calibration

`calibrate_cosine(raw_cosine, baseline)` is the only percentile calibration
implementation. It validates the baseline and returns:

```python
100.0 * bisect_right(baseline.sorted_raw_cosines, raw_cosine) / baseline.n_pairs
```

The right bisect means tied values are counted inclusively. A finite
`raw_cosine` at most `1e-9` beyond `[-1, 1]` is clamped to the nearest bound;
larger excursions or non-finite values raise `ContractValidationError` for
`raw_cosine`.

`check_baseline_compat(baseline, engine, expected_fingerprint)` returns:

- `BASELINE_CONFIG_MISMATCH` when engine or fingerprint differs.
- `BASELINE_INVALID` when a matching baseline violates validation.
- `None` when the baseline is valid and compatible.

## File Formats

### Term Lists

`load_term_list(path, require_lang=...)` reads UTF-8 plain text. There is one
entry per line. Blank lines and comment lines beginning with `#` are ignored.
One optional directive may be present:

```text
# lang: <BCP-47 tag>
```

Entries are stripped, NFC-normalized, deduplicated, and sorted. The returned
`TermList` fields are `entries: list[str]`, `sha256: str`, and `lang: str |
None`. The digest covers the LF-joined normalized sorted entries, prefixed by
the language directive and one LF when a language exists. `require_lang=True`
requires the directive and is used for function-word lists. Invalid UTF-8 and
invalid content produce `INPUT_INVALID` without echoing an entry.

### Labeled Pair NDJSON

`load_labeled_pairs(path)` reads exactly one JSON object per nonempty line,
each conforming to `LabeledPair`. Self-pairs are invalid. Repeated full pair
records are deduplicated; a canonical account pair appearing with conflicting
labels is invalid. Output is deterministically sorted by account IDs, label,
and source snapshot ID.

### JSON Artifacts

`write_json_artifact(obj, path)` canonicalizes JSON through `canonical_json`,
place. `read_json_artifact(path)` accepts a UTF-8 JSON object only and rejects
non-finite JSON constants. Artifact I/O failures use `ARTIFACT_INVALID`.

## Ownership

| Owner | Modules | Required shared interfaces | Constraint |
| --- | --- | --- | --- |
| Agent 1 (Data) | `ingest.py`, `preprocess.py`, `streams.py` | `Message`, `Chunk`, `AccountProfile`, stream chunk types, `IngestConfig`, `PreprocessConfig`, `load_term_list`, `preprocessing_fingerprint` | Must preserve message identity and stream invariants. |
| Agent 2 (Stylometry) | `stylometry.py` | `StylometryConfig`, `StylometryArtifact`, `RawStreamChunk`, `EngineResult`, `stylometry_fingerprint`, `calibrate_cosine`, `check_baseline_compat` | Owns artifact payload, not metadata contract. |
| Agent 3 (Semantic) | `semantic.py` | `SemanticConfig`, `EmbeddingModel`, `NormalizedStreamChunk`, `EngineResult`, `semantic_fingerprint`, `calibrate_cosine`, `check_baseline_compat` | Rebuilds local assets in spawned workers. |
| Agent 4 (Orchestration) | `baselines.py`, `orchestrator.py`, `evaluate.py`, `reporting.py`, `cli.py` | Everything above | Must not reimplement calibration, fingerprints, or file readers. |

No agent may edit Phase 0 modules. Contract change requests go through the
architect.
