# Authorship Attribution Contracts

This document is the source of truth for shared Python objects, serialized
artifacts, error codes, fingerprints, calibration, and module ownership. Runtime
support is Python 3.11 or newer; new shared code uses the standard library and
the pinned `numpy==2.1.3` dependency. The semantic engine's pinned ONNX Runtime
is an approved offline exception. The unchanged legacy external semantic
adapter and manual model-download/setup helper remain explicit opt-in paths and
may use network access. Shared contracts and the new fusion code do not add
network access. All engine owners import these contracts and must not duplicate
their validation, fingerprinting, calibration, or file-reader behavior.

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
- Temporal metadata contains no text, but is still sensitive because it can
  reveal account activity patterns and must receive privacy-protective handling.
- `from_dict()` rejects missing fields, unknown fields, wrong types,
  non-finite floats, invalid enum values, and contract-specific range or
  cross-field violations with `ContractValidationError`.
- The message-text fields are redacted from dataclass reprs: `Message.text`,
  `Chunk.raw_text`, `Chunk.normalized_text`, `RawStreamChunk.raw_text`, and
  `NormalizedStreamChunk.normalized_text`. Contract validation errors contain
  a field name and stable code only, never a supplied value.
- Raw or normalized chat text must not be copied into logs, errors, reports,
  temporal records, or artifact metadata. Content-free temporal metadata remains
  sensitive and must be handled accordingly.

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

`TemporalStreamChunk` has exactly:

```text
schema_version, chunk_id, account_id, session_id, timestamps_utc, char_counts,
n_messages, n_chars
```

The schema version is exactly `"1.0"`; its IDs are nonempty; `timestamps_utc`
is a nonempty list of RFC 3339 UTC `Z` timestamps in nondecreasing parsed epoch
order; and `char_counts` is a list of nonnegative integers. Both lists have
exactly `n_messages` elements, where `n_messages >= 1`, and the sum of
`char_counts` equals nonnegative `n_chars`. Equal instants and fractional
seconds are accepted according to `datetime.fromisoformat()` and
`datetime.timestamp()` behavior, including pre-epoch values. An empty filtered
profile is represented by its zero-line temporal stream, not by an empty
`TemporalStreamChunk`. This projection has no raw or normalized text, message
IDs, or text digest. Its activity metadata is content-free but sensitive.

### `AccountProfile`

| Field | Type and rule |
| --- | --- |
| `schema_version` | Exactly `"1.0"`. |
| `account_id` | Nonempty opaque ID. |
| `game_ids` | Nonempty sorted, unique list of game IDs from accepted messages before the UTC-hour filter. |
| `primary_game_id` | One of `game_ids`, from accepted messages before the UTC-hour filter; chosen upstream by highest message count, then lexicographic tie break. |
| `primary_channel_id` | Channel ID or `None`, from accepted messages before the UTC-hour filter; chosen upstream by the same count/tie rule. |
| `active_months_utc` | Sorted, unique `YYYY-MM` list from post-filter messages; may be empty. |
| `n_messages`, `n_chars`, `session_count` | Nonnegative integers from post-filter messages; each may be 0. |
| `activity_histogram_utc` | Exactly 24 nonnegative integers. |
| `utc_hour_filter` | `None` or sorted, unique integers from 0 through 23. |
| `chunk_ids` | List of nonempty chunk IDs from post-filter messages; may be empty. |
| `raw_stream_path`, `normalized_stream_path` | Nonempty stream path strings. Zero-line raw and normalized stream files are valid, and an empty post-filter profile still points to them. |
| `status` | `READY` exactly when both profile minimums are met; otherwise `INSUFFICIENT_DATA`. |
| `minimum_messages`, `minimum_chars` | Applied minimum thresholds. |
| `preprocessing_fingerprint` | SHA-256 preprocessing fingerprint. |
| `temporal_stream_path` | Nonempty path to the content-free temporal stream. Defaults to `unused-temporal-path` for direct-constructor compatibility. |

`game_ids`, `primary_game_id`, and `primary_channel_id` describe accepted
messages before the UTC-hour filter. `n_messages`, `n_chars`,
`session_count`, `chunk_ids`, and `active_months_utc` describe post-filter
messages. A profile can therefore retain nonempty game identity while having
no post-filter messages.

`n_messages == 0` if and only if `n_chars == 0`, `session_count == 0`,
`chunk_ids == []`, and `active_months_utc == []`. When `n_messages` is
positive, all four corresponding post-filter values are nonempty/nonzero.
`status` is `READY` if and only if both `n_messages >= minimum_messages` and
`n_chars >= minimum_chars`; otherwise it is `INSUFFICIENT_DATA`.

For backward compatibility, `AccountProfile.from_dict()` migrates a legacy
valid dictionary only when `temporal_stream_path` is absent, supplying
`unused-temporal-path`. An explicitly empty, null, or incorrectly typed path is
invalid, as are all other missing or unknown fields. Serialization always
includes the temporal path. The default does not change any empty-profile or
readiness invariants.

An account with no accepted messages at all gets no `AccountProfile`:
`build_account_profiles` omits it, and callers may report only the count of
omitted accounts.

### Pair and Engine Records

`PairCounts` is a `TypedDict` with exactly `candidate: int` and
`suspect: int`. Candidate always means the flagged account.

`EngineResult` fields are:

| Field | Type and rule |
| --- | --- |
| `engine` | `stylometry`, `semantic`, or `chronobiology`. |
| `raw_cosine` | Float in `[-1, 1]` or `None`. |
| `calibrated_percentile` | Float in `[0, 100]` or `None`. |
| `status` | `OK`, `INSUFFICIENT_DATA`, `TIMEOUT`, or `ERROR`. |
| `n_messages`, `n_chars` | `PairCounts` with nonnegative integer counts. |
| `error` | `None` or an `ErrorCode` value string. |

For `OK`, both numeric values are present and `error` is `None`. For all
other statuses, both numeric values are `None` and `error` is a nonempty
`ErrorCode` value.

### `CombinedResult`

`CombinedResult` has exactly:

```text
status, combined_score, weighted_percentile, fisher_tail_score,
active_engines, weights_used, error
```

`status` is `OK`, `INSUFFICIENT_DATA`, `TIMEOUT`, or `ERROR`. On `OK`, the
three finite numeric values are floats in `[0, 100]`, `active_engines` is a
nonempty ordered subset of `stylometry`, `semantic`, `chronobiology`, and
`weights_used` has exactly those keys with positive finite float weights in
`(0, 1]` summing to 1 within absolute tolerance `1e-12`. `combined_score` must
equal the 50/50 blend of the two component scores within that same tolerance,
and `error` is `None`. On failure, numeric fields are `None`, the engine and
weight collections are empty, and `error` is a valid `ErrorCode` value.
`INSUFFICIENT_DATA` requires `MINIMUMS_NOT_MET`; `TIMEOUT` requires
`ENGINE_TIMEOUT`. Combined scores are review-prioritization heuristics, not
probabilities, verdicts, confidence values, or separately calibrated fused
percentiles.

### `BaselineDistribution`

| Field | Type and rule |
| --- | --- |
| `schema_version` | Exactly `"1.0"`. |
| `baseline_id`, `source_snapshot_id` | Nonempty opaque strings. |
| `engine` | `stylometry`, `semantic`, or `chronobiology`. |
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
chronobiology, combined, banner
```

The schema version is `"1.0"`; account IDs are nonempty; `stylometry` must
be an `EngineResult` for the stylometry engine, `semantic` for the semantic
engine, and `chronobiology` for the chronobiology engine. The `combined` field
is a `CombinedResult`; its active engines must agree with the base results.
When all base engines fail, all-insufficient takes precedence, then any timeout,
then an error using the first non-null base error in fixed engine order. The
mandatory `banner` is exactly:

```text
Decision support only. Requires human review.
```

The combined score is a human-review aid only. There are no automated verdict,
probability, confidence, or rank fields.

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

`ChronobiologyArtifact` is metadata only. It has exactly:

```text
artifact_id, artifact_sha256, feature_spec_version, feature_means, feature_stds,
payload_path, configuration_fingerprint
```

IDs and path are nonempty; digest and fingerprint fields are lower-case
SHA-256; feature version is nonempty; and the means and standard-deviation
arrays each contain exactly 122 finite floats. Standard deviations are
nonnegative; zero is valid. The engine-owned canonical JSON payload contains
exactly `feature_spec_version`, `block_weights`, `feature_means`,
`feature_stds`, `preprocessing_fingerprint`, `baseline_corpus_identity`, and
`baseline_account_count`. Its digest covers canonical payload bytes; its ID is
`chronobiology-` plus that digest; and its filename is the ID plus `.json`.
Loading and verifying that payload and fitting with at least two eligible
accounts belong to the chronobiology engine, not this metadata contract.

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
baseline_paths["stylometry" | "semantic" | "chronobiology"],
game_terms_path, usernames_path, function_words_path, cache_dir,
chronobiology_artifact_path
```

All values are `Path` values in memory and path strings in JSON. Exactly three
baseline keys—one for each base engine—are required. Workers rebuild assets
after process spawn.

## Configuration Records

All configuration records are frozen, slotted dataclasses and validate ranges
in `__post_init__`.

| Record | Fields and defaults |
| --- | --- |
| `IngestConfig` | `input_format="auto"` (`json`, `csv`, or `auto`); `field_map={}`; `system_flag_field=None`; `system_flag_true_values=["true", "1"]`; `dedupe=True`. Field-map keys and values are nonempty strings. |
| `PreprocessConfig` | `minimum_messages=200`; `minimum_chars=5000`; `idle_gap_minutes=30`; `utc_hour_filter=None`; `chunk_target_mode="characters"` (`messages` or `characters`); `chunk_target_value=1000`; `cache_retention_days=7`. Minimums, idle gap, and target are positive; retention is nonnegative. |
| `StylometryConfig` | `feature_spec_version="1.0"`; `hash_space_size=262144`; `ngram_min=2`; `ngram_max=4`; `block_weights={"char_ngrams": 0.5, "structural": 0.25, "function_words": 0.25}`; `minimum_messages=200`; `minimum_chars=5000`. N-gram max is at least min; the three exact finite nonnegative weights sum to 1.0. |
| `ChronobiologyConfig` | `feature_spec_version="1.0"`; `block_weights={"circadian_fourier": 0.45, "burst_process": 0.30, "session_kinematics": 0.25}`; `minimum_messages=200`; `minimum_chars=5000`. The three exact finite float weights are each in `[0, 1]` and sum to 1.0 within absolute tolerance `1e-12`; minimums are positive. Default instances have independent weight mappings. |
| `SemanticConfig` | Required `model_dir` and `cache_dir` paths; `device="auto"` (`auto`, `cpu`, `cuda`); `batch_size_cpu=16`; `batch_size_cuda=64`; `max_tokens=256`; `center_embeddings=True`; `allow_external=False`; `external_endpoint=None`; `cache_retention_days=7`; `minimum_messages=200`; `minimum_chars=5000`. Batch sizes, tokens, and minimums are positive; retention is nonnegative. |
| `BaselineConfig` | `min_pairs=1000`; `max_pairs=50000`; `seed=1729`; `strata=["game", "channel", "month", "volume_bucket"]`; `volume_bucket_edges=[200, 500, 1000, 5000]`; `refresh_max_age_days=31`; `refresh_new_pair_fraction=0.10`. Pair limits are positive and ordered; edges strictly increase; age is positive; fraction is in `[0, 1]`. |
| `RunConfig` | `stylometry_timeout_s=30.0`; `semantic_timeout_s=90.0`; `terminate_timed_out_workers=True`; `output_format="table"` (`table` or `json`); `allow_external=False`; `chronobiology_timeout_s=30.0` (last field). All timeouts are finite, strictly positive floats and apply per pair from job execution start, never batch submission. |
| `EvaluationConfig` | `seed=1729`; `split_fractions=(0.6, 0.2, 0.2)`; `min_precision=0.80`; `n_bootstrap=1000`; `ci_level=0.95`. Fractions are finite nonnegative floats summing to 1.0; precision is in `[0, 1]`; bootstrap count is positive; CI level is strictly between 0 and 1. |

### Evaluation Records

`EngineEvaluation` fields are:

```text
engine, roc_auc, pr_auc, test_base_rate, threshold_percentile,
precision_constraint_met, precision, recall, support_same, support_different,
excluded_insufficient_data, ci
```

Engine is `stylometry`, `semantic`, `chronobiology`, or `combined`; AUC values
are `None` or floats in
`[0,1]`; base rate, precision, and recall are floats in `[0,1]`; threshold is
a percentile in `[0,100]`; support and exclusion values are nonnegative
integers; and `ci` has exactly `roc_auc`, `precision`, and `recall`, each a
nondecreasing two-float interval in `[0,1]`.

`EvaluationReport` has exactly:

```text
schema_version, seed, partition_sizes, stylometry, semantic, chronobiology,
combined, note
```

Its schema version is `"1.0"`; `partition_sizes` has exactly nonnegative
`train`, `validation`, and `test` integers; each aggregate `EngineEvaluation`
matches its named engine; and `note` is exactly:

```text
Thresholds are review-prioritization suggestions only.
```

The `combined` value is an aggregate `EngineEvaluation`, not a per-pair
`CombinedResult`. Combined evaluation metrics remain review guidance only; there
are no automated verdict, probability, confidence, or rank fields.

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
| `preprocessing_fingerprint(config, game_terms_sha256, usernames_sha256)` | Normalization spec version, `message_order_spec_version="2.0"`, idle gap, UTC filter, chunk target mode/value, minimum messages/chars, game-term digest, username digest. | Cache retention days. |
| `stylometry_fingerprint(config, function_words_sha256, function_words_lang, artifact_sha256, preprocessing_fp)` | Feature spec version, hash-space size, n-gram min/max, block weights, minimums, function-word digest/language, artifact digest, preprocessing fingerprint. | Nothing from the listed behavior inputs. |
| `semantic_fingerprint(config, model_manifest_sha256, tokenizer_sha256, preprocessing_fp)` | Max tokens, centering flag, minimums, model-manifest digest, tokenizer digest, preprocessing fingerprint. | Device, CPU/CUDA batch sizes, cache directory/retention, external permission, external endpoint, and model directory. |
| `chronobiology_fingerprint(config, artifact_sha256, preprocessing_fp)` | Feature spec version, block weights, minimum messages/chars, artifact digest, preprocessing fingerprint. | Nothing from the listed behavior inputs. |

`message_order_spec_version="2.0"` identifies the deterministic `game_id`
tie-breaker used when timestamps and message IDs match across games. It
deliberately changes preprocessing fingerprints. Reingest data and rebuild
affected profiles, cache entries, engine artifacts, and baselines as
appropriate; do not silently relabel old profiles or baselines. Cache
retention remains excluded.

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

`compute_combined_result(stylometry, semantic, chronobiology, weights=None)`
validates inputs in that fixed engine order. Default base weights are
`0.40`, `0.35`, and `0.25`. Custom weights must contain exactly the three base
engine names, each with a strictly positive finite Python `float` no greater
than `1.0`, and sum to `1.0` within absolute tolerance `1e-12` and zero relative
tolerance. Weights are renormalized over only the successful (`OK`) engines.
For each active engine with calibrated percentile `P_e`, let
`u_e=max(0.0005, 1-P_e/100+0.0005)`. The weighted percentile is
`clamp(sum(w_e*P_e), 0, 100)`. With `S=math.fsum(w_e*(-log(u_e)))`,
`fisher_tail_score=100*clamp(1.0005-exp(-S), 0, 1)`, and
`combined_score=clamp(0.5*weighted_percentile+0.5*fisher_tail_score, 0, 100)`.
This tail value is not a Fisher chi-square p-value. Neither it nor the blended
score is a probability, verdict, confidence, or separately calibrated combined
percentile; fusion is a review-prioritization heuristic only.

If no base engine succeeds, result status/error precedence is: all
`INSUFFICIENT_DATA` gives `INSUFFICIENT_DATA` / `MINIMUMS_NOT_MET`; otherwise
any `TIMEOUT` gives `TIMEOUT` / `ENGINE_TIMEOUT`; otherwise status is `ERROR`
and the first non-null base-engine error in fixed order is used. Failed combined
results have null numeric values and empty active-engine and weight collections.

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

| Owner | Write ownership | Shared interfaces governed here |
| --- | --- | --- |
| Subagent 1 | Shared contracts, `calibration.py`, and `fingerprints.py` (including their exports, documentation, and owned tests). | All records, validation, fusion, and fingerprint definitions in this document. |
| Subagent 2 | Preprocessing and stream modules. | `Message`, `Chunk`, `AccountProfile`, stream records, `IngestConfig`, `PreprocessConfig`, and preprocessing fingerprints. |
| Subagent 3 | The chronobiology engine. | `TemporalStreamChunk`, `ChronobiologyConfig`, `ChronobiologyArtifact`, `EngineResult`, `BaselineDistribution`, and chronobiology fingerprints. |
| Subagent 4 | Orchestration, baselines, evaluation, reporting, and CLI. | Comparison/evaluation assets and reports, `RunConfig`, `CombinedResult`, and the fusion interface. |

These four ownership sets have no overlapping writes. Their components may
read one another's interfaces, but the shared contracts in this document govern
those interfaces and each set's owner is responsible for its implementation.
