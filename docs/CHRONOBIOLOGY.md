# Chronobiology engine

The chronobiology engine is a deterministic, content-free temporal feature
extractor and scorer. It works only with caller-supplied
`TemporalStreamChunk` records; it does not open a profile's temporal, raw, or
normalized stream paths. The implementation uses Python 3.11+ and NumPy 2.1.3.
Its 122 features summarize UTC timing, message character counts, and session
structure. They are review-support measurements, not authorship probabilities,
verdicts, or automated enforcement decisions.

The public API in `authorship_attribution.chronobiology` consists of:

- `extract_chronobiology_features(stream)` — return an unstandardized
  `float64` vector of length 122.
- `fit_chronobiology_artifact(baseline_streams, *, config,
  preprocessing_fp, output_dir)` — fit population feature statistics and write
  their content-addressed artifact.
- `compute_chronobiology_raw(candidate_stream, suspect_stream, *, artifact,
  config)` — compute the normalized-vector cosine.
- `score_chronobiology(candidate, suspect, candidate_stream, suspect_stream,
  *, artifact, baseline, config)` — validate the pair, score it, and calibrate
  it against a supplied known-different baseline.

## Time and session interpretation

Each event timestamp is parsed as
`datetime.fromisoformat(f"{timestamp[:-1]}+00:00").timestamp()` and represented
as a floating-point epoch in seconds. All phase calculations use UTC. There is
no inferred local timezone, timezone database, geolocation, or daylight-saving
conversion. The Monday-UTC weekly phase origin is zero: the day number is
`int(floor((t + 259200.0) / 86400.0)) % 7`, with Monday 0 through Sunday 6.

Chunks are traversed in input order. Adjacent chunks with the same
`session_id` form one session, so splitting a session across chunks does not
break its transitions. A session ID may not reappear after another session has
started; chunk IDs must be unique, all chunks must belong to one account, and
timestamps must be globally nondecreasing. These checks ensure that session
boundaries and event order are unambiguous.

An actual within-session interval is the difference between consecutive event
epochs, clamped to `[0.5, 1800.0]` seconds. Burst-process interval summaries
fall back to clamped globally consecutive intervals only when there are no
within-session intervals at all. If there is fewer than two total events, that
fallback is `[1800.0]`. This fallback does not create session transitions:
memory, velocity, composition correlation, and session features use actual
within-session transitions only. Fano features use event times within each
session, not the interval fallback.

## The 122 features

The vector contains three contiguous blocks, in this order:

| Indices | Block | Size | Default block weight |
| --- | --- | ---: | ---: |
| 0–78 | `circadian_fourier` | 79 | 0.45 |
| 79–107 | `burst_process` | 29 | 0.30 |
| 108–121 | `session_kinematics` | 14 | 0.25 |

All event-level means, population variances, and population standard deviations
use `math.fsum` for accumulation. Quantiles use NumPy's linear interpolation
(`method="linear"`).

### Circadian Fourier block: indices 0–78

- **0–14 — daily harmonics.** For `k = (1, 2, 3, 4, 6)`, in that order, emit
  the mean cosine, mean sine, and resultant magnitude for angles
  `2πkt / 86400`. Each mean is over all events; magnitude is `hypot(mean_cos,
  mean_sin)`.
- **15–23 — weekly harmonics.** For `m = (1, 2, 3)`, emit the same triplet
  for `2πm(t + 259200) / 604800`. The offset puts Monday UTC at weekly phase
  zero.
- **24–71 — smoothed half-hour activity.** First count events in 48 UTC
  half-hour bins using `int((t % 86400) // 1800)` and divide each count by the
  total event count to obtain `p_r`. For output bin `b`, use the finite
  circular-distance kernel
  `K(b,r) = exp(-0.5 * d²)`, where
  `d = min(abs(b-r), 48-abs(b-r))`. Compute
  `u_b = fsum(p_r * K(b,r) for r in 0..47)` and normalize by `fsum(u)`.
  This is a finite wrapped kernel, not an infinite periodized Gaussian.
- **72 — activity entropy.** `-sum(smooth_b * log(smooth_b))`, omitting any
  zero terms.
- **73 — circular activity variation.** `1 - hypot(sum(smooth_b*cos(2π(b+0.5)/48)),
  sum(smooth_b*sin(2π(b+0.5)/48)))`, clamped to `[0, 1]` for roundoff.
- **74–76 — activity trough.** For each start bin `b`, sum the 12 consecutive
  smoothed bins beginning at `b`, wrapping around midnight. Choose the minimum
  by `(round(mass, 12), b)`, so ties after rounding select the lowest start
  bin. At the selected 12-bin window's midpoint, angle
  `2π((b+6) % 48)/48`, emit cosine, sine, and the original unrounded mass.
- **77 — weekend share.** Number events on Saturday and Sunday (day numbers 5
  and 6) divided by all events.
- **78 — weekday/weekend phase alignment.** Compute each subset's daily first
  harmonic and emit the normalized dot product of their cosine/sine vectors,
  clamped to `[-1, 1]`. Emit `0.0` if either subset is empty or either
  resultant magnitude is at most `1e-9`.

### Burst-process block: indices 79–107

Let `delta` be the clamped interval sequence described above and
`tau = log1p(delta)`.

- **79–94 — interval histogram.** Sixteen normalized bins from
  `numpy.histogram(tau, bins=16, range=(0, log(1801)))`.
- **95–101 — log-interval summaries.** Emit, in order, the mean and population
  standard deviation of `tau`, standardized skewness, excess kurtosis, and
  linear q10, q50, and q90. If the population standard deviation is below
  `1e-12`, emit zero for the standard deviation, skewness, and kurtosis; the
  mean and quantiles are still computed normally.
- **102 — interval dispersion.** For raw clamped intervals, emit
  `(std_delta - mean_delta) / (std_delta + mean_delta)`.
- **103 — interval memory.** Compute Pearson correlation for consecutive
  pairs of actual within-session intervals, never across sessions and never
  using the global fallback. Emit zero with fewer than three pairs or if
  either side's population standard deviation is below `1e-12`; otherwise
  clamp the correlation to `[-1, 1]`.
- **104–107 — sparse-window Fano summaries.** For widths 60, 300, 900, and
  1800 seconds, compute one `log1p(Fano)` per width. For each session,
  `W_s = max(1, ceil((last-first)/width))`; each event is assigned to
  `min(W_s-1, int((t-first)//width))`. Only occupied-bin counts are stored in
  a `Counter`, so a very long session does not allocate memory for every empty
  window. Across sessions, let `W = sum(W_s)`,
  `Q = sum(occupied_count**2)`, `mu = N/W`,
  `variance = max(0, Q/W - mu**2)`, and
  `Fano = variance / max(1e-9, mu)`.

### Session-kinematics block: indices 108–121

- **108–111 — character velocity.** For each actual within-session
  transition, use the current event's character count `L` and clamped interval
  `delta`. Keep transitions with `delta <= 120` seconds and compute
  `L / max(1, delta)`. Emit its mean, population standard deviation, linear
  median, and linear q90. If none qualify, emit four zeros.
- **112 — composition/interval correlation.** Pearson correlation between
  `log1p(current L)` and `log1p(delta)` over all actual within-session
  transitions. Emit zero with fewer than two observations or if either
  population standard deviation is below `1e-12`.
- **113–117 — session summaries.** Emit `log1p(mean session message count)`,
  population standard deviation of session message count divided by its mean,
  `log1p(mean session duration)`, population standard deviation of
  `log1p(session duration)`, and the fraction of sessions containing one
  message. Duration is the last event epoch minus the first.
- **118–121 — session endpoints.** Average, separately across sessions, the
  start cosine, start sine, end cosine, and end sine for daily angle
  `2πt/86400`.

## Fitting, normalization, and scoring

Artifact fitting skips empty streams, rejects duplicate account IDs, filters
accounts using the configured message and character minimums, sorts eligible
accounts by ID, and requires at least two eligible accounts. It computes a
per-feature mean and population standard deviation over eligible accounts.
Individual account identities and temporal streams are not written into the
payload; their deterministic corpus identity is represented by a digest.

For scoring, each feature is standardized against the fitted statistics; a
feature whose fitted standard deviation is zero contributes zero. Each of the
three standardized blocks is normalized independently, multiplied by its
configured nonnegative block weight, concatenated, and normalized once more.
A zero-norm block contributes zeros. For active blocks, the cosine contribution
is proportional to the **square** of the block weight because the final vector
is normalized after weighting. If the final vector norm is zero or nonfinite,
scoring fails with `ENGINE_FAILURE`. The resulting cosine is clamped to
`[-1, 1]`.

The calibration baseline contains sorted raw cosines from known-different
account pairs. The shared inclusive empirical percentile is
`100 * bisect_right(sorted_raw_cosines, raw_cosine) / n_pairs`; ties count as
baseline values less than or equal to the score. It describes the score's
position in that reference distribution. It is **not** an authorship
probability, confidence, or likelihood.

## Artifact identity and compatibility

The canonical JSON payload contains exactly these seven keys:

1. `feature_spec_version`
2. `block_weights`
3. `feature_means`
4. `feature_stds`
5. `preprocessing_fingerprint`
6. `baseline_corpus_identity`
7. `baseline_account_count`

Its SHA-256 digest is computed over canonical UTF-8 JSON. The artifact ID is
`chronobiology-` followed by that digest, and the file is named
`<artifact_id>.json`. A pre-existing file at that content-addressed path is
accepted only if its bytes are identical; a different file is not overwritten.
Loading validates the exact key set, digest, artifact ID, statistic metadata,
types and sizes, and the configured behavior. The configuration fingerprint
covers the feature version, block weights, minimums, artifact digest, and
preprocessing fingerprint, so an artifact or baseline with incompatible
settings cannot be silently reused.

SHA-256 digests are integrity and identity mechanisms, not anonymization.
Feature statistics, timestamps, character counts, account IDs, stream IDs,
artifacts, and derived scores remain sensitive metadata and should be protected
accordingly.

## Validation and error behavior

Errors contain stable field names and error codes, not supplied timestamp,
count, account, or stream values. The principal categories are:

- `INPUT_INVALID` — malformed input records, invalid ordering or session
  structure, mixed account ownership, profile/stream disagreement, invalid
  config or argument types.
- `MINIMUMS_NOT_MET` — an empty feature stream, fewer than two eligible
  fitting accounts, or a scoring profile that is not ready or does not meet
  configured message/character minimums. Scoring returns
  `INSUFFICIENT_DATA` for the last case.
- `ENGINE_FAILURE` — unrepresentable or nonfinite feature calculations, or a
  zero/nonfinite final normalized vector.
- `ARTIFACT_INVALID` — unreadable, malformed, tampered, or internally
  inconsistent artifact data, including a conflicting content-addressed file.
- `BASELINE_CONFIG_MISMATCH` — baseline engine/fingerprint/minimums,
  preprocessing fingerprint, or artifact feature behavior does not match the
  supplied configuration.
- `BASELINE_INVALID` — a baseline fails its own shared contract validation.

For valid profiles, scoring maps these failures to a sanitized `ERROR`
`EngineResult` with null scores. Invalid or mutated-invalid profile contracts
are rejected before a result can safely capture their pair counts. The shared
result contract also has a `TIMEOUT`/`ENGINE_TIMEOUT` status, but this
synchronous temporal engine does not create timeout results. Local-model and
external-service error codes belong to other engines, not this implementation.

## Privacy, isolation, and human review

This engine sees only timestamps, character counts, account/session/chunk
identifiers, and derived temporal features in `TemporalStreamChunk` records.
It never reads message text or opens raw, normalized, or temporal stream paths;
it loads no model and makes no network request. It does not print or log
caller-supplied values, and exception messages are sanitized. The semantic
engine's pinned ONNX Runtime remains an offline exception for that separate
engine. Its opt-in HTTP adapter and the manual network-capable model setup
helper remain unchanged; neither is used by chronobiology.

Use scores only to support human review. They do not establish authorship and
must not be treated as an automatic enforcement decision.
