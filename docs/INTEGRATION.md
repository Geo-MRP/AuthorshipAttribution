# Integration and moderation-use guide

Stylometry, semantic, and chronobiology each report raw cosine, an inclusive
known-different baseline percentile, status, and counts. A combined
review-prioritization heuristic uses successful engine percentiles and
renormalized weights: stylometry 0.40, semantic 0.35, chronobiology 0.25.
It blends the weighted percentile and an endpoint-preserving geometric-tail
score. Neither the combined value nor a base percentile is a probability,
confidence level, automatic verdict, or separately calibrated combined
percentile. The tail is not a Fisher chi-square p-value. Every comparison
retains the banner: Decision support only. Requires human review.

## Local CLI

Set INPUT, OUT, GAME_TERMS, USERNAMES, FUNCTION_WORDS, PAIRS, MODEL, FLAGGED,
and SUSPECT to valid local input paths or opaque account IDs. FUNCTION_WORDS
must have its language directive. MODEL must already contain the pinned local
model and manifest; no model download occurs at runtime. The default fit prefix
is bg_ and must select at least two eligible background accounts. PAIRS must
contain at least 1,000 eligible distinct reviewed different-account pairs for
baseline construction. Use the same minimums and UTC-hour filter at ingest and
comparison. Do not use zero-retention ingest for a later CLI process: its
streams are removed when the ingest process exits.

```bash
ARTIFACT_DIR="$OUT/artifacts"
BASELINE_DIR="$OUT/baselines"
CACHE_DIR="$OUT/cache"
venv/bin/python -m authorship_attribution ingest "$INPUT" --output "$OUT" --game-terms "$GAME_TERMS" --usernames "$USERNAMES"
venv/bin/python fit_artifact.py --profiles "$OUT" --function-words "$FUNCTION_WORDS" --output "$ARTIFACT_DIR"
venv/bin/python -m authorship_attribution build-baseline --pairs "$PAIRS" --profiles "$OUT" --stylometry-artifact "$ARTIFACT_DIR/artifact.json" --chronobiology-artifact "$ARTIFACT_DIR/chronobiology-artifact.json" --model-dir "$MODEL" --output "$BASELINE_DIR" --function-words "$FUNCTION_WORDS" --cache-dir "$CACHE_DIR"
```

The baseline builder writes immutable content-addressed files. Set
STYLOMETRY_BASELINE, SEMANTIC_BASELINE, and CHRONOBIOLOGY_BASELINE to the
exact files produced by this run.

```bash
venv/bin/python -m authorship_attribution compare --profiles "$OUT" --flagged "$FLAGGED" --suspect "$SUSPECT" --stylometry-artifact "$ARTIFACT_DIR/artifact.json" --chronobiology-artifact "$ARTIFACT_DIR/chronobiology-artifact.json" --model-dir "$MODEL" --stylometry-baseline "$STYLOMETRY_BASELINE" --semantic-baseline "$SEMANTIC_BASELINE" --chronobiology-baseline "$CHRONOBIOLOGY_BASELINE" --game-terms "$GAME_TERMS" --usernames "$USERNAMES" --function-words "$FUNCTION_WORDS" --cache-dir "$CACHE_DIR" --device cpu --format json
venv/bin/python -m authorship_attribution evaluate --pairs "$PAIRS" --profiles "$OUT" --output "$OUT/evaluation.json" --stylometry-artifact "$ARTIFACT_DIR/artifact.json" --chronobiology-artifact "$ARTIFACT_DIR/chronobiology-artifact.json" --model-dir "$MODEL" --stylometry-baseline "$STYLOMETRY_BASELINE" --semantic-baseline "$SEMANTIC_BASELINE" --chronobiology-baseline "$CHRONOBIOLOGY_BASELINE" --game-terms "$GAME_TERMS" --usernames "$USERNAMES" --function-words "$FUNCTION_WORDS" --cache-dir "$CACHE_DIR" --format json
```

The compare timeout flags are --stylometry-timeout, --semantic-timeout, and
--chronobiology-timeout; their defaults are 30, 90, and 30 seconds. Per-pair
deadlines start at worker execution, not batch submission. A separate bounded
startup wait prevents a stalled process start. Timed-out workers are
terminated. Workers use local process-management IPC and do not send data to a
remote service. One failed engine does not suppress successful engines. If all
engines are insufficient, combined is INSUFFICIENT_DATA/MINIMUMS_NOT_MET;
otherwise any timeout makes combined TIMEOUT/ENGINE_TIMEOUT; otherwise it is
ERROR with the first engine error. In each all-failed case combined numeric
fields are null.

## Baselines and evaluation

Production baselines use reviewed different-label pairs only, deterministic
stratified sampling, and at least 1,000 eligible distinct pairs. Outputs are
immutable and carry engine, fingerprint, counts, snapshot digest, seed, and
non-identifying strata; no chat text. Refresh monthly, at configured corpus
growth, or after relevant preprocessing, model, feature, or minimum changes.
The Phase 2 message-order fingerprint changed, so reingest and rebuild affected
artifacts and baselines instead of relabeling old files.

Evaluation groups accounts connected by same-label edges before a seeded
account-disjoint train/validation/test split. Cross-partition pairs are omitted.
Stylometry and chronobiology artifacts and every known-different calibration
distribution use training accounts and pairs only. A missing training reference
for any base engine fails with training_baselines:MINIMUMS_NOT_MET; it is never
replaced by percentile zero. Validation selects each engine's threshold by
highest recall, then precision, then lowest threshold, subject to the configured
minimum precision if possible; that threshold is frozen for test metrics.
Base-engine ROC/PR uses raw cosine; combined ROC/PR uses the combined score.
Bootstrap confidence intervals resample complete test pairs. The retained
excluded_insufficient_data field counts all unavailable held-out engine/pair
results, including errors, not only insufficient profiles. The combined metric
uses the combined score directly, not a fourth calibration baseline.

## Privacy and external capability

Only raw stream records reach stylometry; only normalized records reach semantic;
only temporal records reach chronobiology. Temporal records contain timestamps
and character counts but no message content; they and derived artifacts remain
sensitive. Apply the configured retention policy and access controls. Reports
contain opaque flagged and suspect account IDs but not message text, individual
timestamps, per-message lengths, or arbitrary exception messages.

The new chronobiology and fusion modules make no network calls. The existing
semantic engine may use the approved pinned ONNX Runtime offline. Its external
HTTP adapter remains explicitly opt-in and network-capable, and the standalone
setup_minilm_onnx.py helper remains a manual network-capable setup tool. This
upgrade does not remove either legacy capability. No IP, geolocation, named
timezone, or geographic inference is used.
