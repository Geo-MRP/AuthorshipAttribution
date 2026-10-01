# Integration and moderation-use guide

The integration layer treats stylometry and semantic results as separate
measurements. Each result carries its own raw cosine, **Known-different baseline
percentile**, status, and message counts. Percentiles describe an empirical
known-different reference distribution; they are not probabilities, confidence,
or percent same person. Results are decision support and require human review.
They must not trigger account actions or moderation enforcement.

## CLI

```text
authorship-attribution ingest INPUT... --output DIR --game-terms FILE --usernames FILE
authorship-attribution compare --profiles DIR --flagged ID --suspect ID ... [assets/options]
authorship-attribution build-baseline --pairs PAIRS.ndjson --profiles DIR ...
authorship-attribution evaluate --pairs PAIRS.ndjson --profiles DIR --output report.json ...
```

`ingest` accepts the Phase 0 JSON/CSV options, minimums, UTC-hour filters, and
cache-retention setting. `compare` defaults to 200 messages / 5,000 characters,
30 seconds for stylometry and 90 seconds for semantic scoring, and local model
execution. External mode is opt-in via `--allow-external`; when enabled, the
CLI logs only that the mode was enabled. Chat content is never written to logs
or reports.

## Baselines and refresh

Build baselines exclusively from reviewed `different` labeled pairs whose two
profiles meet the configured minimums. At least 1,000 eligible distinct pairs
are required; up to 50,000 are sampled deterministically (default seed 1729),
stratified to favor pairs sharing game, channel, and active UTC month. Baseline
JSON stores only engine scores, opaque baseline/source digests, configuration
metadata, and non-identifying stratum counts. Files are immutable.

Refresh at least monthly in UTC, when eligible reviewed-pair volume grows by at
least 10%, and whenever model, term-list, preprocessing, feature, function-word,
or minimum-data fingerprints change. `baseline_needs_refresh` exposes the
age/growth/configuration checks to deployment schedulers. Baseline source
snapshot identifiers must identify reviewed inputs without embedding account
or pair IDs.

## Evaluation and threshold selection

Evaluation groups accounts connected by `same` labels before a seeded
60/20/20 train/validation/test split. Pairs crossing partitions are discarded.
Engine artifacts and known-different reference distributions use training
accounts only. Each engine selects a separate validation percentile threshold,
preferring highest F1 subject to 0.80 minimum precision, otherwise highest F1
and a marked unmet precision constraint. That threshold is frozen before the
test metrics are calculated. Test ROC-AUC is computed from raw cosine; test
precision, recall, support, and exclusions are reported separately per engine.
Thresholds are review-prioritization suggestions only.

## Failure and privacy behavior

Each pair starts independent spawned, one-worker process pools for stylometry
and semantic scoring. Per-engine deadlines begin when that worker starts. A
timeout terminates that worker and returns `TIMEOUT / ENGINE_TIMEOUT`; an engine
exception returns a sanitized stable error code. The other result remains
available. If either profile misses configured minimums, neither engine runs
and both return `INSUFFICIENT_DATA / MINIMUMS_NOT_MET`. Cache retention follows
the selected days; zero retention removes generated stream/vector-cache data
when the command completes.

Only raw stream records are supplied to the stylometry worker and only
normalized stream records to semantic. The CLI and persisted outputs contain no
chat text. No IP, geolocation, named timezone, or geographic inference is used.
