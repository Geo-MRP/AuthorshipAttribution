# Classical stylometry engine

The stylometry module consumes only `RawStreamChunk` records. It never opens
either profile stream path; `AccountProfile` is used only for identity and
minimum/count checks. All computation is deterministic and local.

## Feature definitions

Each raw message is an independent document and is processed as-is. Character
2-, 3-, and 4-grams retain case, punctuation, whitespace, emoji, and repeated
characters. No separator is inserted between messages. Every n-gram is mapped
to one of 262,144 buckets by keyed BLAKE2b: the 32-byte key is SHA-256 of
`stylometry-ngram-v1:<seed>`, the digest is 8 bytes with personalization
`styl-ngram-v1`, and the unsigned big-endian digest integer modulo 262,144 is
the bucket. The fitted artifact stores bucket IDF values, not n-grams.

For each account, character-term frequency is bucket occurrence count divided
by all extracted character n-gram occurrences. IDF is
`ln((1 + number_of_baseline_message_documents) / (1 + document_frequency)) + 1`;
an unseen bucket uses `ln(1 + number_of_baseline_message_documents) + 1`.
Document frequency counts each bucket at most once per original raw message.
The resulting sparse character block is L2-normalized.

Structural values are calculated across the account's raw messages:

* Double-space rate: overlapping `"  "` occurrences / `max(1, raw code points)`.
* Punctuation vector, in order `! " # $ % & ' ( ) * + , - . / : ; < = > ? @ [ \\ ] ^ _ \u0060 { | } ~`: each code point's count / the same denominator.
* Repeated-punctuation rate: non-overlapping matches of `\.{3,}`, U+2026, or
  two or more identical ASCII punctuation code points / the same denominator.
* Capitalization ratio: uppercase alphabetic code points / `max(1, alphabetic
  code points)`.
* Elongation rate: non-overlapping same-code-point alphabetic runs of length at
  least three / the raw code-point denominator.
* Emoji rate: code points in U+1F300–U+1FAFF, U+2600–U+27BF, or U+2300–U+23FF /
  the raw code-point denominator. Variation selectors and joiners are not bases.
* Average message length: raw code points / `max(1, message count)`.
* Function-word frequencies: exact tokens formed by contiguous Unicode
  alphabetic code points, casefolded only for lookup, divided by
  `max(1, total alphabetic-token count)`. There is no stemming or lemmatizing.

The fixed ASCII punctuation vector is then followed by ellipsis/repeated
punctuation, capitalization, elongation, emoji, average message length, and the
sorted casefolded function-word list. Structural means and population standard
deviations are fitted across eligible baseline accounts. A zero standard
deviation maps that standardized dimension to zero. The structural and
function-word blocks are each L2-normalized. Each normalized block is scaled by
its configured `block_weights` value, then the sparse concatenation is
L2-normalized. Raw cosine is the dot product, is bounded by
the cosine domain, and is not a probability. The
valid result labels are **Raw cosine** and **Known-different baseline
percentile**.

## Artifact and compatibility

The metadata record is the shared Phase 0 `StylometryArtifact`. Its sibling JSON
payload contains only sparse bucket-index/IDF pairs, the casefolded supplied
function-word vocabulary, feature metadata, corpus digest, seed, and structural
statistics. It stores no message text or recoverable n-gram strings. Baseline
corpus identity hashes account IDs, counts, and per-account raw-stream digests.
The configuration fingerprint includes the Phase 0 stylometry fingerprint
(feature version, hash space, n-gram range, block weights, minimums, function
word-list hash/language, payload hash, and preprocessing fingerprint). The
preprocessing fingerprint includes the feature definition version, minimums,
and baseline corpus identity. A baseline must be a valid `stylometry`
`BaselineDistribution` with exactly the artifact configuration fingerprint and
the configured minimum counts. Only then is Phase 0 `calibrate_cosine` called.

## Status and error codes

* `INSUFFICIENT_DATA` / `MINIMUMS_NOT_MET`: profile status or configured count
  thresholds fail; both score fields are null.
* `ERROR` / `BASELINE_CONFIG_MISMATCH`: baseline engine, fingerprint, or
  minimums do not match; both score fields are null.
* `ERROR` / `BASELINE_INVALID`: a matching baseline fails shared validation.
* `ERROR` / `ARTIFACT_INVALID`: payload read, digest, or payload metadata fails.
* `ERROR` / `INPUT_INVALID`: raw stream/profile counts or ownership are
  inconsistent.
* `ERROR` / `ENGINE_FAILURE`: an unexpected safe-to-classify engine exception.

No message text appears in logs or exception messages. Successful results carry
both the raw cosine and inclusive known-different baseline percentile. The
percentile is descriptive calibration only and does not provide a verdict.
