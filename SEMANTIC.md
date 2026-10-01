# Semantic engine

The semantic engine consumes only `NormalizedStreamChunk` values. Its public
numeric outputs are **Raw cosine** and **Known-different baseline percentile**;
neither is a probability. They are decision-support measurements and require
human review.

## Local model and manifest

Place the ONNX export for `sentence-transformers/all-MiniLM-L6-v2` under
`models/all-MiniLM-L6-v2-onnx/`. Runtime model acquisition is disabled. The
directory must contain a UTF-8 `manifest.json` object with:

```json
{
  "model_id": "sentence-transformers/all-MiniLM-L6-v2",
  "embedding_dim": 384,
  "maximum_token_length": 256,
  "license": "Apache-2.0",
  "onnx_model": "model.onnx",
  "model_files": {"model.onnx": "<64 lowercase hex sha256>"},
  "tokenizer_vocab": "vocab.txt",
  "tokenizer_vocab_sha256": "<64 lowercase hex sha256>",
  "do_lower_case": true
}
```

Every model file named in `model_files` is checked before loading. The model
fingerprint is SHA-256 over the exact manifest bytes; tokenizer vocabulary hash
is tracked separately for baseline compatibility. Missing files return
`LOCAL_MODEL_UNAVAILABLE`; malformed manifests, vocabulary, and hashes return
`LOCAL_MODEL_INVALID`. No model files are bundled in this repository.

The tokenizer implements BERT basic tokenization and greedy WordPiece from the
verified vocabulary. At most 256 WordPiece IDs including `[CLS]`/`[SEP]` enter
each inference window. Longer normalized chunks are split at full basic-token
boundaries while retaining the exact original source slices, including
whitespace. No normalized content is modified or discarded. ONNX output is
attention-mask mean pooled.

## Device and batching

`SemanticConfig.device` accepts `auto`, `cpu`, or `cuda`. `auto` selects CUDA
only if `CUDAExecutionProvider` is available; otherwise it uses
`CPUExecutionProvider`. Explicit CUDA without an available provider fails with
`LOCAL_MODEL_UNAVAILABLE`. Defaults are 16 inputs per CPU batch and 64 per CUDA
batch. Batch size and device do not affect the device-independent embedding
fingerprint.

## Cache and vectors

Cache entries are JSON files under `<cache_dir>/<key-prefix>/<key>.json`. The
key includes SHA-256 of exact UTF-8 window text, SHA-256 of its token-ID list,
and the model-manifest fingerprint. A cache value contains only that key,
token-ID hash, model fingerprint, numeric vector, creation timestamp, and
expiry timestamp—never normalized text. Entries expire according to
`cache_retention_days`; zero retention removes cache files and directories at
the end of each score operation.

Each account vector is the arithmetic mean of its chunk/window embeddings,
optionally centered by the baseline's relative `centering_vector_path`, then
L2-normalized. The centering artifact is canonical JSON with `embedding_dim`
and a `mean_embedding` numeric array. The final account-vector cosine is
computed deterministically from normalized float32 vectors.

## Baseline compatibility

`semantic_fingerprint` from Phase 0 is authoritative. It covers model manifest
hash, vocabulary hash, maximum tokens, centering setting, minimum counts, and
the upstream preprocessing fingerprint (which identifies normalization and
game-term configuration). The profiles must agree on preprocessing
fingerprints, and baseline minimums must match the engine. Mismatch returns
`BASELINE_CONFIG_MISMATCH` without either public score. Percentile conversion
uses the shared `calibrate_cosine` implementation only.

## External providers

Local inference is the default. The optional HTTP adapter requires both an
explicit `allow_external=True` configuration and a CLI caller that has passed
`--allow-external` into that setting; a configured endpoint alone never
enables requests. It uses standard-library HTTP, reads an optional
`AUTHORSHIP_ATTRIBUTION_API_KEY` environment credential, sends only normalized
masked text, and does not log request or response bodies. The adapter expects
an object with an `embeddings` array and an explicitly configured endpoint.
There is no network path for local model acquisition.

## Diagnostics and privacy

Stable public failure codes come from Phase 0: `MINIMUMS_NOT_MET`,
`BASELINE_CONFIG_MISMATCH`, `BASELINE_INVALID`, `LOCAL_MODEL_UNAVAILABLE`,
`LOCAL_MODEL_INVALID`, `EXTERNAL_NOT_ALLOWED`, `INPUT_INVALID`,
`ARTIFACT_INVALID`, `ENGINE_TIMEOUT`, and `ENGINE_FAILURE`. Logs use status,
stable codes, opaque baseline identifiers, model fingerprints, and counts only.
Normalized text is never logged, included in exception text, or written to
cache metadata.
