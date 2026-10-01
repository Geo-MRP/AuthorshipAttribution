"""Independent, isolated per-engine process orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import logging
import multiprocessing as mp
import os
from pathlib import Path
import queue
import time
from typing import Literal, cast

from .contracts import (
    AccountProfile,
    BaselineDistribution,
    ComparisonAssets,
    ComparisonReport,
    EngineResult,
    NormalizedStreamChunk,
    RawStreamChunk,
    RunConfig,
    SemanticConfig,
    StylometryArtifact,
    StylometryConfig,
)
from .errors import ContractValidationError, ErrorCode, error_code_for
from .calibration import check_baseline_compat
from .fileformats import read_json_artifact
from .fingerprints import semantic_fingerprint, sha256_hex
from .semantic import score_semantic_local
from .streams import load_normalized_stream, load_raw_stream
from .stylometry import score_stylometry

LOGGER = logging.getLogger(__name__)
_WORKER_START_QUEUE: object | None = None


def _init_worker(start_queue: object) -> None:
    global _WORKER_START_QUEUE
    _WORKER_START_QUEUE = start_queue


def _worker_score(
    engine: Literal["stylometry", "semantic"],
    token: str,
    candidate: AccountProfile,
    suspect: AccountProfile,
    candidate_stream: Sequence[RawStreamChunk] | Sequence[NormalizedStreamChunk],
    suspect_stream: Sequence[RawStreamChunk] | Sequence[NormalizedStreamChunk],
    assets: ComparisonAssets,
    allow_external: bool,
) -> EngineResult:
    """Rebuild all resources in the spawned child; each worker gets one view."""
    start_queue = _WORKER_START_QUEUE
    if start_queue is not None:
        start_queue.put((token, time.monotonic()))  # type: ignore[attr-defined]
    try:
        baseline_data = read_json_artifact(assets.baseline_paths[engine])
        baseline = BaselineDistribution.from_dict(baseline_data)
        if engine == "stylometry":
            artifact = StylometryArtifact.from_dict(read_json_artifact(assets.stylometry_artifact_path))
            payload = read_json_artifact(Path(artifact.payload_path))
            config = StylometryConfig(
                feature_spec_version=artifact.feature_spec_version,
                hash_space_size=artifact.hash_space_size,
                ngram_min=int(payload.get("ngram_min", 2)),
                ngram_max=int(payload.get("ngram_max", 4)),
                minimum_messages=baseline.minimum_messages,
                minimum_chars=baseline.minimum_chars,
            )
            words = payload.get("function_words_sha256")
            # Config compatibility is ultimately checked by the engine against the artifact.
            del words
            return score_stylometry(
                candidate, suspect,
                cast(Sequence[RawStreamChunk], candidate_stream),
                cast(Sequence[RawStreamChunk], suspect_stream),
                artifact=artifact, baseline=baseline, config=config,
            )
        config = SemanticConfig(
            model_dir=assets.semantic_model_dir,
            cache_dir=assets.cache_dir,
            center_embeddings=False,
            device=os.environ.get("AUTH_ATTR_DEVICE", "auto"),  # type: ignore[arg-type]
            allow_external=allow_external,
            cache_retention_days=int(os.environ.get("AUTH_ATTR_CACHE_RETENTION_DAYS", "7")),
            minimum_messages=baseline.minimum_messages,
            minimum_chars=baseline.minimum_chars,
        )
        return score_semantic_local(
            candidate, suspect,
            cast(Sequence[NormalizedStreamChunk], candidate_stream),
            cast(Sequence[NormalizedStreamChunk], suspect_stream),
            baseline=baseline, config=config,
        )
    except BaseException as exc:
        code = error_code_for(exc)
        counts_messages = {"candidate": candidate.n_messages, "suspect": suspect.n_messages}
        counts_chars = {"candidate": candidate.n_chars, "suspect": suspect.n_chars}
        return EngineResult(
            engine=engine, raw_cosine=None, calibrated_percentile=None,
            status="ERROR", n_messages=counts_messages, n_chars=counts_chars,
            error=code.value,
        )


def _failure(
    engine: Literal["stylometry", "semantic"], candidate: AccountProfile, suspect: AccountProfile,
    status: Literal["INSUFFICIENT_DATA", "TIMEOUT", "ERROR"], code: ErrorCode,
) -> EngineResult:
    return EngineResult(
        engine=engine, raw_cosine=None, calibrated_percentile=None, status=status,
        n_messages={"candidate": candidate.n_messages, "suspect": suspect.n_messages},
        n_chars={"candidate": candidate.n_chars, "suspect": suspect.n_chars}, error=code.value,
    )


def _stop_executor(executor: ProcessPoolExecutor, *, terminate: bool) -> None:
    processes = tuple(getattr(executor, "_processes", {}).values())
    if terminate:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            if process.is_alive():
                process.join(timeout=0.2)
    executor.shutdown(wait=False, cancel_futures=True)


def _run_engine(
    engine: Literal["stylometry", "semantic"],
    candidate: AccountProfile,
    suspect: AccountProfile,
    assets: ComparisonAssets,
    config: RunConfig,
    context: object,
) -> tuple[EngineResult, ProcessPoolExecutor | None, object | None]:
    manager = cast(object, context).Manager()
    starts = manager.Queue()
    token = f"{engine}-{time.monotonic_ns()}"
    executor = ProcessPoolExecutor(
        max_workers=1, mp_context=context, initializer=_init_worker, initargs=(starts,)
    )
    submitted_at = time.monotonic()
    try:
        baseline = BaselineDistribution.from_dict(read_json_artifact(assets.baseline_paths[engine]))
        requested_messages = int(os.environ.get("AUTH_ATTR_MINIMUM_MESSAGES", str(baseline.minimum_messages)))
        requested_chars = int(os.environ.get("AUTH_ATTR_MINIMUM_CHARS", str(baseline.minimum_chars)))
        executor.shutdown(wait=False, cancel_futures=True)
        executor = ProcessPoolExecutor(
            max_workers=1, mp_context=context, initializer=_init_worker, initargs=(starts,)
        )
        if baseline.minimum_messages != requested_messages or baseline.minimum_chars != requested_chars:
            _stop_executor(executor, terminate=False)
            manager.shutdown()
            return _failure(engine, candidate, suspect, "ERROR", ErrorCode.BASELINE_CONFIG_MISMATCH), None, None
        if engine == "stylometry":
            artifact = StylometryArtifact.from_dict(read_json_artifact(assets.stylometry_artifact_path))
            expected_fingerprint = artifact.configuration_fingerprint
        else:
            semantic_config = SemanticConfig(
                model_dir=assets.semantic_model_dir, cache_dir=assets.cache_dir,
                center_embeddings=False,
                device=os.environ.get("AUTH_ATTR_DEVICE", "auto"),  # type: ignore[arg-type]
                allow_external=config.allow_external,
                cache_retention_days=int(os.environ.get("AUTH_ATTR_CACHE_RETENTION_DAYS", "7")),
                minimum_messages=baseline.minimum_messages,
                minimum_chars=baseline.minimum_chars,
            )
            manifest_path = assets.semantic_model_dir / "manifest.json"
            manifest_hash = sha256_hex(manifest_path.read_bytes())
            manifest_data = read_json_artifact(manifest_path)
            tokenizer_hash = manifest_data.get("tokenizer_vocab_sha256")
            if not isinstance(tokenizer_hash, str):
                raise ContractValidationError("manifest", ErrorCode.LOCAL_MODEL_INVALID)
            expected_fingerprint = semantic_fingerprint(
                semantic_config, manifest_hash, tokenizer_hash,
                candidate.preprocessing_fingerprint,
            )
        incompatibility = check_baseline_compat(baseline, engine, expected_fingerprint)
        if incompatibility is not None:
            _stop_executor(executor, terminate=False)
            manager.shutdown()
            return _failure(engine, candidate, suspect, "ERROR", incompatibility), None, None
        if baseline.minimum_messages > min(candidate.n_messages, suspect.n_messages) or baseline.minimum_chars > min(candidate.n_chars, suspect.n_chars):
            _stop_executor(executor, terminate=False)
            manager.shutdown()
            return _failure(engine, candidate, suspect, "INSUFFICIENT_DATA", ErrorCode.MINIMUMS_NOT_MET), None, None
        if engine == "stylometry":
            left, right = load_raw_stream(candidate), load_raw_stream(suspect)
            timeout_s = config.stylometry_timeout_s
        else:
            left, right = load_normalized_stream(candidate), load_normalized_stream(suspect)
            timeout_s = config.semantic_timeout_s
        # ProcessPoolExecutor's one-worker queues begin this job immediately. The
        # worker's start signal anchors the requested execution-time deadline.
        future = executor.submit(
            _worker_score, engine, token, candidate, suspect, left, right, assets, config.allow_external
        )
        started: float | None = None
        while started is None:
            try:
                seen_token, started_at = starts.get(timeout=0.01)
                if seen_token == token:
                    started = started_at
            except queue.Empty:
                if future.done():
                    started = time.monotonic()
                    break
                # If startup stalls before task code signals, bound waiting from
                # actual executor submission rather than hang indefinitely.
                startup_limit = max(config.stylometry_timeout_s, config.semantic_timeout_s)
                if time.monotonic() - submitted_at >= startup_limit:
                    _stop_executor(executor, terminate=True)
                    manager.shutdown()
                    return _failure(engine, candidate, suspect, "TIMEOUT", ErrorCode.ENGINE_TIMEOUT), None, None
        remaining = max(0.0, timeout_s - (time.monotonic() - started))
        try:
            return future.result(timeout=remaining), executor, manager
        except TimeoutError:
            _stop_executor(executor, terminate=True)
            manager.shutdown()
            return _failure(engine, candidate, suspect, "TIMEOUT", ErrorCode.ENGINE_TIMEOUT), None, None
    except BaseException as exc:
        _stop_executor(executor, terminate=True)
        manager.shutdown()
        code = error_code_for(exc)
        return _failure(engine, candidate, suspect, "ERROR", code), None, None


def compare_shortlist(
    flagged: AccountProfile,
    suspects: Sequence[AccountProfile],
    *,
    assets: ComparisonAssets,
    config: RunConfig,
) -> Sequence[ComparisonReport]:
    """Compare a flagged profile against each suspect with independent engines."""
    if not isinstance(flagged, AccountProfile) or not isinstance(assets, ComparisonAssets):
        raise ContractValidationError("input", ErrorCode.INPUT_INVALID)
    reports: list[ComparisonReport] = []
    context = mp.get_context("spawn")
    for suspect in sorted(suspects, key=lambda item: item.account_id):
        if flagged.account_id == suspect.account_id:
            continue
        minimums_met = all(
            profile.status == "READY"
            and profile.n_messages >= max(profile.minimum_messages, _minimum_for_engine(assets, "messages"))
            and profile.n_chars >= max(profile.minimum_chars, _minimum_for_engine(assets, "chars"))
            for profile in (flagged, suspect)
        )
        if not minimums_met:
            stylo = _failure("stylometry", flagged, suspect, "INSUFFICIENT_DATA", ErrorCode.MINIMUMS_NOT_MET)
            semantic = _failure("semantic", flagged, suspect, "INSUFFICIENT_DATA", ErrorCode.MINIMUMS_NOT_MET)
        else:
            # Dedicated single-worker pools per pair guarantee each engine has an
            # independent per-pair deadline and failure domain.
            import threading
            outcomes: dict[str, EngineResult] = {}
            def execute(engine: Literal["stylometry", "semantic"]) -> None:
                result, executor, manager = _run_engine(engine, flagged, suspect, assets, config, context)
                outcomes[engine] = result
                if executor is not None:
                    _stop_executor(cast(ProcessPoolExecutor, executor), terminate=False)
                if manager is not None:
                    cast(object, manager).shutdown()
            threads = [threading.Thread(target=execute, args=(engine,)) for engine in ("stylometry", "semantic")]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            stylo, semantic = outcomes["stylometry"], outcomes["semantic"]
        reports.append(ComparisonReport(
            schema_version="1.0", flagged_account_id=flagged.account_id,
            suspect_account_id=suspect.account_id, stylometry=stylo, semantic=semantic,
            banner=ComparisonReport.BANNER,
        ))
    return reports


def _minimum_for_engine(assets: ComparisonAssets, kind: Literal["messages", "chars"]) -> int:
    """Read the baseline's applied minimum without exposing artifact contents."""
    env_name = "AUTH_ATTR_MINIMUM_MESSAGES" if kind == "messages" else "AUTH_ATTR_MINIMUM_CHARS"
    configured = os.environ.get(env_name)
    if configured is not None:
        try:
            return int(configured)
        except ValueError:
            return 200 if kind == "messages" else 5000
    try:
        data = read_json_artifact(assets.baseline_paths["stylometry"])
        return int(data[f"minimum_{kind}"])
    except (KeyError, TypeError, ValueError, ContractValidationError):
        return 200 if kind == "messages" else 5000


__all__ = ["compare_shortlist"]
