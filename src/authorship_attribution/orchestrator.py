"""Independent per-engine process orchestration and deterministic fusion."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, TimeoutError
import multiprocessing as mp
import os
from pathlib import Path
import queue
import threading
import time
from typing import Literal, cast

from .calibration import check_baseline_compat, compute_combined_result
from .chronobiology import score_chronobiology
from .contracts import (
    AccountProfile,
    BaselineDistribution,
    ChronobiologyArtifact,
    ChronobiologyConfig,
    ComparisonAssets,
    ComparisonReport,
    EngineResult,
    NormalizedStreamChunk,
    RawStreamChunk,
    RunConfig,
    SemanticConfig,
    StylometryArtifact,
    StylometryConfig,
    TemporalStreamChunk,
)
from .errors import ContractValidationError, ErrorCode, error_code_for
from .fileformats import read_json_artifact
from .fingerprints import semantic_fingerprint, sha256_hex
from .semantic import score_semantic_local
from .streams import load_normalized_stream, load_raw_stream, load_temporal_stream
from .stylometry import score_stylometry


EngineName = Literal["stylometry", "semantic", "chronobiology"]
Stream = Sequence[RawStreamChunk] | Sequence[NormalizedStreamChunk] | Sequence[TemporalStreamChunk]
_ENGINE_ORDER: tuple[EngineName, EngineName, EngineName] = (
    "stylometry", "semantic", "chronobiology"
)
_WORKER_START_QUEUE: object | None = None


def _init_worker(start_queue: object) -> None:
    global _WORKER_START_QUEUE
    _WORKER_START_QUEUE = start_queue


def _failure(
    engine: EngineName,
    candidate: AccountProfile,
    suspect: AccountProfile,
    status: Literal["INSUFFICIENT_DATA", "TIMEOUT", "ERROR"],
    code: ErrorCode,
) -> EngineResult:
    return EngineResult(
        engine=engine,
        raw_cosine=None,
        calibrated_percentile=None,
        status=status,
        n_messages={"candidate": candidate.n_messages, "suspect": suspect.n_messages},
        n_chars={"candidate": candidate.n_chars, "suspect": suspect.n_chars},
        error=code.value,
    )


def _read_baseline(path: Path) -> BaselineDistribution:
    data = read_json_artifact(path)
    try:
        return BaselineDistribution.from_dict(data)
    except ContractValidationError:
        raise ContractValidationError("baseline", ErrorCode.BASELINE_INVALID) from None


def _chrono_config(
    artifact: ChronobiologyArtifact,
    baseline: BaselineDistribution,
) -> ChronobiologyConfig:
    if baseline.minimum_messages < 1 or baseline.minimum_chars < 1:
        raise ContractValidationError("baseline", ErrorCode.BASELINE_CONFIG_MISMATCH)
    payload = read_json_artifact(Path(artifact.payload_path))
    weights = payload.get("block_weights")
    if type(weights) is not dict:
        raise ContractValidationError("artifact", ErrorCode.ARTIFACT_INVALID)
    try:
        return ChronobiologyConfig(
            feature_spec_version=artifact.feature_spec_version,
            block_weights=weights,
            minimum_messages=baseline.minimum_messages,
            minimum_chars=baseline.minimum_chars,
        )
    except ContractValidationError:
        raise ContractValidationError("artifact", ErrorCode.ARTIFACT_INVALID) from None


def _semantic_config(
    assets: ComparisonAssets,
    baseline: BaselineDistribution,
    allow_external: bool,
) -> SemanticConfig:
    if baseline.minimum_messages < 1 or baseline.minimum_chars < 1:
        raise ContractValidationError("baseline", ErrorCode.BASELINE_CONFIG_MISMATCH)
    return SemanticConfig(
        model_dir=assets.semantic_model_dir,
        cache_dir=assets.cache_dir,
        center_embeddings=False,
        device=os.environ.get("AUTH_ATTR_DEVICE", "auto"),
        allow_external=allow_external,
        cache_retention_days=int(os.environ.get("AUTH_ATTR_CACHE_RETENTION_DAYS", "7")),
        minimum_messages=baseline.minimum_messages,
        minimum_chars=baseline.minimum_chars,
    )


def _worker_score(
    engine: EngineName,
    token: str,
    candidate: AccountProfile,
    suspect: AccountProfile,
    candidate_stream: Stream,
    suspect_stream: Stream,
    assets: ComparisonAssets,
    allow_external: bool,
) -> EngineResult:
    start_queue = _WORKER_START_QUEUE
    if start_queue is not None:
        start_queue.put((token, time.monotonic()))
    try:
        baseline = _read_baseline(assets.baseline_paths[engine])
        if engine == "stylometry":
            artifact = StylometryArtifact.from_dict(
                read_json_artifact(assets.stylometry_artifact_path)
            )
            payload = read_json_artifact(Path(artifact.payload_path))
            config = StylometryConfig(
                feature_spec_version=artifact.feature_spec_version,
                hash_space_size=artifact.hash_space_size,
                ngram_min=int(payload.get("ngram_min", 2)),
                ngram_max=int(payload.get("ngram_max", 4)),
                minimum_messages=baseline.minimum_messages,
                minimum_chars=baseline.minimum_chars,
            )
            return score_stylometry(
                candidate, suspect,
                cast(Sequence[RawStreamChunk], candidate_stream),
                cast(Sequence[RawStreamChunk], suspect_stream),
                artifact=artifact, baseline=baseline, config=config,
            )
        if engine == "chronobiology":
            artifact = ChronobiologyArtifact.from_dict(
                read_json_artifact(assets.chronobiology_artifact_path)
            )
            config = _chrono_config(artifact, baseline)
            return score_chronobiology(
                candidate, suspect,
                cast(Sequence[TemporalStreamChunk], candidate_stream),
                cast(Sequence[TemporalStreamChunk], suspect_stream),
                artifact=artifact, baseline=baseline, config=config,
            )
        return score_semantic_local(
            candidate, suspect,
            cast(Sequence[NormalizedStreamChunk], candidate_stream),
            cast(Sequence[NormalizedStreamChunk], suspect_stream),
            baseline=baseline,
            config=_semantic_config(assets, baseline, allow_external),
        )
    except BaseException as exc:
        return _failure(engine, candidate, suspect, "ERROR", error_code_for(exc))


def _stop_executor(executor: ProcessPoolExecutor, *, terminate: bool) -> None:
    processes = tuple((getattr(executor, "_processes", None) or {}).values())
    if terminate:
        for process in processes:
            try:
                if process.is_alive():
                    process.terminate()
            except Exception:
                pass
        for process in processes:
            try:
                if process.is_alive():
                    process.join(timeout=0.2)
            except Exception:
                pass
    try:
        executor.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass


def _shutdown_manager(manager: object) -> None:
    try:
        manager.shutdown()
    except Exception:
        pass


def _expected_fingerprint(
    engine: EngineName,
    candidate: AccountProfile,
    assets: ComparisonAssets,
    baseline: BaselineDistribution,
    config: RunConfig,
) -> str:
    if engine == "stylometry":
        artifact = StylometryArtifact.from_dict(read_json_artifact(assets.stylometry_artifact_path))
        return artifact.configuration_fingerprint
    if engine == "chronobiology":
        artifact = ChronobiologyArtifact.from_dict(read_json_artifact(assets.chronobiology_artifact_path))
        return artifact.configuration_fingerprint
    semantic_config = _semantic_config(assets, baseline, config.allow_external)
    manifest_path = assets.semantic_model_dir / "manifest.json"
    try:
        manifest_hash = sha256_hex(manifest_path.read_bytes())
    except OSError:
        raise ContractValidationError("manifest", ErrorCode.LOCAL_MODEL_UNAVAILABLE) from None
    try:
        manifest_data = read_json_artifact(manifest_path)
    except ContractValidationError:
        raise ContractValidationError("manifest", ErrorCode.LOCAL_MODEL_INVALID) from None
    tokenizer_hash = manifest_data.get("tokenizer_vocab_sha256")
    if not isinstance(tokenizer_hash, str):
        raise ContractValidationError("manifest", ErrorCode.LOCAL_MODEL_INVALID)
    return semantic_fingerprint(
        semantic_config, manifest_hash, tokenizer_hash, candidate.preprocessing_fingerprint
    )


def _run_engine(
    engine: EngineName,
    candidate: AccountProfile,
    suspect: AccountProfile,
    assets: ComparisonAssets,
    config: RunConfig,
    context: object,
) -> tuple[EngineResult, ProcessPoolExecutor | None, object | None]:
    if candidate.status != "READY" or suspect.status != "READY":
        return _failure(engine, candidate, suspect, "INSUFFICIENT_DATA", ErrorCode.MINIMUMS_NOT_MET), None, None

    manager: object | None = None
    executor: ProcessPoolExecutor | None = None
    try:
        baseline = _read_baseline(assets.baseline_paths[engine])
        if baseline.minimum_messages < 1 or baseline.minimum_chars < 1:
            return _failure(engine, candidate, suspect, "ERROR", ErrorCode.BASELINE_CONFIG_MISMATCH), None, None
        requested_messages = int(os.environ.get("AUTH_ATTR_MINIMUM_MESSAGES", str(baseline.minimum_messages)))
        requested_chars = int(os.environ.get("AUTH_ATTR_MINIMUM_CHARS", str(baseline.minimum_chars)))
        if baseline.minimum_messages != requested_messages or baseline.minimum_chars != requested_chars:
            return _failure(engine, candidate, suspect, "ERROR", ErrorCode.BASELINE_CONFIG_MISMATCH), None, None
        if any(
            profile.n_messages < baseline.minimum_messages or profile.n_chars < baseline.minimum_chars
            for profile in (candidate, suspect)
        ):
            return _failure(engine, candidate, suspect, "INSUFFICIENT_DATA", ErrorCode.MINIMUMS_NOT_MET), None, None
        expected = _expected_fingerprint(engine, candidate, assets, baseline, config)
        incompatibility = check_baseline_compat(baseline, engine, expected)
        if incompatibility is not None:
            return _failure(engine, candidate, suspect, "ERROR", incompatibility), None, None

        if engine == "stylometry":
            left, right = load_raw_stream(candidate), load_raw_stream(suspect)
            timeout_s = config.stylometry_timeout_s
        elif engine == "semantic":
            left, right = load_normalized_stream(candidate), load_normalized_stream(suspect)
            timeout_s = config.semantic_timeout_s
        else:
            left, right = load_temporal_stream(candidate), load_temporal_stream(suspect)
            timeout_s = config.chronobiology_timeout_s

        manager = context.Manager()
        starts = manager.Queue()
        token = f"{engine}-{time.monotonic_ns()}"
        executor = ProcessPoolExecutor(
            max_workers=1, mp_context=context, initializer=_init_worker, initargs=(starts,)
        )
        submitted_at = time.monotonic()
        future = executor.submit(
            _worker_score, engine, token, candidate, suspect, left, right, assets, config.allow_external
        )
        started: float | None = None
        startup_limit = max(
            config.stylometry_timeout_s, config.semantic_timeout_s, config.chronobiology_timeout_s
        )
        while started is None:
            try:
                seen_token, started_at = starts.get(timeout=0.01)
                if seen_token == token:
                    started = started_at
            except queue.Empty:
                if future.done():
                    started = time.monotonic()
                    break
                if time.monotonic() - submitted_at >= startup_limit:
                    _stop_executor(executor, terminate=True)
                    _shutdown_manager(manager)
                    return _failure(engine, candidate, suspect, "TIMEOUT", ErrorCode.ENGINE_TIMEOUT), None, None
        remaining = max(0.0, timeout_s - (time.monotonic() - started))
        try:
            return future.result(timeout=remaining), executor, manager
        except TimeoutError:
            _stop_executor(executor, terminate=True)
            _shutdown_manager(manager)
            return _failure(engine, candidate, suspect, "TIMEOUT", ErrorCode.ENGINE_TIMEOUT), None, None
    except BaseException as exc:
        if executor is not None:
            _stop_executor(executor, terminate=True)
        if manager is not None:
            _shutdown_manager(manager)
        return _failure(engine, candidate, suspect, "ERROR", error_code_for(exc)), None, None


def compare_shortlist(
    flagged: AccountProfile,
    suspects: Sequence[AccountProfile],
    *,
    assets: ComparisonAssets,
    config: RunConfig,
) -> Sequence[ComparisonReport]:
    """Compare each suspect with three independent engines and shared fusion."""
    if not isinstance(flagged, AccountProfile) or not isinstance(assets, ComparisonAssets):
        raise ContractValidationError("input", ErrorCode.INPUT_INVALID)
    if not isinstance(config, RunConfig):
        raise ContractValidationError("config", ErrorCode.INPUT_INVALID)
    flagged._validate()
    assets._validate()
    config._validate()
    for suspect in suspects:
        if not isinstance(suspect, AccountProfile):
            raise ContractValidationError("suspect", ErrorCode.INPUT_INVALID)
        suspect._validate()
    reports: list[ComparisonReport] = []
    context = mp.get_context("spawn")
    for suspect in sorted(suspects, key=lambda item: item.account_id):
        if flagged.account_id == suspect.account_id:
            continue
        outcomes: dict[str, EngineResult] = {}

        def execute(engine: EngineName) -> None:
            executor: ProcessPoolExecutor | None = None
            manager: object | None = None
            try:
                result, executor, manager = _run_engine(engine, flagged, suspect, assets, config, context)
                outcomes[engine] = result
            except BaseException as exc:
                outcomes[engine] = _failure(engine, flagged, suspect, "ERROR", error_code_for(exc))
            finally:
                if executor is not None:
                    _stop_executor(executor, terminate=False)
                if manager is not None:
                    _shutdown_manager(manager)

        threads = [threading.Thread(target=execute, args=(engine,)) for engine in _ENGINE_ORDER]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        stylo = outcomes["stylometry"]
        semantic = outcomes["semantic"]
        chrono = outcomes["chronobiology"]
        combined = compute_combined_result(stylo, semantic, chrono)
        reports.append(
            ComparisonReport(
                schema_version="1.0",
                flagged_account_id=flagged.account_id,
                suspect_account_id=suspect.account_id,
                stylometry=stylo,
                semantic=semantic,
                chronobiology=chrono,
                combined=combined,
                banner=ComparisonReport.BANNER,
            )
        )
    return reports


__all__ = ["compare_shortlist"]
