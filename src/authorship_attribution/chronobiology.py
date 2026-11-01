"""Deterministic, content-free temporal authorship features and scoring."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from datetime import datetime
import math
from pathlib import Path
import re

import numpy as np
from numpy.typing import NDArray

from .calibration import calibrate_cosine, check_baseline_compat
from .contracts import (
    AccountProfile,
    BaselineDistribution,
    ChronobiologyArtifact,
    ChronobiologyConfig,
    EngineResult,
    TemporalStreamChunk,
)
from .errors import ContractValidationError, ErrorCode, error_code_for
from .fileformats import read_json_artifact, write_json_artifact
from .fingerprints import (
    canonical_json,
    chronobiology_fingerprint,
    fingerprint,
    sha256_hex,
)


_FEATURE_COUNT = 122
_BLOCKS = (
    ("circadian_fourier", 0, 79),
    ("burst_process", 79, 108),
    ("session_kinematics", 108, 122),
)
_PAYLOAD_KEYS = frozenset(
    (
        "feature_spec_version",
        "block_weights",
        "feature_means",
        "feature_stds",
        "preprocessing_fingerprint",
        "baseline_corpus_identity",
        "baseline_account_count",
    )
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_STD_EPSILON = 1e-12
_RESULTANT_EPSILON = 1e-9

Event = tuple[float, int]
Session = list[Event]


def _invalid(field_name: str, code: ErrorCode = ErrorCode.INPUT_INVALID) -> None:
    raise ContractValidationError(field_name, code)


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None


def _extract_events(
    stream: Sequence[TemporalStreamChunk],
) -> tuple[str, list[Session]]:
    if not isinstance(stream, Sequence) or isinstance(stream, (str, bytes, bytearray)):
        _invalid("temporal_stream")
    if not stream:
        _invalid("temporal_stream", ErrorCode.MINIMUMS_NOT_MET)

    account_id: str | None = None
    current_session: str | None = None
    completed_sessions: set[str] = set()
    chunk_ids: set[str] = set()
    sessions: list[Session] = []
    previous_epoch: float | None = None

    for chunk in stream:
        if not isinstance(chunk, TemporalStreamChunk):
            _invalid("temporal_stream")
        chunk._validate()
        if account_id is None:
            account_id = chunk.account_id
        elif chunk.account_id != account_id:
            _invalid("account_id")
        if chunk.chunk_id in chunk_ids:
            _invalid("chunk_ids")
        chunk_ids.add(chunk.chunk_id)
        if chunk.session_id != current_session:
            if current_session is not None:
                completed_sessions.add(current_session)
            if chunk.session_id in completed_sessions:
                _invalid("session_id")
            current_session = chunk.session_id
            sessions.append([])

        for timestamp, length in zip(chunk.timestamps_utc, chunk.char_counts, strict=True):
            try:
                epoch = datetime.fromisoformat(f"{timestamp[:-1]}+00:00").timestamp()
            except (OSError, OverflowError, ValueError):
                _invalid("timestamps_utc")
            if not math.isfinite(epoch) or (
                previous_epoch is not None and epoch < previous_epoch
            ):
                _invalid("timestamps_utc")
            sessions[-1].append((epoch, length))
            previous_epoch = epoch

    if account_id is None or not sessions:
        _invalid("temporal_stream", ErrorCode.MINIMUMS_NOT_MET)
    return account_id, sessions


def _mean(values: Sequence[float]) -> float:
    return math.fsum(values) / len(values) if values else 0.0


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = _mean(values)
    variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
    return mean, math.sqrt(max(0.0, variance))


def _correlation(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        _invalid("temporal_features")
    if len(left) < 2:
        return 0.0
    left_mean, left_std = _mean_std(left)
    right_mean, right_std = _mean_std(right)
    if left_std < _STD_EPSILON or right_std < _STD_EPSILON:
        return 0.0
    value = math.fsum(
        ((a - left_mean) / left_std) * ((b - right_mean) / right_std)
        for a, b in zip(left, right, strict=True)
    ) / len(left)
    return min(1.0, max(-1.0, value))


def _harmonic(times: Sequence[float], harmonic: int, period: float, shift: float) -> tuple[float, float, float]:
    real = math.fsum(
        math.cos(2.0 * math.pi * harmonic * (value + shift) / period)
        for value in times
    ) / len(times)
    imag = math.fsum(
        math.sin(2.0 * math.pi * harmonic * (value + shift) / period)
        for value in times
    ) / len(times)
    return real, imag, math.hypot(real, imag)


def _smoothed_activity(times: Sequence[float]) -> list[float]:
    counts = [0] * 48
    for value in times:
        counts[int((value % 86400.0) // 1800.0)] += 1
    frequencies = [count / len(times) for count in counts]
    unnormalized = [
        math.fsum(
            frequencies[r] * math.exp(-0.5 * min(abs(b - r), 48 - abs(b - r)) ** 2)
            for r in range(48)
        )
        for b in range(48)
    ]
    total = math.fsum(unnormalized)
    return [value / total for value in unnormalized]


def _circadian_features(times: Sequence[float]) -> list[float]:
    result: list[float] = []
    for harmonic in (1, 2, 3, 4, 6):
        result.extend(_harmonic(times, harmonic, 86400.0, 0.0))
    for harmonic in (1, 2, 3):
        result.extend(_harmonic(times, harmonic, 604800.0, 259200.0))

    smoothed = _smoothed_activity(times)
    result.extend(smoothed)
    entropy = -math.fsum(value * math.log(value) for value in smoothed if value > 0.0)
    real = math.fsum(
        value * math.cos(2.0 * math.pi * (index + 0.5) / 48.0)
        for index, value in enumerate(smoothed)
    )
    imag = math.fsum(
        value * math.sin(2.0 * math.pi * (index + 0.5) / 48.0)
        for index, value in enumerate(smoothed)
    )
    variance = min(1.0, max(0.0, 1.0 - math.hypot(real, imag)))
    masses = [
        math.fsum(smoothed[(start + offset) % 48] for offset in range(12))
        for start in range(48)
    ]
    trough = min(range(48), key=lambda start: (round(masses[start], 12), start))
    trough_angle = 2.0 * math.pi * ((trough + 6) % 48) / 48.0
    result.extend((entropy, variance, math.cos(trough_angle), math.sin(trough_angle), masses[trough]))

    weekday: list[float] = []
    weekend: list[float] = []
    for value in times:
        day = int(math.floor((value + 259200.0) / 86400.0)) % 7
        (weekend if day >= 5 else weekday).append(value)
    result.append(len(weekend) / len(times))
    phase_shift = 0.0
    if weekday and weekend:
        weekday_real, weekday_imag, weekday_resultant = _harmonic(weekday, 1, 86400.0, 0.0)
        weekend_real, weekend_imag, weekend_resultant = _harmonic(weekend, 1, 86400.0, 0.0)
        if weekday_resultant > _RESULTANT_EPSILON and weekend_resultant > _RESULTANT_EPSILON:
            phase_shift = (
                weekend_real * weekday_real + weekend_imag * weekday_imag
            ) / (weekend_resultant * weekday_resultant)
            phase_shift = min(1.0, max(-1.0, phase_shift))
    result.append(phase_shift)
    return result


def _clamped_interval(previous: float, current: float) -> float:
    return min(1800.0, max(0.5, current - previous))


def _session_intervals(sessions: Sequence[Session]) -> list[list[float]]:
    return [
        [_clamped_interval(left[0], right[0]) for left, right in zip(session, session[1:])]
        for session in sessions
    ]


def _fano_feature(sessions: Sequence[Session], width: float) -> float:
    windows = 0
    total = 0
    squared_counts = 0
    for session in sessions:
        first = session[0][0]
        duration = session[-1][0] - first
        session_windows = max(1, int(math.ceil(duration / width)))
        counts = Counter(
            min(session_windows - 1, int((timestamp - first) // width))
            for timestamp, _ in session
        )
        windows += session_windows
        total += len(session)
        squared_counts += sum(count * count for count in counts.values())
    mean = total / windows
    variance = max(0.0, squared_counts / windows - mean * mean)
    return math.log1p(variance / max(1e-9, mean))


def _burst_features(sessions: Sequence[Session], times: Sequence[float]) -> list[float]:
    per_session = _session_intervals(sessions)
    intervals = [value for values in per_session for value in values]
    if not intervals:
        intervals = [
            _clamped_interval(left, right)
            for left, right in zip(times, times[1:])
        ]
        if not intervals:
            intervals = [1800.0]

    tau = [math.log1p(value) for value in intervals]
    histogram = np.histogram(
        np.asarray(tau, dtype=np.float64),
        bins=16,
        range=(0.0, math.log(1801.0)),
    )[0]
    result = [float(value) / len(tau) for value in histogram]
    tau_mean, tau_std = _mean_std(tau)
    if tau_std < _STD_EPSILON:
        tau_std = 0.0
        skew = 0.0
        kurtosis = 0.0
    else:
        skew = math.fsum(((value - tau_mean) / tau_std) ** 3 for value in tau) / len(tau)
        kurtosis = math.fsum(((value - tau_mean) / tau_std) ** 4 for value in tau) / len(tau) - 3.0
    quantiles = np.quantile(np.asarray(tau, dtype=np.float64), [0.10, 0.50, 0.90], method="linear")
    result.extend((tau_mean, tau_std, skew, kurtosis))
    result.extend(float(value) for value in quantiles)
    interval_mean, interval_std = _mean_std(intervals)
    result.append((interval_std - interval_mean) / (interval_std + interval_mean))

    left: list[float] = []
    right: list[float] = []
    for values in per_session:
        for first, second in zip(values, values[1:]):
            left.append(first)
            right.append(second)
    result.append(_correlation(left, right) if len(left) >= 3 else 0.0)
    result.extend(_fano_feature(sessions, width) for width in (60.0, 300.0, 900.0, 1800.0))
    return result


def _session_features(sessions: Sequence[Session]) -> list[float]:
    velocities: list[float] = []
    log_lengths: list[float] = []
    log_intervals: list[float] = []
    for session in sessions:
        for previous, current in zip(session, session[1:]):
            interval = _clamped_interval(previous[0], current[0])
            length = current[1]
            log_lengths.append(math.log1p(length))
            log_intervals.append(math.log1p(interval))
            if interval <= 120.0:
                velocities.append(length / max(1.0, interval))

    if velocities:
        velocity_mean, velocity_std = _mean_std(velocities)
        velocity_quantiles = np.quantile(
            np.asarray(velocities, dtype=np.float64), [0.50, 0.90], method="linear"
        )
        result = [velocity_mean, velocity_std, float(velocity_quantiles[0]), float(velocity_quantiles[1])]
    else:
        result = [0.0, 0.0, 0.0, 0.0]
    result.append(_correlation(log_lengths, log_intervals))

    sizes = [float(len(session)) for session in sessions]
    durations = [session[-1][0] - session[0][0] for session in sessions]
    size_mean, size_std = _mean_std(sizes)
    _, log_duration_std = _mean_std([math.log1p(value) for value in durations])
    result.extend(
        (
            math.log1p(size_mean),
            size_std / size_mean,
            math.log1p(_mean(durations)),
            log_duration_std,
            sum(len(session) == 1 for session in sessions) / len(sessions),
        )
    )
    for endpoint in (0, -1):
        result.append(
            math.fsum(math.cos(2.0 * math.pi * session[endpoint][0] / 86400.0) for session in sessions)
            / len(sessions)
        )
        result.append(
            math.fsum(math.sin(2.0 * math.pi * session[endpoint][0] / 86400.0) for session in sessions)
            / len(sessions)
        )
    return result


def extract_chronobiology_features(
    stream: Sequence[TemporalStreamChunk],
) -> NDArray[np.float64]:
    """Return the finite, unstandardized 122-feature account vector."""
    _, sessions = _extract_events(stream)
    times = [timestamp for session in sessions for timestamp, _ in session]
    try:
        features = (
            _circadian_features(times)
            + _burst_features(sessions, times)
            + _session_features(sessions)
        )
        result = np.asarray(features, dtype=np.float64)
        if result.shape != (_FEATURE_COUNT,) or not np.isfinite(result).all():
            _invalid("temporal_features", ErrorCode.ENGINE_FAILURE)
        return result
    except ContractValidationError:
        raise
    except (ArithmeticError, ValueError, TypeError):
        _invalid("temporal_features", ErrorCode.ENGINE_FAILURE)
    raise AssertionError("unreachable")


def fit_chronobiology_artifact(
    baseline_streams: Iterable[Sequence[TemporalStreamChunk]],
    *,
    config: ChronobiologyConfig,
    preprocessing_fp: str,
    output_dir: Path,
) -> ChronobiologyArtifact:
    """Fit population feature statistics from eligible baseline accounts."""
    if not isinstance(config, ChronobiologyConfig):
        _invalid("config")
    config._validate()
    if not _is_sha256(preprocessing_fp):
        _invalid("preprocessing_fingerprint")
    if not isinstance(output_dir, Path):
        _invalid("output_dir")
    if not isinstance(baseline_streams, Iterable) or isinstance(baseline_streams, (str, bytes, bytearray)):
        _invalid("baseline_streams")

    accounts: list[tuple[str, Sequence[TemporalStreamChunk]]] = []
    seen: set[str] = set()
    try:
        for stream in baseline_streams:
            if not isinstance(stream, Sequence) or isinstance(stream, (str, bytes, bytearray)):
                _invalid("temporal_stream")
            if not stream:
                continue
            account_id, _ = _extract_events(stream)
            if account_id in seen:
                _invalid("account_id")
            seen.add(account_id)
            n_messages = sum(chunk.n_messages for chunk in stream)
            n_chars = sum(chunk.n_chars for chunk in stream)
            if n_messages >= config.minimum_messages and n_chars >= config.minimum_chars:
                accounts.append((account_id, stream))
    except ContractValidationError:
        raise
    except Exception:
        _invalid("baseline_streams", ErrorCode.ENGINE_FAILURE)
    if len(accounts) < 2:
        _invalid("baseline_streams", ErrorCode.MINIMUMS_NOT_MET)
    accounts.sort(key=lambda item: item[0])

    try:
        rows = [extract_chronobiology_features(stream) for _, stream in accounts]
        means: list[float] = []
        stds: list[float] = []
        for column in range(_FEATURE_COUNT):
            values = [float(row[column]) for row in rows]
            mean, std = _mean_std(values)
            means.append(mean)
            stds.append(std)
        identity_entries = [
            {
                "account_id_sha256": sha256_hex(account_id.encode("utf-8", errors="strict")),
                "stream_sha256": sha256_hex(canonical_json([chunk.to_dict() for chunk in stream])),
                "n_messages": sum(chunk.n_messages for chunk in stream),
                "n_chars": sum(chunk.n_chars for chunk in stream),
            }
            for account_id, stream in accounts
        ]
        payload: dict[str, object] = {
            "feature_spec_version": config.feature_spec_version,
            "block_weights": dict(config.block_weights),
            "feature_means": means,
            "feature_stds": stds,
            "preprocessing_fingerprint": preprocessing_fp,
            "baseline_corpus_identity": fingerprint({"accounts": identity_entries}),
            "baseline_account_count": len(accounts),
        }
        payload_bytes = canonical_json(payload)
        artifact_sha256 = sha256_hex(payload_bytes)
        artifact_id = f"chronobiology-{artifact_sha256}"
        payload_path = output_dir / f"{artifact_id}.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        if payload_path.exists():
            if payload_path.read_bytes() != payload_bytes:
                _invalid("artifact", ErrorCode.ARTIFACT_INVALID)
        else:
            write_json_artifact(payload, payload_path)
        return ChronobiologyArtifact(
            artifact_id=artifact_id,
            artifact_sha256=artifact_sha256,
            feature_spec_version=config.feature_spec_version,
            feature_means=means,
            feature_stds=stds,
            payload_path=str(payload_path),
            configuration_fingerprint=chronobiology_fingerprint(
                config, artifact_sha256, preprocessing_fp
            ),
        )
    except ContractValidationError:
        raise
    except (OSError, UnicodeError, TypeError, ValueError):
        _invalid("artifact", ErrorCode.ARTIFACT_INVALID)
    except ArithmeticError:
        _invalid("temporal_features", ErrorCode.ENGINE_FAILURE)
    raise AssertionError("unreachable")


def _load_payload(
    artifact: ChronobiologyArtifact,
    config: ChronobiologyConfig,
) -> dict[str, object]:
    if not isinstance(artifact, ChronobiologyArtifact):
        _invalid("artifact")
    if not isinstance(config, ChronobiologyConfig):
        _invalid("config")
    config._validate()
    try:
        artifact._validate()
    except ContractValidationError:
        _invalid("artifact", ErrorCode.ARTIFACT_INVALID)
    payload = read_json_artifact(Path(artifact.payload_path))
    if set(payload) != _PAYLOAD_KEYS:
        _invalid("artifact", ErrorCode.ARTIFACT_INVALID)
    if sha256_hex(canonical_json(payload)) != artifact.artifact_sha256:
        _invalid("artifact_sha256", ErrorCode.ARTIFACT_INVALID)
    if artifact.artifact_id != f"chronobiology-{artifact.artifact_sha256}":
        _invalid("artifact_id", ErrorCode.ARTIFACT_INVALID)
    feature_version = payload["feature_spec_version"]
    weights = payload["block_weights"]
    means = payload["feature_means"]
    stds = payload["feature_stds"]
    preprocessing_fp = payload["preprocessing_fingerprint"]
    identity = payload["baseline_corpus_identity"]
    account_count = payload["baseline_account_count"]
    if not isinstance(feature_version, str) or not feature_version:
        _invalid("artifact", ErrorCode.ARTIFACT_INVALID)
    if type(weights) is not dict or set(weights) != set(config.block_weights):
        _invalid("artifact", ErrorCode.ARTIFACT_INVALID)
    if any(type(value) is not float or not math.isfinite(value) or value < 0.0 or value > 1.0 for value in weights.values()):
        _invalid("artifact", ErrorCode.ARTIFACT_INVALID)
    if not math.isclose(math.fsum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-12):
        _invalid("artifact", ErrorCode.ARTIFACT_INVALID)
    if type(means) is not list or len(means) != _FEATURE_COUNT:
        _invalid("artifact", ErrorCode.ARTIFACT_INVALID)
    if type(stds) is not list or len(stds) != _FEATURE_COUNT:
        _invalid("artifact", ErrorCode.ARTIFACT_INVALID)
    if any(type(value) is not float or not math.isfinite(value) for value in means):
        _invalid("artifact", ErrorCode.ARTIFACT_INVALID)
    if any(type(value) is not float or not math.isfinite(value) or value < 0.0 for value in stds):
        _invalid("artifact", ErrorCode.ARTIFACT_INVALID)
    if not _is_sha256(preprocessing_fp) or not _is_sha256(identity):
        _invalid("artifact", ErrorCode.ARTIFACT_INVALID)
    if type(account_count) is not int or account_count < 2:
        _invalid("artifact", ErrorCode.ARTIFACT_INVALID)
    if feature_version != artifact.feature_spec_version or means != artifact.feature_means or stds != artifact.feature_stds:
        _invalid("artifact", ErrorCode.ARTIFACT_INVALID)
    if feature_version != config.feature_spec_version or weights != config.block_weights:
        _invalid("configuration", ErrorCode.BASELINE_CONFIG_MISMATCH)
    expected = chronobiology_fingerprint(config, artifact.artifact_sha256, preprocessing_fp)
    if expected != artifact.configuration_fingerprint:
        _invalid("configuration", ErrorCode.BASELINE_CONFIG_MISMATCH)
    return payload


def _account_vector(
    stream: Sequence[TemporalStreamChunk],
    artifact: ChronobiologyArtifact,
    config: ChronobiologyConfig,
) -> list[float]:
    features = extract_chronobiology_features(stream)
    standardized: list[float] = []
    for value, mean, std in zip(features, artifact.feature_means, artifact.feature_stds, strict=True):
        standardized.append(0.0 if std == 0.0 else (float(value) - mean) / std)
    if any(not math.isfinite(value) for value in standardized):
        _invalid("chronobiology_norm", ErrorCode.ENGINE_FAILURE)
    combined: list[float] = []
    for name, start, end in _BLOCKS:
        block = standardized[start:end]
        norm = math.hypot(*block)
        if not math.isfinite(norm):
            _invalid("chronobiology_norm", ErrorCode.ENGINE_FAILURE)
        if norm == 0.0:
            combined.extend([0.0] * len(block))
        else:
            combined.extend(config.block_weights[name] * value / norm for value in block)
    final_norm = math.hypot(*combined)
    if final_norm == 0.0 or not math.isfinite(final_norm):
        _invalid("chronobiology_norm", ErrorCode.ENGINE_FAILURE)
    return [value / final_norm for value in combined]


def compute_chronobiology_raw(
    candidate_stream: Sequence[TemporalStreamChunk],
    suspect_stream: Sequence[TemporalStreamChunk],
    *,
    artifact: ChronobiologyArtifact,
    config: ChronobiologyConfig,
) -> float:
    """Compute the normalized account-vector cosine."""
    _load_payload(artifact, config)
    left = _account_vector(candidate_stream, artifact, config)
    right = _account_vector(suspect_stream, artifact, config)
    cosine = math.fsum(a * b for a, b in zip(left, right, strict=True))
    return min(1.0, max(-1.0, cosine))


def _validate_profile_stream(
    profile: AccountProfile,
    stream: Sequence[TemporalStreamChunk],
) -> None:
    if not stream:
        _invalid("temporal_stream")
    account_id, sessions = _extract_events(stream)
    if account_id != profile.account_id:
        _invalid("account_id")
    if [chunk.chunk_id for chunk in stream] != profile.chunk_ids:
        _invalid("chunk_ids")
    if sum(chunk.n_messages for chunk in stream) != profile.n_messages:
        _invalid("n_messages")
    if sum(chunk.n_chars for chunk in stream) != profile.n_chars:
        _invalid("n_chars")
    if len(sessions) != profile.session_count:
        _invalid("session_count")


def score_chronobiology(
    candidate: AccountProfile,
    suspect: AccountProfile,
    candidate_stream: Sequence[TemporalStreamChunk],
    suspect_stream: Sequence[TemporalStreamChunk],
    *,
    artifact: ChronobiologyArtifact,
    baseline: BaselineDistribution,
    config: ChronobiologyConfig,
) -> EngineResult:
    """Score and calibrate one pair using the temporal projection only."""
    if not isinstance(candidate, AccountProfile) or not isinstance(suspect, AccountProfile):
        _invalid("profile")
    candidate._validate()
    suspect._validate()
    counts_messages = {"candidate": candidate.n_messages, "suspect": suspect.n_messages}
    counts_chars = {"candidate": candidate.n_chars, "suspect": suspect.n_chars}

    def failure(status: str, code: ErrorCode) -> EngineResult:
        return EngineResult(
            engine="chronobiology",
            raw_cosine=None,
            calibrated_percentile=None,
            status=status,
            n_messages=counts_messages,
            n_chars=counts_chars,
            error=code.value,
        )

    try:
        if not isinstance(config, ChronobiologyConfig):
            _invalid("config")
        config._validate()
        if any(
            profile.status != "READY"
            or profile.n_messages < config.minimum_messages
            or profile.n_chars < config.minimum_chars
            for profile in (candidate, suspect)
        ):
            return failure("INSUFFICIENT_DATA", ErrorCode.MINIMUMS_NOT_MET)
        if not isinstance(baseline, BaselineDistribution):
            _invalid("baseline")
        if baseline.minimum_messages != config.minimum_messages or baseline.minimum_chars != config.minimum_chars:
            _invalid("baseline", ErrorCode.BASELINE_CONFIG_MISMATCH)
        payload = _load_payload(artifact, config)
        if (
            candidate.preprocessing_fingerprint != payload["preprocessing_fingerprint"]
            or suspect.preprocessing_fingerprint != payload["preprocessing_fingerprint"]
        ):
            _invalid("preprocessing_fingerprint", ErrorCode.BASELINE_CONFIG_MISMATCH)
        incompatibility = check_baseline_compat(
            baseline, "chronobiology", artifact.configuration_fingerprint
        )
        if incompatibility is not None:
            _invalid("baseline", incompatibility)
        _validate_profile_stream(candidate, candidate_stream)
        _validate_profile_stream(suspect, suspect_stream)
        raw = compute_chronobiology_raw(
            candidate_stream, suspect_stream, artifact=artifact, config=config
        )
        percentile = calibrate_cosine(raw, baseline)
        return EngineResult(
            engine="chronobiology",
            raw_cosine=raw,
            calibrated_percentile=percentile,
            status="OK",
            n_messages=counts_messages,
            n_chars=counts_chars,
            error=None,
        )
    except Exception as exc:
        return failure("ERROR", error_code_for(exc))


__all__ = [
    "extract_chronobiology_features",
    "fit_chronobiology_artifact",
    "compute_chronobiology_raw",
    "score_chronobiology",
]
