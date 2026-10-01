"""Known-different baseline construction and refresh decisions."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
import logging
from pathlib import Path
import random
from typing import Literal

from .contracts import (
    AccountProfile,
    BaselineConfig,
    BaselineDistribution,
    EmbeddingModel,
    LabeledPair,
    RawStreamChunk,
    SemanticConfig,
    StylometryArtifact,
    StylometryConfig,
)
from .errors import ContractValidationError, ErrorCode
from .fileformats import read_json_artifact, write_json_artifact
from .fingerprints import canonical_json, semantic_fingerprint, sha256_hex
from .semantic import compute_semantic_raw
from .streams import load_normalized_stream, load_raw_stream
from .stylometry import compute_stylometry_raw

LOGGER = logging.getLogger(__name__)


def _stratum(profile_a: AccountProfile, profile_b: AccountProfile) -> str:
    """Return a coarse, non-identifying sampling stratum."""
    game = "same-game" if profile_a.primary_game_id == profile_b.primary_game_id else "cross-game"
    channel = "same-channel" if profile_a.primary_channel_id is not None and profile_a.primary_channel_id == profile_b.primary_channel_id else "other-channel"
    months_a, months_b = set(profile_a.active_months_utc), set(profile_b.active_months_utc)
    month = "same-month" if months_a & months_b else "other-month"
    return f"{game}|{channel}|{month}"


def _eligible_pairs(
    training_pairs: Sequence[LabeledPair], profiles: Mapping[str, AccountProfile], config: BaselineConfig
) -> list[tuple[LabeledPair, str]]:
    seen: set[tuple[str, str]] = set()
    result: list[tuple[LabeledPair, str]] = []
    for pair in training_pairs:
        if pair.label != "different":
            continue
        key = (pair.account_id_a, pair.account_id_b)
        if key in seen:
            continue
        seen.add(key)
        left, right = profiles.get(key[0]), profiles.get(key[1])
        if left is None or right is None:
            continue
        minimum_messages = max(config_minimum(pair, left, right)[0], 0)
        minimum_chars = max(config_minimum(pair, left, right)[1], 0)
        if all(p.status == "READY" and p.n_messages >= minimum_messages and p.n_chars >= minimum_chars for p in (left, right)):
            result.append((pair, _stratum(left, right)))
    return result


def config_minimum(pair: LabeledPair, left: AccountProfile, right: AccountProfile) -> tuple[int, int]:
    """Return the common profile minimums without relaxing either account."""
    del pair
    return max(left.minimum_messages, right.minimum_messages), max(left.minimum_chars, right.minimum_chars)


def _sample_stratified(
    eligible: Sequence[tuple[LabeledPair, str]], maximum: int, seed: int
) -> list[tuple[LabeledPair, str]]:
    groups: dict[str, list[tuple[LabeledPair, str]]] = defaultdict(list)
    for item in eligible:
        groups[item[1]].append(item)
    rng = random.Random(seed)
    for values in groups.values():
        values.sort(key=lambda item: (item[0].account_id_a, item[0].account_id_b, item[0].source_snapshot_id))
        rng.shuffle(values)
    total = min(maximum, len(eligible))
    names = sorted(groups)
    allocations = {name: (total * len(groups[name])) // len(eligible) for name in names}
    remaining = total - sum(allocations.values())
    for name in sorted(names, key=lambda key: (-(total * len(groups[key]) % len(eligible)), key)):
        if remaining == 0:
            break
        if allocations[name] < len(groups[name]):
            allocations[name] += 1
            remaining -= 1
    selected = [item for name in names for item in groups[name][:allocations[name]]]
    return sorted(selected, key=lambda item: (item[0].account_id_a, item[0].account_id_b, item[0].source_snapshot_id))


def _snapshot_id(pairs: Sequence[LabeledPair]) -> str:
    snapshots = sorted({pair.source_snapshot_id for pair in pairs})
    return "snapshot-" + sha256_hex(canonical_json(snapshots))


def _baseline(
    engine: str,
    scores: Sequence[float],
    pairs: Sequence[LabeledPair],
    strata: Sequence[str],
    fingerprint: str,
    seed: int,
    minimum_messages: int,
    minimum_chars: int,
) -> BaselineDistribution:
    ordered = sorted(float(value) for value in scores)
    source = _snapshot_id(pairs)
    created = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    baseline_id = f"{engine}-baseline-{sha256_hex(canonical_json({ 'scores': ordered, 'source': source, 'fingerprint': fingerprint, 'seed': seed }))}"
    return BaselineDistribution(
        schema_version="1.0", baseline_id=baseline_id, engine=engine, created_at_utc=created,
        source_snapshot_id=source, configuration_fingerprint=fingerprint, random_seed=seed,
        calibration_population="known_different_account_pairs", n_pairs=len(ordered),
        sorted_raw_cosines=ordered, minimum_messages=minimum_messages, minimum_chars=minimum_chars,
        strata_counts=dict(sorted(Counter(strata).items())), centering_vector_path=None,
    )


def build_baselines(
    training_pairs: Sequence[LabeledPair],
    profiles: Mapping[str, AccountProfile],
    *,
    stylometry_artifact: StylometryArtifact,
    semantic_model: EmbeddingModel,
    config: BaselineConfig,
    seed: int,
    output_dir: Path,
) -> Mapping[Literal["stylometry", "semantic"], BaselineDistribution]:
    """Build per-engine empirical baselines from reviewed different-label pairs."""
    if not isinstance(config, BaselineConfig) or not isinstance(output_dir, Path):
        raise ContractValidationError("input", ErrorCode.INPUT_INVALID)
    eligible = _eligible_pairs(training_pairs, profiles, config)
    required = max(1000, config.min_pairs)
    if len(eligible) < required:
        raise ContractValidationError("known_different_pairs", ErrorCode.MINIMUMS_NOT_MET)
    sampled = _sample_stratified(eligible, config.max_pairs, seed)
    if len(sampled) < required:
        raise ContractValidationError("known_different_pairs", ErrorCode.MINIMUMS_NOT_MET)
    min_messages = max(profiles[p.account_id_a].minimum_messages for p, _ in sampled)
    min_messages = max(min_messages, max(profiles[p.account_id_b].minimum_messages for p, _ in sampled))
    min_chars = max(profiles[p.account_id_a].minimum_chars for p, _ in sampled)
    min_chars = max(min_chars, max(profiles[p.account_id_b].minimum_chars for p, _ in sampled))
    artifact_payload = read_json_artifact(Path(stylometry_artifact.payload_path))
    preprocessing_fps = {profiles[pair.account_id_a].preprocessing_fingerprint for pair, _ in sampled}
    preprocessing_fps.update(profiles[pair.account_id_b].preprocessing_fingerprint for pair, _ in sampled)
    if len(preprocessing_fps) != 1:
        raise ContractValidationError("preprocessing_fingerprint", ErrorCode.BASELINE_CONFIG_MISMATCH)
    stylo_config = StylometryConfig(
        feature_spec_version=stylometry_artifact.feature_spec_version,
        hash_space_size=stylometry_artifact.hash_space_size,
        ngram_min=int(artifact_payload.get("ngram_min", 2)),
        ngram_max=int(artifact_payload.get("ngram_max", 4)),
        minimum_messages=min_messages, minimum_chars=min_chars,
    )
    semantic_config = SemanticConfig(
        model_dir=Path("."), cache_dir=output_dir / "cache", center_embeddings=False,
        minimum_messages=min_messages, minimum_chars=min_chars,
    )
    semantic_fp = semantic_fingerprint(
        semantic_config, semantic_model.model_fingerprint,
        getattr(semantic_model, "tokenizer_sha256", semantic_model.model_fingerprint),
        next(iter(preprocessing_fps)),
    )
    stylo_scores: list[float] = []
    semantic_scores: list[float] = []
    chosen_pairs: list[LabeledPair] = []
    chosen_strata: list[str] = []
    for pair, stratum in sampled:
        left, right = profiles[pair.account_id_a], profiles[pair.account_id_b]
        stylo_scores.append(compute_stylometry_raw(
            load_raw_stream(left), load_raw_stream(right), artifact=stylometry_artifact, config=stylo_config
        ))
        semantic_scores.append(compute_semantic_raw(
            load_normalized_stream(left), load_normalized_stream(right), model=semantic_model, config=semantic_config
        ))
        chosen_pairs.append(pair)
        chosen_strata.append(stratum)
    output_dir.mkdir(parents=True, exist_ok=True)
    stylometry = _baseline("stylometry", stylo_scores, chosen_pairs, chosen_strata,
                           stylometry_artifact.configuration_fingerprint, seed, min_messages, min_chars)
    semantic = _baseline("semantic", semantic_scores, chosen_pairs, chosen_strata,
                         semantic_fp, seed, min_messages, min_chars)
    for artifact in (stylometry, semantic):
        path = output_dir / f"{artifact.engine}-{artifact.baseline_id.rsplit('-', 1)[-1]}.json"
        # Baseline outputs are immutable: never replace an existing baseline.
        if path.exists():
            existing = BaselineDistribution.from_dict(read_json_artifact(path))
            existing_data = existing.to_dict()
            candidate_data = artifact.to_dict()
            existing_data.pop("created_at_utc")
            candidate_data.pop("created_at_utc")
            if existing_data != candidate_data:
                raise ContractValidationError("baseline_path", ErrorCode.ARTIFACT_INVALID)
            if artifact.engine == "stylometry":
                stylometry = existing
            else:
                semantic = existing
        else:
            write_json_artifact(artifact, path)
    LOGGER.info("baseline_built engines=2 pairs=%d seed=%d", len(chosen_pairs), seed)
    return {"stylometry": stylometry, "semantic": semantic}


def baseline_needs_refresh(
    baseline: BaselineDistribution,
    *,
    current_fingerprint: str,
    new_eligible_pairs: int,
    config: BaselineConfig,
    now: datetime | None = None,
) -> bool:
    """Evaluate monthly, corpus-growth, and configuration refresh conditions."""
    baseline._validate()
    current = now or datetime.now(UTC)
    created = datetime.fromisoformat(baseline.created_at_utc.replace("Z", "+00:00"))
    age_days = (current - created).total_seconds() / 86400
    monthly_boundary_crossed = (created.year, created.month) != (current.year, current.month)
    return (
        baseline.configuration_fingerprint != current_fingerprint
        or monthly_boundary_crossed
        or age_days >= config.refresh_max_age_days
        or new_eligible_pairs / max(1, baseline.n_pairs) >= config.refresh_new_pair_fraction
    )


__all__ = ["baseline_needs_refresh", "build_baselines"]
