"""Held-out labeled-pair evaluation with account-disjoint partitions."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import math
from pathlib import Path
import random
from typing import Literal

from .contracts import (
    AccountProfile,
    BaselineDistribution,
    ChronobiologyConfig,
    ComparisonAssets,
    EngineResult,
    EngineEvaluation,
    EvaluationConfig,
    EvaluationReport,
    LabeledPair,
    SemanticConfig,
    StylometryArtifact,
    StylometryConfig,
)
from .calibration import calibrate_cosine, compute_combined_result
from .chronobiology import compute_chronobiology_raw, fit_chronobiology_artifact
from .errors import ContractValidationError, ErrorCode, error_code_for
from .fileformats import load_term_list, write_json_artifact
from .fingerprints import semantic_fingerprint
from .semantic import compute_semantic_raw, load_local_model
from .streams import load_normalized_stream, load_raw_stream, load_temporal_stream
from .stylometry import compute_stylometry_raw, fit_stylometry_artifact


def _components(pairs: Sequence[LabeledPair]) -> list[set[str]]:
    graph: dict[str, set[str]] = defaultdict(set)
    for pair in pairs:
        if pair.label == "same":
            graph[pair.account_id_a].add(pair.account_id_b)
            graph[pair.account_id_b].add(pair.account_id_a)
    groups: list[set[str]] = []
    visited: set[str] = set()
    for account in sorted(graph):
        if account in visited:
            continue
        stack = [account]
        group: set[str] = set()
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            group.add(current)
            stack.extend(sorted(graph[current] - visited, reverse=True))
        groups.append(group)
    grouped = set().union(*groups) if groups else set()
    for account in sorted(({p.account_id_a for p in pairs} | {p.account_id_b for p in pairs}) - grouped):
        if account not in grouped:
            groups.append({account})
    return sorted(groups, key=lambda group: min(group))


def _partition_accounts(
    pairs: Sequence[LabeledPair], seed: int, fractions: tuple[float, float, float]
) -> dict[str, str]:
    groups = _components(pairs)
    random.Random(seed).shuffle(groups)
    assignments: dict[str, str] = {}
    names = ("train", "validation", "test")
    for index, group in enumerate(groups):
        normalized = (index + 0.5) / max(1, len(groups))
        if normalized < fractions[0]:
            partition = names[0]
        elif normalized < fractions[0] + fractions[1]:
            partition = names[1]
        else:
            partition = names[2]
        for account in group:
            assignments[account] = partition
    return assignments


def _auc(rows: Sequence[tuple[float, bool]]) -> float | None:
    positives = sum(label for _, label in rows)
    negatives = len(rows) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(rows, key=lambda row: row[0])
    rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = (index + 1 + end) / 2
        rank_sum += average_rank * sum(label for _, label in ordered[index:end])
        index = end
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def _pr_auc(rows: Sequence[tuple[float, bool]]) -> float | None:
    positives = sum(label for _, label in rows)
    if positives == 0:
        return None
    ordered = sorted(rows, key=lambda row: (-row[0], not row[1]))
    tp = fp = 0
    previous_recall = 0.0
    area = 0.0
    for _, label in ordered:
        tp += int(label)
        fp += int(not label)
        recall = tp / positives
        precision = tp / (tp + fp)
        area += (recall - previous_recall) * precision
        previous_recall = recall
    return area


def _metrics(rows: Sequence[tuple[float, bool]], threshold: float) -> tuple[float, float]:
    tp = sum(label and score >= threshold for score, label in rows)
    fp = sum((not label) and score >= threshold for score, label in rows)
    fn = sum(label and score < threshold for score, label in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return precision, recall


def _threshold(
    rows: Sequence[tuple[float, bool]], minimum_precision: float
) -> tuple[float, bool]:
    if not rows:
        return 100.0, False
    candidates = {score for score, _ in rows}
    sentinel = math.nextafter(max(candidates), math.inf)
    if sentinel <= 100.0:
        candidates.add(sentinel)
    scored = [(threshold, *_metrics(rows, threshold)) for threshold in sorted(candidates)]
    qualifying = [row for row in scored if row[1] >= minimum_precision]
    pool = qualifying if qualifying else scored
    best = max(pool, key=lambda row: (row[2], row[1], -row[0]))
    return float(best[0]), bool(qualifying)

def _ci(values: Sequence[float], seed: int, level: float, n_bootstrap: int) -> list[float]:
    if not values:
        return [0.0, 0.0]
    rng = random.Random(seed)
    samples = sorted(
        sum(rng.choice(values) for _ in values) / len(values)
        for _ in range(n_bootstrap)
    )
    alpha = (1.0 - level) / 2.0
    low = min(len(samples) - 1, max(0, int(alpha * len(samples))))
    high = min(len(samples) - 1, max(0, int((1.0 - alpha) * len(samples)) - 1))
    return [float(samples[low]), float(samples[high])]


def _bootstrap_metric_ci(
    rows: Sequence[tuple[float, bool]],
    *,
    metric: Literal["roc_auc", "precision", "recall"],
    threshold: float,
    config: EvaluationConfig,
    seed_offset: int,
) -> list[float]:
    if not rows:
        return [0.0, 0.0]
    rng = random.Random(config.seed + seed_offset)
    values: list[float] = []
    for _ in range(config.n_bootstrap):
        sample = [rng.choice(rows) for _ in rows]
        if metric == "roc_auc":
            value = _auc(sample)
            if value is None:
                continue
        else:
            precision, recall = _metrics(sample, threshold)
            value = precision if metric == "precision" else recall
        values.append(float(value))
    if not values:
        return [0.0, 0.0]
    values.sort()
    alpha = (1.0 - config.ci_level) / 2.0
    low = min(len(values) - 1, max(0, int(alpha * len(values))))
    high = min(len(values) - 1, max(0, int((1.0 - alpha) * len(values)) - 1))
    return [values[low], values[high]]

def _evaluate_engine(
    engine: Literal["stylometry", "semantic", "chronobiology", "combined"],
    validation: Sequence[tuple[float, bool]],
    test: Sequence[tuple[float, bool]],
    test_raw: Sequence[tuple[float, bool]],
    excluded: int,
    config: EvaluationConfig,
) -> EngineEvaluation:
    threshold, constraint = _threshold(validation, config.min_precision)
    precision, recall = _metrics(test, threshold)
    same = sum(label for _, label in test)
    different = len(test) - same
    return EngineEvaluation(
        engine=engine,
        roc_auc=_auc(test_raw),
        pr_auc=_pr_auc(test_raw),
        test_base_rate=(same / len(test) if test else 0.0),
        threshold_percentile=threshold,
        precision_constraint_met=constraint,
        precision=precision,
        recall=recall,
        support_same=same,
        support_different=different,
        excluded_insufficient_data=excluded,
        ci={
            "roc_auc": _bootstrap_metric_ci(
                test_raw, metric="roc_auc", threshold=threshold, config=config, seed_offset=0
            ),
            "precision": _bootstrap_metric_ci(
                test, metric="precision", threshold=threshold, config=config, seed_offset=1
            ),
            "recall": _bootstrap_metric_ci(
                test, metric="recall", threshold=threshold, config=config, seed_offset=2
            ),
        },
    )


def evaluate_labeled_pairs(
    pairs: Sequence[LabeledPair],
    profiles: Mapping[str, AccountProfile],
    *,
    assets: ComparisonAssets,
    config: EvaluationConfig,
    seed: int,
    output_path: Path,
) -> EvaluationReport:
    """Evaluate held-out pairs using account-disjoint training-only references."""
    if not isinstance(assets, ComparisonAssets) or not isinstance(config, EvaluationConfig):
        raise ContractValidationError("input", ErrorCode.INPUT_INVALID)
    if not isinstance(output_path, Path) or type(seed) is not int:
        raise ContractValidationError("input", ErrorCode.INPUT_INVALID)
    assets._validate()
    config._validate()
    for pair in pairs:
        if not isinstance(pair, LabeledPair):
            raise ContractValidationError("pair", ErrorCode.INPUT_INVALID)
        pair._validate()

    assignments = _partition_accounts(pairs, seed, config.split_fractions)
    partitions: dict[str, list[LabeledPair]] = {"train": [], "validation": [], "test": []}
    for pair in pairs:
        left_part = assignments[pair.account_id_a]
        right_part = assignments[pair.account_id_b]
        if left_part == right_part:
            partitions[left_part].append(pair)

    train_accounts = sorted(
        {pair.account_id_a for pair in partitions["train"]}
        | {pair.account_id_b for pair in partitions["train"]}
    )
    training_profiles = [
        profiles[account] for account in train_accounts
        if account in profiles and profiles[account].status == "READY"
    ]
    if len(training_profiles) < 2:
        raise ContractValidationError("training_accounts", ErrorCode.MINIMUMS_NOT_MET)
    for profile in training_profiles:
        profile._validate()
    min_messages = max(profile.minimum_messages for profile in training_profiles)
    min_chars = max(profile.minimum_chars for profile in training_profiles)
    training_profiles = [
        profile for profile in training_profiles
        if profile.n_messages >= min_messages and profile.n_chars >= min_chars
    ]
    if len(training_profiles) < 2:
        raise ContractValidationError("training_accounts", ErrorCode.MINIMUMS_NOT_MET)
    preprocessing_fps = {profile.preprocessing_fingerprint for profile in training_profiles}
    if len(preprocessing_fps) != 1:
        raise ContractValidationError("preprocessing_fingerprint", ErrorCode.BASELINE_CONFIG_MISMATCH)
    preprocessing_fp = next(iter(preprocessing_fps))

    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        raise ContractValidationError("output_path", ErrorCode.ARTIFACT_INVALID) from None
    words = load_term_list(assets.function_words_path, require_lang=True)
    stylo_config = StylometryConfig(minimum_messages=min_messages, minimum_chars=min_chars)
    chrono_config = ChronobiologyConfig(minimum_messages=min_messages, minimum_chars=min_chars)
    model_config = SemanticConfig(
        model_dir=assets.semantic_model_dir,
        cache_dir=assets.cache_dir,
        center_embeddings=False,
        minimum_messages=min_messages,
        minimum_chars=min_chars,
    )
    artifact_dir = output_path.parent / "evaluation-artifacts"
    try:
        stylo_artifact = fit_stylometry_artifact(
            [load_raw_stream(profile) for profile in training_profiles],
            function_words=words.entries,
            config=stylo_config,
            seed=seed,
            output_dir=artifact_dir,
        )
    except Exception:
        stylo_artifact = None
    try:
        chrono_artifact = fit_chronobiology_artifact(
            [load_temporal_stream(profile) for profile in training_profiles],
            config=chrono_config,
            preprocessing_fp=preprocessing_fp,
            output_dir=artifact_dir,
        )
    except Exception:
        chrono_artifact = None
    try:
        model = load_local_model(model_config)
    except Exception:
        model = None

    engine_names = ("stylometry", "semantic", "chronobiology")

    def raw_score(engine: str, left: AccountProfile, right: AccountProfile) -> float:
        if engine == "stylometry":
            if stylo_artifact is None:
                raise ContractValidationError("training_baselines", ErrorCode.MINIMUMS_NOT_MET)
            return compute_stylometry_raw(
                load_raw_stream(left), load_raw_stream(right),
                artifact=stylo_artifact, config=stylo_config,
            )
        if engine == "semantic":
            if model is None:
                raise ContractValidationError("training_baselines", ErrorCode.MINIMUMS_NOT_MET)
            return compute_semantic_raw(
                load_normalized_stream(left), load_normalized_stream(right),
                model=model, config=model_config,
            )
        if chrono_artifact is None:
            raise ContractValidationError("training_baselines", ErrorCode.MINIMUMS_NOT_MET)
        return compute_chronobiology_raw(
            load_temporal_stream(left), load_temporal_stream(right),
            artifact=chrono_artifact, config=chrono_config,
        )

    train_scores: dict[str, list[float]] = {name: [] for name in engine_names}
    for pair in partitions["train"]:
        if pair.label != "different":
            continue
        left = profiles.get(pair.account_id_a)
        right = profiles.get(pair.account_id_b)
        if left is None or right is None or any(
            profile.status != "READY"
            or profile.n_messages < min_messages
            or profile.n_chars < min_chars
            or profile.preprocessing_fingerprint != preprocessing_fp
            for profile in (left, right)
        ):
            continue
        for engine in engine_names:
            try:
                value = float(raw_score(engine, left, right))
                if math.isfinite(value) and -1.0 <= value <= 1.0:
                    train_scores[engine].append(value)
            except Exception:
                continue
    if any(not train_scores[engine] for engine in engine_names):
        raise ContractValidationError("training_baselines", ErrorCode.MINIMUMS_NOT_MET)
    if stylo_artifact is None or chrono_artifact is None or model is None:
        raise ContractValidationError("training_baselines", ErrorCode.MINIMUMS_NOT_MET)

    fingerprints = {
        "stylometry": stylo_artifact.configuration_fingerprint,
        "semantic": semantic_fingerprint(
            model_config,
            model.model_fingerprint,
            getattr(model, "tokenizer_sha256", model.model_fingerprint),
            preprocessing_fp,
        ),
        "chronobiology": chrono_artifact.configuration_fingerprint,
    }
    baselines: dict[str, BaselineDistribution] = {}
    for engine in engine_names:
        values = sorted(train_scores[engine])
        baselines[engine] = BaselineDistribution(
            schema_version="1.0",
            baseline_id=f"evaluation-{engine}",
            engine=engine,
            created_at_utc="2026-10-01T00:00:00Z",
            source_snapshot_id="evaluation-training",
            configuration_fingerprint=fingerprints[engine],
            random_seed=seed,
            calibration_population="known_different_account_pairs",
            n_pairs=len(values),
            sorted_raw_cosines=values,
            minimum_messages=min_messages,
            minimum_chars=min_chars,
            strata_counts={},
            centering_vector_path=None,
        )

    names = (*engine_names, "combined")
    rows: dict[str, dict[str, list[tuple[float, bool]]]] = {
        name: {"validation": [], "test": [], "test_raw": []} for name in names
    }
    exclusions = {name: 0 for name in names}

    def failure(
        engine: str,
        left: AccountProfile | None,
        right: AccountProfile | None,
        status: str,
        code: ErrorCode,
    ) -> EngineResult:
        return EngineResult(
            engine=engine,
            raw_cosine=None,
            calibrated_percentile=None,
            status=status,
            n_messages={
                "candidate": 0 if left is None else left.n_messages,
                "suspect": 0 if right is None else right.n_messages,
            },
            n_chars={
                "candidate": 0 if left is None else left.n_chars,
                "suspect": 0 if right is None else right.n_chars,
            },
            error=code.value,
        )

    for partition in ("validation", "test"):
        for pair in partitions[partition]:
            left = profiles.get(pair.account_id_a)
            right = profiles.get(pair.account_id_b)
            outcomes: dict[str, EngineResult] = {}
            for engine in engine_names:
                if left is None or right is None or any(
                    profile.status != "READY"
                    or profile.n_messages < min_messages
                    or profile.n_chars < min_chars
                    for profile in (left, right)
                ):
                    result = failure(engine, left, right, "INSUFFICIENT_DATA", ErrorCode.MINIMUMS_NOT_MET)
                elif any(
                    profile.preprocessing_fingerprint != preprocessing_fp
                    for profile in (left, right)
                ):
                    result = failure(engine, left, right, "ERROR", ErrorCode.BASELINE_CONFIG_MISMATCH)
                else:
                    try:
                        raw = float(raw_score(engine, left, right))
                        percentile = calibrate_cosine(raw, baselines[engine])
                        result = EngineResult(
                            engine=engine,
                            raw_cosine=raw,
                            calibrated_percentile=percentile,
                            status="OK",
                            n_messages={"candidate": left.n_messages, "suspect": right.n_messages},
                            n_chars={"candidate": left.n_chars, "suspect": right.n_chars},
                            error=None,
                        )
                    except Exception as exc:
                        result = failure(engine, left, right, "ERROR", error_code_for(exc))
                outcomes[engine] = result
                if result.status == "OK":
                    rows[engine][partition].append((result.calibrated_percentile, pair.label == "same"))
                    if partition == "test":
                        rows[engine]["test_raw"].append((result.raw_cosine, pair.label == "same"))
                else:
                    exclusions[engine] += 1
            combined = compute_combined_result(
                outcomes["stylometry"], outcomes["semantic"], outcomes["chronobiology"]
            )
            if combined.status == "OK":
                rows["combined"][partition].append((combined.combined_score, pair.label == "same"))
                if partition == "test":
                    rows["combined"]["test_raw"].append((combined.combined_score, pair.label == "same"))
            else:
                exclusions["combined"] += 1

    report = EvaluationReport(
        schema_version="1.0",
        seed=seed,
        partition_sizes={name: len(values) for name, values in partitions.items()},
        stylometry=_evaluate_engine(
            "stylometry", rows["stylometry"]["validation"], rows["stylometry"]["test"],
            rows["stylometry"]["test_raw"], exclusions["stylometry"], config,
        ),
        semantic=_evaluate_engine(
            "semantic", rows["semantic"]["validation"], rows["semantic"]["test"],
            rows["semantic"]["test_raw"], exclusions["semantic"], config,
        ),
        chronobiology=_evaluate_engine(
            "chronobiology", rows["chronobiology"]["validation"], rows["chronobiology"]["test"],
            rows["chronobiology"]["test_raw"], exclusions["chronobiology"], config,
        ),
        combined=_evaluate_engine(
            "combined", rows["combined"]["validation"], rows["combined"]["test"],
            rows["combined"]["test_raw"], exclusions["combined"], config,
        ),
        note=EvaluationReport.NOTE,
    )
    write_json_artifact(report, output_path)
    return report

__all__ = ["evaluate_labeled_pairs"]
