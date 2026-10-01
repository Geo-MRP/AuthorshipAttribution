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
    ComparisonAssets,
    EngineEvaluation,
    EvaluationConfig,
    EvaluationReport,
    LabeledPair,
    SemanticConfig,
    StylometryArtifact,
    StylometryConfig,
)
from .calibration import calibrate_cosine
from .errors import ContractValidationError, ErrorCode
from .fileformats import load_term_list, write_json_artifact
from .semantic import compute_semantic_raw, load_local_model
from .streams import load_normalized_stream, load_raw_stream
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


def _threshold(rows: Sequence[tuple[float, bool]], minimum_precision: float) -> tuple[float, bool]:
    candidates = sorted({score for score, _ in rows} | {math.nextafter(max((s for s, _ in rows), default=0.0), math.inf)})
    scored = [(threshold, *_metrics(rows, threshold)) for threshold in candidates]
    qualifying = [row for row in scored if row[1] >= minimum_precision]
    pool = qualifying if qualifying else scored
    best = max(pool, key=lambda row: (row[2], row[1], -row[0])) if pool else (100.0, 0.0, 0.0)
    # Candidate scores are calibrated percentiles; raw score thresholds are
    # selected in the same domain as those empirical calibration values.
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


def _evaluate_engine(
    engine: Literal["stylometry", "semantic"],
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
    auc = _auc(test_raw)
    pr = _pr_auc(test_raw)
    return EngineEvaluation(
        engine=engine, roc_auc=auc, pr_auc=pr,
        test_base_rate=(same / len(test) if test else 0.0),
        threshold_percentile=threshold, precision_constraint_met=constraint,
        precision=precision, recall=recall, support_same=same,
        support_different=different, excluded_insufficient_data=excluded,
        ci={"roc_auc": [0.0 if auc is None else auc, 0.0 if auc is None else auc],
            "precision": _ci([precision], config.seed, config.ci_level, config.n_bootstrap),
            "recall": _ci([recall], config.seed + 1, config.ci_level, config.n_bootstrap)},
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
    """Evaluate held-out pairs after same-label connected-component splitting."""
    assignments = _partition_accounts(pairs, seed, config.split_fractions)
    partitions: dict[str, list[LabeledPair]] = {"train": [], "validation": [], "test": []}
    for pair in pairs:
        left_part, right_part = assignments[pair.account_id_a], assignments[pair.account_id_b]
        if left_part == right_part:
            partitions[left_part].append(pair)
    # Train artifacts are fit from train-partition account profiles, not from
    # validation/test records; cross-partition pairs are omitted above.
    train_accounts = sorted({p.account_id_a for p in partitions["train"]} | {p.account_id_b for p in partitions["train"]})
    training_profiles = [profiles[account] for account in train_accounts if account in profiles and profiles[account].status == "READY"]
    function_words = load_term_list(assets.function_words_path, require_lang=True)
    if len(training_profiles) < 2:
        raise ContractValidationError("training_accounts", ErrorCode.MINIMUMS_NOT_MET)
    min_messages = max(profile.minimum_messages for profile in training_profiles)
    min_chars = max(profile.minimum_chars for profile in training_profiles)
    stylo_config = StylometryConfig(minimum_messages=min_messages, minimum_chars=min_chars)
    artifact = fit_stylometry_artifact(
        [load_raw_stream(profile) for profile in training_profiles],
        function_words=function_words.entries, config=stylo_config, seed=seed,
        output_dir=output_path.parent / "evaluation-artifacts",
    )
    model_config = SemanticConfig(
        model_dir=assets.semantic_model_dir, cache_dir=assets.cache_dir,
        center_embeddings=False, minimum_messages=min_messages, minimum_chars=min_chars,
    )
    model = load_local_model(model_config)
    raw_by_engine: dict[str, dict[LabeledPair, float]] = {"stylometry": {}, "semantic": {}}
    validation_raw: dict[str, dict[LabeledPair, float]] = {"stylometry": {}, "semantic": {}}
    exclusions = {"stylometry": 0, "semantic": 0}
    for part in ("train", "validation", "test"):
        for pair in partitions[part]:
            left, right = profiles.get(pair.account_id_a), profiles.get(pair.account_id_b)
            if left is None or right is None or any(
                p.status != "READY" or p.n_messages < min_messages or p.n_chars < min_chars for p in (left, right)
            ):
                exclusions["stylometry"] += int(part != "train")
                exclusions["semantic"] += int(part != "train")
                continue
            if part == "train":
                # Training labels establish the artifact only.
                continue
            destination = validation_raw if part == "validation" else raw_by_engine
            destination["stylometry"][pair] = compute_stylometry_raw(
                load_raw_stream(left), load_raw_stream(right), artifact=artifact, config=stylo_config
            )
            destination["semantic"][pair] = compute_semantic_raw(
                load_normalized_stream(left), load_normalized_stream(right), model=model, config=model_config
            )
    # Training different-pair scores alone define the evaluation calibration references.
    train_scores: dict[str, list[float]] = {"stylometry": [], "semantic": []}
    for pair in (item for item in partitions["train"] if item.label == "different"):
        left, right = profiles.get(pair.account_id_a), profiles.get(pair.account_id_b)
        if left is None or right is None or left.status != "READY" or right.status != "READY":
            continue
        train_scores["stylometry"].append(compute_stylometry_raw(
            load_raw_stream(left), load_raw_stream(right), artifact=artifact, config=stylo_config
        ))
        train_scores["semantic"].append(compute_semantic_raw(
            load_normalized_stream(left), load_normalized_stream(right), model=model, config=model_config
        ))
    baselines: dict[str, BaselineDistribution] = {}
    for engine in ("stylometry", "semantic"):
        values = sorted(train_scores[engine])
        if not values:
            continue
        baselines[engine] = BaselineDistribution(
            schema_version="1.0", baseline_id=f"evaluation-{engine}", engine=engine,
            created_at_utc="2026-10-01T00:00:00Z", source_snapshot_id="evaluation-training",
            configuration_fingerprint="0" * 64, random_seed=seed,
            calibration_population="known_different_account_pairs", n_pairs=len(values),
            sorted_raw_cosines=values, minimum_messages=min_messages,
            minimum_chars=min_chars, strata_counts={}, centering_vector_path=None,
        )
    evaluation_rows: dict[str, dict[str, list[tuple[float, bool]]]] = {
        engine: {"validation": [], "test": []} for engine in ("stylometry", "semantic")
    }
    for engine in evaluation_rows:
        for part in ("validation", "test"):
            for pair in partitions[part]:
                raw = (validation_raw if part == "validation" else raw_by_engine)[engine].get(pair)
                if raw is not None:
                    calibrated = calibrate_cosine(raw, baselines[engine]) if engine in baselines else 0.0
                    evaluation_rows[engine][part].append((calibrated, pair.label == "same"))
    report = EvaluationReport(
        schema_version="1.0", seed=seed,
        partition_sizes={name: len(values) for name, values in partitions.items()},
        stylometry=_evaluate_engine(
            "stylometry", evaluation_rows["stylometry"]["validation"],
            evaluation_rows["stylometry"]["test"],
            [(raw_by_engine["stylometry"][pair], pair.label == "same")
             for pair in partitions["test"] if pair in raw_by_engine["stylometry"]],
            exclusions["stylometry"], config,
        ),
        semantic=_evaluate_engine(
            "semantic", evaluation_rows["semantic"]["validation"],
            evaluation_rows["semantic"]["test"],
            [(raw_by_engine["semantic"][pair], pair.label == "same")
             for pair in partitions["test"] if pair in raw_by_engine["semantic"]],
            exclusions["semantic"], config,
        ),
        note=EvaluationReport.NOTE,
    )
    write_json_artifact(report, output_path)
    return report


__all__ = ["evaluate_labeled_pairs"]
