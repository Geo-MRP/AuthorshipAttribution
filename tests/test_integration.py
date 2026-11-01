from __future__ import annotations

from concurrent.futures import TimeoutError
from datetime import UTC, datetime
import itertools
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import pytest

import authorship_attribution.baselines as baseline_module
import authorship_attribution.cli as cli_module
import authorship_attribution.evaluate as evaluation_module
import authorship_attribution.orchestrator as orchestrator_module
from authorship_attribution.baselines import baseline_needs_refresh, build_baselines
from authorship_attribution.calibration import calibrate_cosine, compute_combined_result
from authorship_attribution.chronobiology import (
    compute_chronobiology_raw,
    fit_chronobiology_artifact,
)
from authorship_attribution.contracts import (
    AccountProfile,
    BaselineConfig,
    BaselineDistribution,
    ChronobiologyConfig,
    ComparisonAssets,
    ComparisonReport,
    EngineResult,
    EvaluationConfig,
    LabeledPair,
    Message,
    PreprocessConfig,
    RunConfig,
    StylometryConfig,
)
from authorship_attribution.errors import ContractValidationError, ErrorCode
from authorship_attribution.evaluate import _components, _partition_accounts, evaluate_labeled_pairs
from authorship_attribution.fileformats import write_json_artifact
from authorship_attribution.preprocess import build_account_profiles
from authorship_attribution.reporting import render_compare_report, render_evaluation_report
from authorship_attribution.streams import load_raw_stream, load_temporal_stream
from authorship_attribution.stylometry import compute_stylometry_raw, fit_stylometry_artifact


def _result(
    engine: str,
    *,
    status: str = "OK",
    raw: float | None = 0.4,
    percentile: float | None = 50.0,
    error: str | None = None,
) -> EngineResult:
    return EngineResult(
        engine=engine,
        raw_cosine=raw,
        calibrated_percentile=percentile,
        status=status,
        n_messages={"candidate": 210, "suspect": 220},
        n_chars={"candidate": 5100, "suspect": 5300},
        error=error,
    )


def _baseline(
    engine: str = "semantic",
    fingerprint: str = "a" * 64,
    values: list[float] | None = None,
    *,
    minimum_messages: int = 200,
    minimum_chars: int = 5000,
) -> BaselineDistribution:
    scores = [-0.5, 0.1, 0.1, 0.9] if values is None else sorted(values)
    return BaselineDistribution(
        schema_version="1.0",
        baseline_id="b-id",
        engine=engine,
        created_at_utc="2026-10-01T00:00:00Z",
        source_snapshot_id="snapshot",
        configuration_fingerprint=fingerprint,
        random_seed=1729,
        calibration_population="known_different_account_pairs",
        n_pairs=len(scores),
        sorted_raw_cosines=scores,
        minimum_messages=minimum_messages,
        minimum_chars=minimum_chars,
        strata_counts={"same-game|same-channel|same-month": len(scores)},
        centering_vector_path=None,
    )


def _pair(left: str, right: str, label: str) -> LabeledPair:
    return LabeledPair("1.0", left, right, label, "snapshot")


def _message(account: str, hour: int, text: str) -> Message:
    return Message(
        schema_version="1.0",
        message_id="message-" + account,
        account_id=account,
        game_id="game",
        channel_id="channel",
        timestamp_utc=f"2026-10-01T{hour:02d}:00:00Z",
        text=text,
        is_system=False,
        source_id="synthetic",
        source_row=1,
    )


def _make_profiles(tmp_path: Path, names: list[str], hours: list[int]) -> dict[str, AccountProfile]:
    return dict(
        build_account_profiles(
            [_message(name, hour, "SYNTHETIC_PRIVATE_TEXT_2981 " + name) for name, hour in zip(names, hours, strict=True)],
            game_terms=[],
            usernames=[],
            config=PreprocessConfig(minimum_messages=1, minimum_chars=1),
            output_dir=tmp_path / "work",
        )
    )


def _assets(
    tmp_path: Path,
    *,
    stylometry_artifact: Path | None = None,
    chronobiology_artifact: Path | None = None,
    baseline_paths: dict[str, Path] | None = None,
) -> ComparisonAssets:
    words = tmp_path / "function_words.txt"
    words.write_text("# lang: en\nand\n", encoding="utf-8")
    return ComparisonAssets(
        stylometry_artifact_path=stylometry_artifact or tmp_path / "unused-stylometry.json",
        semantic_model_dir=tmp_path / "missing-local-model",
        baseline_paths=baseline_paths or {
            "stylometry": tmp_path / "unused-stylometry-baseline.json",
            "semantic": tmp_path / "unused-semantic-baseline.json",
            "chronobiology": tmp_path / "unused-chronobiology-baseline.json",
        },
        game_terms_path=tmp_path / "games.txt",
        usernames_path=tmp_path / "users.txt",
        function_words_path=words,
        cache_dir=tmp_path / "cache",
        chronobiology_artifact_path=chronobiology_artifact or tmp_path / "unused-chrono.json",
    )


class _SyntheticModel:
    model_fingerprint = "e" * 64
    tokenizer_sha256 = "f" * 64
    embedding_dim = 3

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(
            [[float(len(text) + 1), float(sum(map(ord, text)) % 97 + 1), 1.0] for text in texts],
            dtype=np.float32,
        )


@pytest.fixture
def prepared(tmp_path: Path) -> tuple[dict[str, AccountProfile], object, object, ComparisonAssets]:
    profiles = _make_profiles(tmp_path, ["acct-a", "acct-b"], [0, 12])
    first, second = profiles["acct-a"], profiles["acct-b"]
    artifact_dir = tmp_path / "artifacts"
    stylo = fit_stylometry_artifact(
        [load_raw_stream(first), load_raw_stream(second)],
        function_words=["and"],
        config=StylometryConfig(minimum_messages=1, minimum_chars=1),
        seed=3,
        output_dir=artifact_dir,
    )
    chrono = fit_chronobiology_artifact(
        [load_temporal_stream(first), load_temporal_stream(second)],
        config=ChronobiologyConfig(minimum_messages=1, minimum_chars=1),
        preprocessing_fp=first.preprocessing_fingerprint,
        output_dir=artifact_dir,
    )
    stylo_meta = artifact_dir / "stylometry-metadata.json"
    chrono_meta = artifact_dir / "chronobiology-metadata.json"
    write_json_artifact(stylo, stylo_meta)
    write_json_artifact(chrono, chrono_meta)
    stylo_raw = compute_stylometry_raw(
        load_raw_stream(first), load_raw_stream(second),
        artifact=stylo, config=StylometryConfig(minimum_messages=1, minimum_chars=1),
    )
    chrono_raw = compute_chronobiology_raw(
        load_temporal_stream(first), load_temporal_stream(second),
        artifact=chrono, config=ChronobiologyConfig(minimum_messages=1, minimum_chars=1),
    )
    baselines = {
        "stylometry": _baseline("stylometry", stylo.configuration_fingerprint, [stylo_raw], minimum_messages=1, minimum_chars=1),
        "semantic": _baseline("semantic", "0" * 64, [0.0], minimum_messages=1, minimum_chars=1),
        "chronobiology": _baseline("chronobiology", chrono.configuration_fingerprint, [chrono_raw], minimum_messages=1, minimum_chars=1),
    }
    baseline_paths = {name: tmp_path / f"{name}-baseline.json" for name in baselines}
    for name, baseline in baselines.items():
        write_json_artifact(baseline, baseline_paths[name])
    assets = _assets(
        tmp_path,
        stylometry_artifact=stylo_meta,
        chronobiology_artifact=chrono_meta,
        baseline_paths=baseline_paths,
    )
    return profiles, stylo, chrono, assets


def test_report_has_banner_three_engines_and_heuristic_combination() -> None:
    stylo = _result("stylometry")
    semantic = _result("semantic")
    chrono = _result("chronobiology")
    report = ComparisonReport(
        schema_version="1.0",
        flagged_account_id="flagged",
        suspect_account_id="suspect",
        stylometry=stylo,
        semantic=semantic,
        chronobiology=chrono,
        combined=compute_combined_result(stylo, semantic, chrono),
        banner=ComparisonReport.BANNER,
    )
    rendered = render_compare_report(report, output_format="table")
    assert rendered.splitlines()[0] == ComparisonReport.BANNER
    assert "Semantic" in rendered and "Stylometry" in rendered and "Chronobiology" in rendered
    assert "Known-different baseline percentile" in rendered
    assert "Combined score (heuristic)" in rendered
    assert "not a probability" in rendered
    table_lines = [line for line in rendered.splitlines() if line.startswith(("+", "|"))]
    assert len({len(line) for line in table_lines}) == 1
    encoded = render_compare_report(report, output_format="json")
    assert encoded.splitlines()[0] == ComparisonReport.BANNER
    assert "combined_score" in encoded
    restored = ComparisonReport.from_dict(json.loads(encoded.splitlines()[1]))
    assert restored.to_dict() == report.to_dict()
    assert "SYNTHETIC_PRIVATE_RAW_CHAT_7319" not in encoded


@pytest.mark.parametrize(("score", "expected"), [(-0.8, 0.0), (0.1, 75.0), (1.0, 100.0)])
def test_contract_calibration_is_inclusive_for_ties_and_boundaries(score: float, expected: float) -> None:
    assert calibrate_cosine(score, _baseline()) == expected


def test_same_label_components_are_kept_in_one_partition() -> None:
    pairs = [
        _pair("a", "b", "same"), _pair("b", "c", "same"),
        _pair("d", "e", "same"), _pair("f", "g", "different"),
    ]
    groups = _components(pairs)
    assert {"a", "b", "c"} in groups
    assert {"d", "e"} in groups
    partitions = _partition_accounts(pairs, 1729, (0.6, 0.2, 0.2))
    assert partitions["a"] == partitions["b"] == partitions["c"]
    assert partitions["d"] == partitions["e"]


def test_refresh_detects_month_pair_growth_and_fingerprint() -> None:
    baseline = _baseline()
    config = BaselineConfig()
    assert baseline_needs_refresh(baseline, current_fingerprint="b" * 64, new_eligible_pairs=0, config=config)
    assert baseline_needs_refresh(baseline, current_fingerprint="a" * 64, new_eligible_pairs=1, config=config)
    assert baseline_needs_refresh(
        baseline, current_fingerprint="a" * 64, new_eligible_pairs=0, config=config,
        now=datetime(2026, 11, 5, tzinfo=UTC),
    )


def test_comparison_keeps_two_successes_after_one_timeout(
    prepared: tuple[dict[str, AccountProfile], object, object, ComparisonAssets],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profiles, _, _, assets = prepared

    def fake_run(engine: str, candidate: AccountProfile, suspect: AccountProfile, assets: object, config: object, context: object) -> tuple[EngineResult, None, None]:
        if engine == "chronobiology":
            return _result(engine, status="TIMEOUT", raw=None, percentile=None, error="ENGINE_TIMEOUT"), None, None
        return _result(engine, percentile=80.0 if engine == "stylometry" else 20.0), None, None

    monkeypatch.setattr(orchestrator_module, "_run_engine", fake_run)
    report = orchestrator_module.compare_shortlist(
        profiles["acct-a"], [profiles["acct-b"]], assets=assets, config=RunConfig()
    )[0]
    assert report.chronobiology.status == "TIMEOUT"
    assert report.stylometry.status == report.semantic.status == "OK"
    assert report.combined.active_engines == ["stylometry", "semantic"]
    assert report.combined.weighted_percentile == pytest.approx(52.0)
    assert report.combined.fisher_tail_score == pytest.approx(61.79428782333969)
    assert report.combined.combined_score == pytest.approx(56.89714391166984)


def test_real_spawned_comparison_keeps_healthy_engines_when_model_missing(
    prepared: tuple[dict[str, AccountProfile], object, object, ComparisonAssets],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profiles, _, _, assets = prepared
    monkeypatch.setenv("AUTH_ATTR_MINIMUM_MESSAGES", "1")
    monkeypatch.setenv("AUTH_ATTR_MINIMUM_CHARS", "1")
    monkeypatch.setenv("AUTH_ATTR_DEVICE", "cpu")
    monkeypatch.setenv("AUTH_ATTR_CACHE_RETENTION_DAYS", "7")
    report = orchestrator_module.compare_shortlist(
        profiles["acct-a"], [profiles["acct-b"]], assets=assets, config=RunConfig()
    )[0]
    assert report.stylometry.status == "OK"
    assert report.chronobiology.status == "OK"
    assert report.semantic.status == "ERROR"
    assert report.semantic.error == "LOCAL_MODEL_UNAVAILABLE"
    assert report.combined.active_engines == ["stylometry", "chronobiology"]
    assert report.combined.weights_used == pytest.approx({"stylometry": 0.4 / 0.65, "chronobiology": 0.25 / 0.65})
    encoded = render_compare_report(report, output_format="json")
    assert "SYNTHETIC_PRIVATE_TEXT_2981" not in encoded


def test_timed_out_worker_is_terminated_and_manager_closed(
    prepared: tuple[dict[str, AccountProfile], object, object, ComparisonAssets],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profiles, stylo, _, assets = prepared
    baseline = _baseline("stylometry", stylo.configuration_fingerprint, [0.0], minimum_messages=1, minimum_chars=1)

    class Process:
        terminated = False
        def is_alive(self) -> bool:
            return not self.terminated
        def terminate(self) -> None:
            self.terminated = True
        def join(self, timeout: float) -> None:
            return None

    class Manager:
        token = ""
        closed = False
        def Queue(self) -> Manager:
            return self
        def get(self, timeout: float) -> tuple[str, float]:
            return self.token, time.monotonic()
        def shutdown(self) -> None:
            self.closed = True

    class Future:
        def result(self, timeout: float) -> object:
            raise TimeoutError()
        def done(self) -> bool:
            return False

    process = Process()
    manager = Manager()

    class Executor:
        _processes = {1: process}
        closed = False
        def submit(self, function: object, engine: str, token: str, *args: object) -> Future:
            manager.token = token
            return Future()
        def shutdown(self, *, wait: bool, cancel_futures: bool) -> None:
            self.closed = True

    executor = Executor()

    class Context:
        def Manager(self) -> Manager:
            return manager

    monkeypatch.setenv("AUTH_ATTR_MINIMUM_MESSAGES", "1")
    monkeypatch.setenv("AUTH_ATTR_MINIMUM_CHARS", "1")
    monkeypatch.setattr(orchestrator_module, "_read_baseline", lambda path: baseline)
    monkeypatch.setattr(orchestrator_module, "_expected_fingerprint", lambda *args: stylo.configuration_fingerprint)
    monkeypatch.setattr(orchestrator_module, "load_raw_stream", lambda profile: [])
    monkeypatch.setattr(orchestrator_module, "ProcessPoolExecutor", lambda **kwargs: executor)
    result, returned_executor, returned_manager = orchestrator_module._run_engine(
        "stylometry", profiles["acct-a"], profiles["acct-b"], assets, RunConfig(), Context()
    )
    assert result.status == "TIMEOUT" and result.error == "ENGINE_TIMEOUT"
    assert returned_executor is None and returned_manager is None
    assert process.terminated and executor.closed and manager.closed


def test_builds_three_immutable_1000_pair_baselines(
    tmp_path: Path,
    prepared: tuple[dict[str, AccountProfile], object, object, ComparisonAssets],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profiles, stylo, chrono, _ = prepared
    exemplar = profiles["acct-a"]
    from dataclasses import replace

    corpus = {
        f"p{index:02d}": replace(exemplar, account_id=f"p{index:02d}")
        for index in range(46)
    }
    names = sorted(corpus)
    pairs = [
        _pair(left, right, "different")
        for left, right in itertools.islice(itertools.combinations(names, 2), 1000)
    ]
    monkeypatch.setattr(baseline_module, "load_raw_stream", lambda profile: [])
    monkeypatch.setattr(baseline_module, "load_normalized_stream", lambda profile: [])
    monkeypatch.setattr(baseline_module, "load_temporal_stream", lambda profile: [])
    monkeypatch.setattr(baseline_module, "compute_stylometry_raw", lambda *args, **kwargs: 0.1)
    monkeypatch.setattr(baseline_module, "compute_semantic_raw", lambda *args, **kwargs: 0.2)
    monkeypatch.setattr(baseline_module, "compute_chronobiology_raw", lambda *args, **kwargs: 0.3)
    output = tmp_path / "baselines"
    kwargs = dict(
        stylometry_artifact=stylo,
        semantic_model=_SyntheticModel(),
        chronobiology_artifact=chrono,
        config=BaselineConfig(max_pairs=1000),
        seed=1729,
        output_dir=output,
    )
    built = build_baselines(pairs, corpus, **kwargs)
    assert set(built) == {"stylometry", "semantic", "chronobiology"}
    assert all(item.n_pairs == 1000 for item in built.values())
    assert built["chronobiology"].configuration_fingerprint == chrono.configuration_fingerprint
    assert built["chronobiology"].sorted_raw_cosines == [0.3] * 1000
    before = {path.name: path.read_bytes() for path in output.glob("*.json")}
    assert len(before) == 3
    repeated = build_baselines(pairs, corpus, **kwargs)
    assert {path.name: path.read_bytes() for path in output.glob("*.json")} == before
    assert {name: item.to_dict() for name, item in repeated.items()} == {
        name: item.to_dict() for name, item in built.items()
    }
    with pytest.raises(ContractValidationError) as caught:
        build_baselines(pairs[:999], corpus, **kwargs)
    assert caught.value.code is ErrorCode.MINIMUMS_NOT_MET


def _evaluation_data(tmp_path: Path) -> tuple[list[LabeledPair], dict[str, AccountProfile], ComparisonAssets, dict[str, str]]:
    names = ["ta", "tb", "va", "vb", "xa", "xb", "xc", "xd"]
    profiles = _make_profiles(tmp_path, names, [0, 12, 3, 9, 2, 8, 4, 10])
    pairs = [
        _pair("ta", "tb", "different"),
        _pair("va", "vb", "same"),
        _pair("xa", "xb", "same"),
        _pair("xc", "xd", "different"),
    ]
    assignments = {
        "ta": "train", "tb": "train", "va": "validation", "vb": "validation",
        "xa": "test", "xb": "test", "xc": "test", "xd": "test",
    }
    return pairs, profiles, _assets(tmp_path), assignments


def test_evaluation_fits_training_accounts_only_and_emits_combined_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pairs, profiles, assets, assignments = _evaluation_data(tmp_path)
    monkeypatch.setattr(evaluation_module, "_partition_accounts", lambda pairs, seed, fractions: assignments)
    monkeypatch.setattr(evaluation_module, "load_local_model", lambda config: _SyntheticModel())
    captured: dict[str, set[str]] = {}
    original_stylo = evaluation_module.fit_stylometry_artifact
    original_chrono = evaluation_module.fit_chronobiology_artifact

    def fit_stylo(streams: object, **kwargs: object) -> object:
        streams = list(streams)
        captured["stylometry"] = {stream[0].account_id for stream in streams}
        return original_stylo(streams, **kwargs)

    def fit_chrono(streams: object, **kwargs: object) -> object:
        streams = list(streams)
        captured["chronobiology"] = {stream[0].account_id for stream in streams}
        return original_chrono(streams, **kwargs)

    monkeypatch.setattr(evaluation_module, "fit_stylometry_artifact", fit_stylo)
    monkeypatch.setattr(evaluation_module, "fit_chronobiology_artifact", fit_chrono)
    config = EvaluationConfig(n_bootstrap=50)
    report = evaluate_labeled_pairs(
        pairs, profiles, assets=assets, config=config, seed=1729, output_path=tmp_path / "evaluation.json"
    )
    assert captured == {"stylometry": {"ta", "tb"}, "chronobiology": {"ta", "tb"}}
    assert report.partition_sizes == {"train": 1, "validation": 1, "test": 2}
    assert report.chronobiology.engine == "chronobiology"
    assert report.combined.engine == "combined"
    assert report.combined.support_same == report.combined.support_different == 1
    assert report.combined.roc_auc is not None
    assert report.combined.pr_auc is not None
    assert (tmp_path / "evaluation.json").exists()
    assert "SYNTHETIC_PRIVATE_TEXT_2981" not in (tmp_path / "evaluation.json").read_text(encoding="utf-8")
    table = render_evaluation_report(report, output_format="table")
    assert table.splitlines()[0] == report.NOTE
    assert "chronobiology" in table and "combined" in table and "not a probability" in table
    assert json.loads(render_evaluation_report(report, output_format="json"))["combined"]["engine"] == "combined"


def test_evaluation_missing_training_baseline_fails_without_fabricated_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pairs, profiles, assets, assignments = _evaluation_data(tmp_path)
    monkeypatch.setattr(evaluation_module, "_partition_accounts", lambda pairs, seed, fractions: assignments)
    monkeypatch.setattr(evaluation_module, "load_local_model", lambda config: _SyntheticModel())

    def failed_semantic(*args: object, **kwargs: object) -> float:
        raise ContractValidationError("model", ErrorCode.ENGINE_FAILURE)

    monkeypatch.setattr(evaluation_module, "compute_semantic_raw", failed_semantic)
    with pytest.raises(ContractValidationError) as caught:
        evaluate_labeled_pairs(
            pairs, profiles, assets=assets, config=EvaluationConfig(n_bootstrap=5),
            seed=1729, output_path=tmp_path / "evaluation.json",
        )
    assert str(caught.value) == "training_baselines:MINIMUMS_NOT_MET"
    assert not (tmp_path / "evaluation.json").exists()


def test_threshold_and_bootstrap_are_bounded_and_deterministic() -> None:
    assert evaluation_module._threshold([], 0.8) == (100.0, False)
    assert evaluation_module._threshold([(100.0, False)], 0.8) == (100.0, False)
    rows = [(90.0, True), (80.0, False), (70.0, False), (20.0, True)]
    config = EvaluationConfig(seed=5, n_bootstrap=100)
    first = evaluation_module._bootstrap_metric_ci(rows, metric="precision", threshold=50.0, config=config, seed_offset=1)
    second = evaluation_module._bootstrap_metric_ci(rows, metric="precision", threshold=50.0, config=config, seed_offset=1)
    assert first == second
    assert 0.0 <= first[0] < first[1] <= 1.0


def test_cli_accepts_three_engine_flags_without_global_baseline_preflight(
    prepared: tuple[dict[str, AccountProfile], object, object, ComparisonAssets],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    profiles, _, _, _ = prepared
    for name in ("AUTH_ATTR_MINIMUM_MESSAGES", "AUTH_ATTR_MINIMUM_CHARS", "AUTH_ATTR_DEVICE", "AUTH_ATTR_CACHE_RETENTION_DAYS"):
        monkeypatch.setenv(name, "original")
    monkeypatch.setattr(cli_module, "_profiles", lambda directory: profiles)
    captured: dict[str, float] = {}

    def fake_compare(flagged: AccountProfile, suspects: object, *, assets: ComparisonAssets, config: RunConfig) -> list[ComparisonReport]:
        captured["timeout"] = config.chronobiology_timeout_s
        stylo = _result("stylometry")
        semantic = _result("semantic", status="ERROR", raw=None, percentile=None, error="ARTIFACT_INVALID")
        chrono = _result("chronobiology")
        return [ComparisonReport("1.0", flagged.account_id, "acct-b", stylo, semantic, chrono, compute_combined_result(stylo, semantic, chrono), ComparisonReport.BANNER)]

    monkeypatch.setattr(cli_module, "compare_shortlist", fake_compare)
    common = [
        "--stylometry-artifact", "missing-style.json", "--chronobiology-artifact", "missing-chrono.json",
        "--model-dir", "missing-model", "--stylometry-baseline", "missing-style-baseline.json",
        "--semantic-baseline", "missing-semantic-baseline.json", "--chronobiology-baseline", "missing-chrono-baseline.json",
        "--game-terms", "games.txt", "--usernames", "users.txt", "--function-words", "words.txt", "--cache-dir", "cache",
    ]
    argv = [
        "compare", "--profiles", "unused", "--flagged", "acct-a", "--suspect", "acct-b",
        "--minimum-messages", "1", "--minimum-chars", "1", "--chronobiology-timeout", "1.25", "--format", "json",
        *common,
    ]
    assert cli_module.run_cli(argv) == 0
    assert captured["timeout"] == 1.25
    output = capsys.readouterr().out
    assert output.splitlines()[0] == ComparisonReport.BANNER
    assert json.loads(output.splitlines()[1])["combined"]["active_engines"] == ["stylometry", "chronobiology"]
    assert "SYNTHETIC_PRIVATE_TEXT_2981" not in output


def test_fit_artifact_script_writes_both_metadata_records(tmp_path: Path) -> None:
    profiles = _make_profiles(tmp_path, ["bg_a", "bg_b"], [0, 12])
    assert all(profile.status == "READY" for profile in profiles.values())
    words = tmp_path / "words.txt"
    words.write_text("# lang: en\nand\n", encoding="utf-8")
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    result = subprocess.run(
        [
            sys.executable, str(root / "fit_artifact.py"),
            "--profiles", str(tmp_path / "work"), "--function-words", str(words),
            "--output", str(tmp_path / "artifacts"),
            "--minimum-messages", "1", "--minimum-chars", "1",
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "artifacts" / "artifact.json").exists()
    assert (tmp_path / "artifacts" / "chronobiology-artifact.json").exists()
    assert "SYNTHETIC_PRIVATE_TEXT_2981" not in result.stdout + result.stderr

def test_worker_receives_only_its_own_stream_projection(
    prepared: tuple[dict[str, AccountProfile], object, object, ComparisonAssets],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from authorship_attribution.contracts import (
        NormalizedStreamChunk,
        RawStreamChunk,
        TemporalStreamChunk,
    )
    from authorship_attribution.streams import load_normalized_stream

    profiles, stylo, chrono, assets = prepared
    left = profiles["acct-a"]
    right = profiles["acct-b"]
    streams = {
        "stylometry": (load_raw_stream(left), load_raw_stream(right), RawStreamChunk),
        "semantic": (load_normalized_stream(left), load_normalized_stream(right), NormalizedStreamChunk),
        "chronobiology": (load_temporal_stream(left), load_temporal_stream(right), TemporalStreamChunk),
    }
    baselines = {
        "stylometry": _baseline("stylometry", stylo.configuration_fingerprint, [0.0], minimum_messages=1, minimum_chars=1),
        "semantic": _baseline("semantic", "0" * 64, [0.0], minimum_messages=1, minimum_chars=1),
        "chronobiology": _baseline("chronobiology", chrono.configuration_fingerprint, [0.0], minimum_messages=1, minimum_chars=1),
    }
    seen: list[str] = []

    def scorer(engine: str):
        def inner(candidate: object, suspect: object, candidate_stream: object, suspect_stream: object, **kwargs: object) -> EngineResult:
            expected_left, expected_right, expected_type = streams[engine]
            assert candidate_stream is expected_left
            assert suspect_stream is expected_right
            assert all(isinstance(item, expected_type) for item in candidate_stream)
            assert all(isinstance(item, expected_type) for item in suspect_stream)
            seen.append(engine)
            return _result(engine)
        return inner

    monkeypatch.setattr(orchestrator_module, "_WORKER_START_QUEUE", None)
    monkeypatch.setattr(orchestrator_module, "score_stylometry", scorer("stylometry"))
    monkeypatch.setattr(orchestrator_module, "score_semantic_local", scorer("semantic"))
    monkeypatch.setattr(orchestrator_module, "score_chronobiology", scorer("chronobiology"))
    for engine in ("stylometry", "semantic", "chronobiology"):
        monkeypatch.setattr(orchestrator_module, "_read_baseline", lambda path, engine=engine: baselines[engine])
        candidate_stream, suspect_stream, _ = streams[engine]
        result = orchestrator_module._worker_score(
            engine, "synthetic-token", left, right, candidate_stream, suspect_stream, assets, False
        )
        assert result.status == "OK"
    assert seen == ["stylometry", "semantic", "chronobiology"]


@pytest.mark.parametrize(
    ("statuses", "expected_status", "expected_error"),
    [
        (("INSUFFICIENT_DATA", "INSUFFICIENT_DATA", "INSUFFICIENT_DATA"), "INSUFFICIENT_DATA", "MINIMUMS_NOT_MET"),
        (("ERROR", "TIMEOUT", "INSUFFICIENT_DATA"), "TIMEOUT", "ENGINE_TIMEOUT"),
        (("INSUFFICIENT_DATA", "ERROR", "INSUFFICIENT_DATA"), "ERROR", "MINIMUMS_NOT_MET"),
    ],
)
def test_comparison_all_failed_status_precedence(
    prepared: tuple[dict[str, AccountProfile], object, object, ComparisonAssets],
    monkeypatch: pytest.MonkeyPatch,
    statuses: tuple[str, str, str],
    expected_status: str,
    expected_error: str,
) -> None:
    profiles, _, _, assets = prepared
    by_engine = dict(zip(("stylometry", "semantic", "chronobiology"), statuses, strict=True))
    errors = {
        "INSUFFICIENT_DATA": "MINIMUMS_NOT_MET",
        "TIMEOUT": "ENGINE_TIMEOUT",
        "ERROR": "ENGINE_FAILURE",
    }

    def fake_run(engine: str, candidate: object, suspect: object, assets: object, config: object, context: object) -> tuple[EngineResult, None, None]:
        status = by_engine[engine]
        return _result(engine, status=status, raw=None, percentile=None, error=errors[status]), None, None

    monkeypatch.setattr(orchestrator_module, "_run_engine", fake_run)
    report = orchestrator_module.compare_shortlist(
        profiles["acct-a"], [profiles["acct-b"]], assets=assets, config=RunConfig()
    )[0]
    assert report.combined.status == expected_status
    assert report.combined.error == expected_error
    assert report.combined.combined_score is None
    assert report.combined.weighted_percentile is None
    assert report.combined.fisher_tail_score is None
    assert report.combined.active_engines == []
    assert report.combined.weights_used == {}


def test_cli_build_baseline_and_evaluate_chronobiology_flags() -> None:
    parser = cli_module._build_parser()
    baseline_args = [
        "build-baseline", "--pairs", "pairs.ndjson", "--profiles", "work",
        "--stylometry-artifact", "style.json", "--chronobiology-artifact", "chrono.json",
        "--model-dir", "model", "--output", "baselines", "--function-words", "words.txt",
        "--cache-dir", "cache",
    ]
    parsed = parser.parse_args(baseline_args)
    assert parsed.chronobiology_artifact == Path("chrono.json")
    missing = baseline_args.copy()
    position = missing.index("--chronobiology-artifact")
    del missing[position:position + 2]
    with pytest.raises(SystemExit) as caught:
        parser.parse_args(missing)
    assert caught.value.code == 2

    evaluation_args = [
        "evaluate", "--pairs", "pairs.ndjson", "--profiles", "work", "--output", "evaluation.json",
        "--stylometry-artifact", "style.json", "--chronobiology-artifact", "chrono.json", "--model-dir", "model",
        "--stylometry-baseline", "style-baseline.json", "--semantic-baseline", "semantic-baseline.json",
        "--chronobiology-baseline", "chrono-baseline.json", "--game-terms", "games.txt",
        "--usernames", "users.txt", "--function-words", "words.txt", "--cache-dir", "cache",
    ]
    parsed = parser.parse_args(evaluation_args)
    assert parsed.chronobiology_artifact == "chrono.json"
    assert parsed.chronobiology_baseline == "chrono-baseline.json"
    missing = evaluation_args.copy()
    position = missing.index("--chronobiology-baseline")
    del missing[position:position + 2]
    with pytest.raises(SystemExit) as caught:
        parser.parse_args(missing)
    assert caught.value.code == 2


def test_training_scores_precede_heldout_scores_for_each_engine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pairs, profiles, assets, assignments = _evaluation_data(tmp_path)
    monkeypatch.setattr(evaluation_module, "_partition_accounts", lambda pairs, seed, fractions: assignments)
    monkeypatch.setattr(evaluation_module, "load_local_model", lambda config: _SyntheticModel())
    calls: dict[str, list[tuple[str, str]]] = {
        "stylometry": [], "semantic": [], "chronobiology": []
    }
    for engine, function_name in (
        ("stylometry", "compute_stylometry_raw"),
        ("semantic", "compute_semantic_raw"),
        ("chronobiology", "compute_chronobiology_raw"),
    ):
        original = getattr(evaluation_module, function_name)

        def wrapper(left: object, right: object, *, _engine: str = engine, _original: object = original, **kwargs: object) -> float:
            calls[_engine].append((left[0].account_id, right[0].account_id))
            return _original(left, right, **kwargs)

        monkeypatch.setattr(evaluation_module, function_name, wrapper)
    evaluate_labeled_pairs(
        pairs, profiles, assets=assets, config=EvaluationConfig(n_bootstrap=5),
        seed=1729, output_path=tmp_path / "evaluation.json",
    )
    for engine in calls:
        assert calls[engine][0] == ("ta", "tb")
        assert set(calls[engine][1:]) == {("va", "vb"), ("xa", "xb"), ("xc", "xd")}
