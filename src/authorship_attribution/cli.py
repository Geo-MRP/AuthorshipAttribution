"""Local CLI for ingest, comparison, baselines, and evaluation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import logging
import os
from pathlib import Path
import sys

from .contracts import (
    AccountProfile,
    BaselineConfig,
    ComparisonAssets,
    EvaluationConfig,
    IngestConfig,
    PreprocessConfig,
    RunConfig,
    SemanticConfig,
    StylometryArtifact,
    StylometryConfig,
)
from .errors import ContractValidationError, ErrorCode, error_code_for
from .evaluate import evaluate_labeled_pairs
from .fileformats import load_labeled_pairs, load_term_list, read_json_artifact, write_json_artifact
from .ingest import ingest_chat_logs
from .orchestrator import compare_shortlist
from .preprocess import build_account_profiles
from .reporting import render_compare_report, render_evaluation_report
from .semantic import load_local_model
from .stylometry import fit_stylometry_artifact

LOGGER = logging.getLogger(__name__)


def _profiles(directory: Path) -> dict[str, AccountProfile]:
    profiles: dict[str, AccountProfile] = {}
    for path in sorted((directory / "profiles").glob("*.json")):
        data = read_json_artifact(path)
        profile = AccountProfile.from_dict(data)
        profiles[profile.account_id] = profile
    return profiles


def _assets(args: argparse.Namespace) -> ComparisonAssets:
    return ComparisonAssets(
        stylometry_artifact_path=Path(args.stylometry_artifact),
        semantic_model_dir=Path(args.model_dir),
        baseline_paths={"stylometry": Path(args.stylometry_baseline), "semantic": Path(args.semantic_baseline)},
        game_terms_path=Path(args.game_terms), usernames_path=Path(args.usernames),
        function_words_path=Path(args.function_words), cache_dir=Path(args.cache_dir),
    )


def _shared_asset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--stylometry-artifact", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--stylometry-baseline", required=True)
    parser.add_argument("--semantic-baseline", required=True)
    parser.add_argument("--game-terms", required=True)
    parser.add_argument("--usernames", required=True)
    parser.add_argument("--function-words", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--allow-external", action="store_true")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="authorship-attribution")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="ingest chat logs and build isolated stream/profile artifacts")
    ingest.add_argument("inputs", nargs="+", type=Path)
    ingest.add_argument("--output", required=True, type=Path)
    ingest.add_argument("--input-format", choices=("auto", "json", "csv"), default="auto")
    ingest.add_argument("--minimum-messages", type=int, default=200)
    ingest.add_argument("--minimum-chars", type=int, default=5000)
    ingest.add_argument("--utc-hour", action="append", type=int)
    ingest.add_argument("--cache-retention-days", type=int, default=7)
    ingest.add_argument("--game-terms", required=True, type=Path)
    ingest.add_argument("--usernames", required=True, type=Path)

    compare = commands.add_parser("compare", help="compare a flagged account with suspect accounts")
    compare.add_argument("--profiles", required=True, type=Path)
    compare.add_argument("--flagged", required=True)
    compare.add_argument("--suspect", required=True, action="append")
    compare.add_argument("--stylometry-artifact", required=True)
    compare.add_argument("--model-dir", required=True)
    compare.add_argument("--stylometry-baseline", required=True)
    compare.add_argument("--semantic-baseline", required=True)
    compare.add_argument("--game-terms", required=True)
    compare.add_argument("--usernames", required=True)
    compare.add_argument("--function-words", required=True)
    compare.add_argument("--cache-dir", required=True)
    compare.add_argument("--minimum-messages", type=int, default=200)
    compare.add_argument("--minimum-chars", type=int, default=5000)
    compare.add_argument("--utc-hour", action="append", type=int)
    compare.add_argument("--cache-retention-days", type=int, default=7)
    compare.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    compare.add_argument("--stylometry-timeout", type=float, default=30.0)
    compare.add_argument("--semantic-timeout", type=float, default=90.0)
    compare.add_argument("--format", choices=("table", "json"), default="table")
    compare.add_argument("--allow-external", action="store_true")

    baseline = commands.add_parser("build-baseline", help="build immutable per-engine different-pair baselines")
    baseline.add_argument("--pairs", required=True, type=Path)
    baseline.add_argument("--profiles", required=True, type=Path)
    baseline.add_argument("--stylometry-artifact", required=True, type=Path)
    baseline.add_argument("--model-dir", required=True, type=Path)
    baseline.add_argument("--output", required=True, type=Path)
    baseline.add_argument("--function-words", required=True, type=Path)
    baseline.add_argument("--cache-dir", required=True, type=Path)
    baseline.add_argument("--minimum-messages", type=int, default=200)
    baseline.add_argument("--minimum-chars", type=int, default=5000)
    baseline.add_argument("--max-pairs", type=int, default=50000)
    baseline.add_argument("--seed", type=int, default=1729)

    evaluation = commands.add_parser("evaluate", help="evaluate labeled pairs with account-disjoint partitions")
    evaluation.add_argument("--pairs", required=True, type=Path)
    evaluation.add_argument("--profiles", required=True, type=Path)
    evaluation.add_argument("--output", required=True, type=Path)
    evaluation.add_argument("--format", choices=("table", "json"), default="table")
    _shared_asset_args(evaluation)
    return parser


def run_cli(argv: Sequence[str]) -> int:
    """Run CLI and return a process exit status without printing content values."""
    parser = _build_parser()
    args = parser.parse_args(list(argv))
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        if args.command == "ingest":
            messages = list(ingest_chat_logs(args.inputs, config=IngestConfig(input_format=args.input_format)))
            terms = load_term_list(args.game_terms, require_lang=False)
            names = load_term_list(args.usernames, require_lang=False)
            config = PreprocessConfig(
                minimum_messages=args.minimum_messages, minimum_chars=args.minimum_chars,
                utc_hour_filter=None if args.utc_hour is None else sorted(set(args.utc_hour)),
                cache_retention_days=args.cache_retention_days,
            )
            profiles = build_account_profiles(messages, game_terms=terms.entries, usernames=names.entries, config=config, output_dir=args.output)
            LOGGER.info("ingest_complete profiles=%d", len(profiles))
            return 0
        if args.command == "compare":
            profiles = _profiles(args.profiles)
            flagged = profiles[args.flagged]
            suspects = [profiles[account] for account in sorted(set(args.suspect))]
            requested_hours = None if args.utc_hour is None else sorted(set(args.utc_hour))
            if any(profile.utc_hour_filter != requested_hours for profile in [flagged, *suspects]):
                print("INPUT_INVALID", file=sys.stderr)
                return 2
            os.environ["AUTH_ATTR_MINIMUM_MESSAGES"] = str(args.minimum_messages)
            os.environ["AUTH_ATTR_MINIMUM_CHARS"] = str(args.minimum_chars)
            os.environ["AUTH_ATTR_DEVICE"] = args.device
            os.environ["AUTH_ATTR_CACHE_RETENTION_DAYS"] = str(args.cache_retention_days)
            assets = _assets(args)
            for profile in [flagged, *suspects]:
                if (profile.n_messages < args.minimum_messages or profile.n_chars < args.minimum_chars):
                    continue
                if profile.status != "READY":
                    continue
            for engine_path in assets.baseline_paths.values():
                artifact = read_json_artifact(engine_path)
                if (artifact.get("minimum_messages") != args.minimum_messages or artifact.get("minimum_chars") != args.minimum_chars):
                    print("BASELINE_CONFIG_MISMATCH", file=sys.stderr)
                    return 2
            config = RunConfig(
                stylometry_timeout_s=args.stylometry_timeout, semantic_timeout_s=args.semantic_timeout,
                output_format=args.format, allow_external=args.allow_external,
            )
            if args.allow_external:
                LOGGER.info("external_mode_enabled=true")
            for report in compare_shortlist(flagged, suspects, assets=assets, config=config):
                print(render_compare_report(report, output_format=args.format))
            return 0
        if args.command == "build-baseline":
            pairs = load_labeled_pairs(args.pairs)
            profiles = _profiles(args.profiles)
            artifact = StylometryArtifact.from_dict(read_json_artifact(args.stylometry_artifact))
            function_words = load_term_list(args.function_words, require_lang=True)
            model_config = SemanticConfig(model_dir=args.model_dir, cache_dir=args.cache_dir, center_embeddings=False)
            model = load_local_model(model_config)
            # Baseline fitting relies on the engine artifact's feature settings.
            from .baselines import build_baselines
            from dataclasses import replace
            artifact_config = StylometryConfig(
                feature_spec_version=artifact.feature_spec_version,
                hash_space_size=artifact.hash_space_size,
                minimum_messages=args.minimum_messages,
                minimum_chars=args.minimum_chars,
            )
            # The artifact was fitted with the indicated engine configuration.
            # Baseline computation must use the same feature version and bounds.
            config = BaselineConfig(max_pairs=args.max_pairs)
            _ = function_words, artifact_config
            built = build_baselines(pairs, profiles, stylometry_artifact=artifact, semantic_model=model,
                                    config=config, seed=args.seed, output_dir=args.output)
            LOGGER.info("baseline_complete engines=%d", len(built))
            return 0
        if args.command == "evaluate":
            pairs = load_labeled_pairs(args.pairs)
            profiles = _profiles(args.profiles)
            report = evaluate_labeled_pairs(pairs, profiles, assets=_assets(args), config=EvaluationConfig(), seed=1729, output_path=args.output)
            print(render_evaluation_report(report, output_format=args.format))
            return 0
    except Exception as exc:
        print(error_code_for(exc).value, file=sys.stderr)
        return 2
    return 2


__all__ = ["run_cli"]


def main() -> None:
    """Console-script entry point."""
    raise SystemExit(run_cli(sys.argv[1:]))
