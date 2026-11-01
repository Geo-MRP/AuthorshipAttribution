#!/usr/bin/env python3
"""Fit stylometry and chronobiology metadata from local background profiles."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from authorship_attribution.chronobiology import fit_chronobiology_artifact
from authorship_attribution.contracts import (
    AccountProfile,
    ChronobiologyConfig,
    StylometryConfig,
)
from authorship_attribution.errors import error_code_for
from authorship_attribution.fileformats import (
    load_term_list,
    read_json_artifact,
    write_json_artifact,
)
from authorship_attribution.streams import load_raw_stream, load_temporal_stream
from authorship_attribution.stylometry import fit_stylometry_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--function-words", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prefix", default="bg_")
    parser.add_argument("--minimum-messages", type=int, default=200)
    parser.add_argument("--minimum-chars", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    try:
        profiles = [
            AccountProfile.from_dict(read_json_artifact(path))
            for path in sorted((args.profiles / "profiles").glob("*.json"))
        ]
        chosen = [
            profile for profile in profiles
            if profile.account_id.startswith(args.prefix) and profile.status == "READY"
        ]
        if len(chosen) < 2:
            print("MINIMUMS_NOT_MET", file=sys.stderr)
            raise SystemExit(2)
        if any(
            profile.minimum_messages != args.minimum_messages
            or profile.minimum_chars != args.minimum_chars
            or profile.n_messages < args.minimum_messages
            or profile.n_chars < args.minimum_chars
            for profile in chosen
        ):
            print("BASELINE_CONFIG_MISMATCH", file=sys.stderr)
            raise SystemExit(2)
        fingerprints = {profile.preprocessing_fingerprint for profile in chosen}
        if len(fingerprints) != 1:
            print("BASELINE_CONFIG_MISMATCH", file=sys.stderr)
            raise SystemExit(2)

        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=True)
        words = load_term_list(args.function_words, require_lang=True)
        stylometry_config = StylometryConfig(
            minimum_messages=args.minimum_messages,
            minimum_chars=args.minimum_chars,
        )
        chronobiology_config = ChronobiologyConfig(
            minimum_messages=args.minimum_messages,
            minimum_chars=args.minimum_chars,
        )
        stylometry = fit_stylometry_artifact(
            [load_raw_stream(profile) for profile in chosen],
            function_words=words.entries,
            config=stylometry_config,
            seed=args.seed,
            output_dir=output,
        )
        chronobiology = fit_chronobiology_artifact(
            [load_temporal_stream(profile) for profile in chosen],
            config=chronobiology_config,
            preprocessing_fp=next(iter(fingerprints)),
            output_dir=output,
        )
        write_json_artifact(stylometry, output / "artifact.json")
        write_json_artifact(chronobiology, output / "chronobiology-artifact.json")
        print(f"fit_complete accounts={len(chosen)} artifacts=2")
    except SystemExit:
        raise
    except Exception as exc:
        print(error_code_for(exc).value, file=sys.stderr)
        raise SystemExit(2) from None


if __name__ == "__main__":
    main()
