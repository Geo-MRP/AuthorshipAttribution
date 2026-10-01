#!/usr/bin/env python3
"""Fit the stylometry artifact and save the metadata file the CLI expects.

The CLI's build-baseline and compare commands both need --stylometry-artifact
pointing at a JSON metadata file, but no CLI command creates it. This script
fills that gap using the library functions directly.

It fits ONLY on the accounts whose id starts with --prefix (default "bg_"), so
the accounts you will later test (alice_main, alice_alt, bob) stay out of the
reference corpus.

Usage:
    python fit_artifact.py --profiles demo/work --function-words demo/function_words.txt \
        --output demo/artifact
Then pass  --stylometry-artifact demo/artifact/artifact.json  to the CLI.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from authorship_attribution.contracts import AccountProfile, StylometryConfig
from authorship_attribution.fileformats import load_term_list, read_json_artifact, write_json_artifact
from authorship_attribution.streams import load_raw_stream
from authorship_attribution.stylometry import fit_stylometry_artifact


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--profiles", required=True, type=Path, help="the --output directory you gave to `ingest`")
    parser.add_argument("--function-words", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prefix", default="bg_", help="only accounts whose id starts with this are used")
    parser.add_argument("--minimum-messages", type=int, default=200)
    parser.add_argument("--minimum-chars", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    profiles = [
        AccountProfile.from_dict(read_json_artifact(path))
        for path in sorted((args.profiles / "profiles").glob("*.json"))
    ]
    chosen = [p for p in profiles if p.account_id.startswith(args.prefix) and p.status == "READY"]
    print(f"found {len(profiles)} profiles; fitting on {len(chosen)} READY accounts starting with {args.prefix!r}")
    if len(chosen) < 2:
        sys.exit("need at least 2 accounts to fit; did `ingest` finish, and does the prefix match?")

    # Absolute path: the artifact records where its payload file lives.
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    words = load_term_list(args.function_words, require_lang=True)
    config = StylometryConfig(minimum_messages=args.minimum_messages, minimum_chars=args.minimum_chars)
    artifact = fit_stylometry_artifact(
        [load_raw_stream(profile) for profile in chosen],
        function_words=words.entries,
        config=config,
        seed=args.seed,
        output_dir=output,
    )
    metadata_path = output / "artifact.json"
    write_json_artifact(artifact.to_dict(), metadata_path)
    print(f"wrote {metadata_path}")
    print(f"use:  --stylometry-artifact {metadata_path}")


if __name__ == "__main__":
    main()
