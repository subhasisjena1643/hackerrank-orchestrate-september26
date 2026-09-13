"""Command-line entry point for Buy or Wait?."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
DEFAULT_DATASET_DIR = REPO_ROOT / "dataset"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "output.csv"


def _repo_relative_path(value: str) -> Path:
    """Resolve relative CLI paths from the repository root, not the current directory."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Buy or Wait? decisions (Phase 1 scaffold)."
    )
    parser.add_argument(
        "--dataset",
        type=_repo_relative_path,
        default=DEFAULT_DATASET_DIR,
        help="Participant dataset directory (default: repository dataset/).",
    )
    parser.add_argument(
        "--output",
        type=_repo_relative_path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output CSV path (default: repository output.csv).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Enable strict validation once the decision pipeline is implemented.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    parser.error("The decision pipeline is not implemented in Phase 1.")


if __name__ == "__main__":
    raise SystemExit(main())
