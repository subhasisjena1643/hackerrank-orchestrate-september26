"""Command-line entry point for public-sample evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
DEFAULT_DATASET_DIR = REPO_ROOT / "dataset"


def _repo_relative_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate Buy or Wait? against public samples (Phase 1 scaffold)."
    )
    parser.add_argument(
        "--dataset",
        type=_repo_relative_path,
        default=DEFAULT_DATASET_DIR,
        help="Participant dataset directory (default: repository dataset/).",
    )
    parser.add_argument(
        "--strict", action="store_true", help="Enable strict validation."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    parser.error("Sample evaluation is not implemented in Phase 1.")


if __name__ == "__main__":
    raise SystemExit(main())
