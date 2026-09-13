"""Run Phase 13 gates followed by the frozen public-sample tournament."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import subprocess
import sys
from typing import Sequence

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "dataset")
    args = parser.parse_args(argv)
    subprocess.run(
        [sys.executable, str(CODE_DIR / "evaluation" / "main.py"), "--manifest-only"],
        cwd=REPO_ROOT,
        check=True,
    )
    tests = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", str(CODE_DIR / "tests")],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    print(tests.stdout, end="")
    if tests.returncode:
        return tests.returncode
    matches = re.findall(r"(\d+) passed", tests.stdout)
    if not matches:
        raise RuntimeError("could not determine passing test count")
    tournament = subprocess.run(
        [
            sys.executable,
            str(CODE_DIR / "evaluation" / "main.py"),
            "--dataset",
            str(args.dataset.resolve()),
            "--tests-passed",
            matches[-1],
        ],
        cwd=REPO_ROOT,
        check=False,
    )
    return tournament.returncode


if __name__ == "__main__":
    raise SystemExit(main())
