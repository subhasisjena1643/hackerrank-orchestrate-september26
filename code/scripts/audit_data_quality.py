"""Generate the read-only data-quality report outside dataset/."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from buy_or_wait.data_quality import audit_dataset, render_markdown  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit participant CSVs without modifying raw data."
    )
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "dataset")
    parser.add_argument(
        "--output",
        type=Path,
        default=CODE_DIR / "evaluation" / "data_quality_report.md",
    )
    args = parser.parse_args()
    report = audit_dataset(args.dataset)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    print(f"data_quality={'PASS' if report.passed else 'FAIL'} report={args.output}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
