"""Run Phase 6 image extraction and print only non-sensitive status fields."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from buy_or_wait.data_loader import load_dataset
from buy_or_wait.evidence.client import create_live_evidence_client
from buy_or_wait.evidence.images import resolve_all_linked_images


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "dataset")
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="Optional validated evidence cache; omitted by default so every image is reprocessed.",
    )
    args = parser.parse_args()
    dataset = load_dataset(args.dataset, include_sample_outputs=False)
    results = resolve_all_linked_images(
        dataset,
        dataset_dir=args.dataset,
        client=create_live_evidence_client(),
        cache_dir=args.cache,
    )
    print(
        "image_id,event_id,ocr_status,vision_status,consensus_adjudication_status,currency,confidence"
    )
    for result in results:
        print(
            ",".join(
                (
                    result.image_id,
                    result.event_id,
                    result.ocr_result.status,
                    result.vision_result.status,
                    result.resolution_status,
                    result.currency.value,
                    result.confidence,
                )
            )
        )
    if len(results) != 16:
        raise RuntimeError(
            f"expected 16 linked blank image amounts; resolved {len(results)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
