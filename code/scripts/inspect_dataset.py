"""Read-only schema and distribution diagnostics for the supplied dataset."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
from typing import Iterable


CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from buy_or_wait.data_loader import CSV_SCHEMAS, load_dataset  # noqa: E402


def _distribution(values: Iterable[object]) -> str:
    counts = Counter(str(value) for value in values)
    return ", ".join(f"{key}={counts[key]}" for key in sorted(counts))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and print read-only dataset schema/distribution diagnostics."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=REPO_ROOT / "dataset",
        help="Dataset directory (default: repository dataset/).",
    )
    parser.add_argument(
        "--include-sample-summary",
        action="store_true",
        help="Explicitly allow parsing and summarizing solved sample output columns.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dataset = load_dataset(
        args.dataset,
        include_sample_outputs=args.include_sample_summary,
    )

    print("Schemas")
    for filename, headers in CSV_SCHEMAS.items():
        print(f"  {filename}: {','.join(headers)}")

    print("Counts")
    print(f"  profiles: {len(dataset.profiles)}")
    print(f"  evaluation_requests: {len(dataset.evaluation_requests)}")
    print(f"  solved_samples: {len(dataset.sample_requests)}")
    print(f"  financial_events: {len(dataset.financial_events)}")
    print(f"  payment_options: {len(dataset.payment_options)}")
    print(f"  exchange_rates: {len(dataset.exchange_rates)}")
    print(f"  messages: {len(dataset.messages)}")
    print(f"  images: {len(dataset.images)}")

    blank_amount_events = [
        event for event in dataset.financial_events if event.amount is None
    ]
    linked_image_events = {
        image.related_event_id
        for image in dataset.images
        if image.related_event_id is not None
    }
    print(f"  blank_financial_event_amounts: {len(blank_amount_events)}")
    print(
        "  blank_amounts_linked_to_images: "
        f"{sum(event.event_id in linked_image_events for event in blank_amount_events)}"
    )

    print("Distributions")
    print(
        f"  profile.home_currency: {_distribution(p.home_currency.value for p in dataset.profiles)}"
    )
    print(
        f"  financial_event.event_type: {_distribution(e.event_type.value for e in dataset.financial_events)}"
    )
    print(
        f"  financial_event.direction: {_distribution(e.direction.value for e in dataset.financial_events)}"
    )
    print(
        f"  financial_event.status: {_distribution(e.status.value for e in dataset.financial_events)}"
    )
    print(
        f"  request.request_type: {_distribution(r.request_type.value for r in dataset.evaluation_requests)}"
    )
    print(
        f"  payment_option.payment_method: {_distribution(o.payment_method.value for o in dataset.payment_options)}"
    )
    print(
        f"  message.source_type: {_distribution(m.source_type.value for m in dataset.messages)}"
    )

    if args.include_sample_summary:
        print("Solved sample output distributions (explicitly enabled)")
        print(
            "  affordability_status: "
            f"{_distribution(o.affordability_status.value for o in dataset.sample_outputs)}"
        )
        print(
            "  recommended_payment_method: "
            f"{_distribution(o.recommended_payment_method.value for o in dataset.sample_outputs)}"
        )
    else:
        print(
            "Solved sample output columns: not read (pass --include-sample-summary to opt in)"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
