"""Baseline-only Phase 9 sample diff; does not generate recommendations."""

from dataclasses import replace
from decimal import Decimal
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from buy_or_wait.data_loader import load_dataset
from buy_or_wait.finance.cashflows import construct_future_cash_flows
from buy_or_wait.finance.forecast import calculate_baseline_forecast
from buy_or_wait.finance.lifecycle import resolve_event_lifecycles
from buy_or_wait.finance.recurrence_policies import BASELINE_POLICY


# Phase 6 evidence values read from the five image-linked sample records.
IMAGE_AMOUNTS = {
    "event_253": "4365000",
    "event_1442": "100000",
    "event_1545": "41272",
    "event_1700": "2854",
    "event_1786": "704.05",
}


def main() -> None:
    dataset = load_dataset(Path(__file__).resolve().parents[2] / "dataset")
    profiles = {item.user_id: item for item in dataset.profiles}
    expected = {item.request_id: item for item in dataset.sample_outputs}
    print(
        "| request_id | expected safe | baseline safe | safe diff | expected earliest | baseline earliest |"
    )
    print("|---|---:|---:|---:|---|---|")
    for request in dataset.sample_requests:
        profile = profiles[request.user_id]
        events = tuple(
            replace(item, amount=Decimal(IMAGE_AMOUNTS[item.event_id]))
            if item.event_id in IMAGE_AMOUNTS
            else item
            for item in dataset.financial_events
            if item.user_id == request.user_id
        )
        resolved = resolve_event_lifecycles(
            events,
            snapshot_date=request.request_date,
            home_currency=profile.home_currency,
            exchange_rates=dataset.exchange_rates,
        )
        flows = construct_future_cash_flows(
            resolved,
            request_date=request.request_date,
            policy=BASELINE_POLICY,
        ).cash_flows
        actual = calculate_baseline_forecast(
            current_available_balance=profile.current_available_balance,
            minimum_balance_to_keep=profile.minimum_balance_to_keep,
            request_date=request.request_date,
            requested_amount=request.requested_amount,
            cash_flows=flows,
            policy=BASELINE_POLICY,
        )
        label = expected[request.request_id]
        print(
            f"| {request.request_id} | {label.amount_safe_to_pay} | {actual.amount_safe_to_pay} | "
            f"{actual.amount_safe_to_pay - label.amount_safe_to_pay} | "
            f"{label.earliest_date_for_full_payment or ''} | "
            f"{actual.earliest_date_for_full_payment or ''} |"
        )


if __name__ == "__main__":
    main()
