"""Focused Phase 2 input-contract and join tests."""

from __future__ import annotations

import csv
from dataclasses import FrozenInstanceError
from decimal import Decimal
from pathlib import Path
import sys

import pytest

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from buy_or_wait.data_loader import CSV_SCHEMAS, build_request_contexts, load_dataset
from buy_or_wait.validation.inputs import InputValidationError


def _write(path: Path, headers: tuple[str, ...], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=headers, lineterminator="\n", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def _valid_rows() -> dict[str, list[dict[str, str]]]:
    profiles = []
    for user_id in ("user_a", "user_b"):
        profiles.append(
            {
                "user_id": user_id,
                "home_currency": "USD",
                "current_available_balance": "1000.10",
                "minimum_balance_to_keep": "100",
                "financial_priorities": "emergency_savings",
                "expense_categories_to_protect": "rent|groceries",
                "expense_categories_user_is_willing_to_reduce": "dining",
                "expense_categories_user_is_willing_to_stop": "streaming",
                "payment_methods_user_will_consider": "full_payment|partial_payment",
                "max_installment_months": "",
            }
        )
    event = {
        "event_id": "event_a",
        "user_id": "user_a",
        "event_type": "expense",
        "description": "Groceries",
        "category": "groceries",
        "direction": "debit",
        "amount": "25.50",
        "currency": "USD",
        "event_date": "2026-01-01",
        "settlement_date": "2026-01-01",
        "status": "settled",
        "linked_event_id": "",
        "flexibility": "fixed",
        "minimum_allowed_amount": "",
    }
    requests = []
    for request_id, user_id, amount in (
        ("request_b", "user_b", "200"),
        ("request_a", "user_a", "100.25"),
    ):
        requests.append(
            {
                "request_id": request_id,
                "user_id": user_id,
                "request_date": "2026-02-01",
                "request_type": "purchase",
                "requested_amount": amount,
                "desired_completion_date": "2026-03-01",
                "allows_partial_payment": "true",
                "request_text": "Can I afford this?",
            }
        )
    sample = dict(requests[1])
    sample["request_id"] = "sample_a"
    sample.update(
        {
            "amount_safe_to_pay": "100.25",
            "affordability_status": "affordable_now",
            "recommended_payment_method": "full_payment",
            "payment_plan": "2026-02-01:100.25",
            "earliest_date_for_full_payment": "2026-02-01",
            "spending_changes_needed": "none",
            "decision_explanation": "Safe in this example.",
        }
    )
    options = []
    for request_id in ("request_b", "request_a", "sample_a"):
        options.extend(
            [
                {
                    "payment_option_id": f"option_{request_id}_full",
                    "request_id": request_id,
                    "payment_method": "full_payment",
                    "payment_amount": "100.25",
                    "number_of_payments": "1",
                    "first_payment_date": "2026-02-01",
                    "payment_frequency_days": "",
                    "financing_fee": "0",
                    "total_payable_amount": "100.25",
                },
                {
                    "payment_option_id": f"option_{request_id}_installment",
                    "request_id": request_id,
                    "payment_method": "installments",
                    "payment_amount": "51",
                    "number_of_payments": "2",
                    "first_payment_date": "2026-02-01",
                    "payment_frequency_days": "30",
                    "financing_fee": "1.75",
                    "total_payable_amount": "102",
                },
            ]
        )
    messages = [
        {
            "message_id": "message_user",
            "user_id": "user_a",
            "request_id": "",
            "related_event_id": "",
            "sent_at": "2026-01-10T10:00:00Z",
            "source_type": "bank",
            "message_text": "User-level notice.",
        },
        {
            "message_id": "message_request",
            "user_id": "user_a",
            "request_id": "request_a",
            "related_event_id": "event_a",
            "sent_at": "2026-01-11T10:00:00Z",
            "source_type": "merchant",
            "message_text": "Request notice.",
        },
    ]
    return {
        "financial_profiles.csv": profiles,
        "financial_events.csv": [event],
        "exchange_rates.csv": [
            {
                "rate_date": "2026-01-01",
                "from_currency": "EUR",
                "to_currency": "USD",
                "rate": "1.09",
            }
        ],
        "requests.csv": requests,
        "sample_requests.csv": [sample],
        "request_payment_options.csv": options,
        "messages.csv": messages,
        "images.csv": [],
    }


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    rows = _valid_rows()
    for filename, headers in CSV_SCHEMAS.items():
        _write(tmp_path / filename, headers, rows[filename])
    return tmp_path


def test_missing_columns_are_actionable(dataset_dir: Path) -> None:
    rows = _valid_rows()["requests.csv"]
    headers = CSV_SCHEMAS["requests.csv"][:-1]
    _write(dataset_dir / "requests.csv", headers, rows)
    with pytest.raises(InputValidationError, match=r"missing columns: request_text"):
        load_dataset(dataset_dir)


def test_duplicate_ids_are_rejected(dataset_dir: Path) -> None:
    rows = _valid_rows()["financial_profiles.csv"]
    rows.append(dict(rows[0]))
    _write(
        dataset_dir / "financial_profiles.csv",
        CSV_SCHEMAS["financial_profiles.csv"],
        rows,
    )
    with pytest.raises(
        InputValidationError, match=r"duplicate user_id; first seen on row 2"
    ):
        load_dataset(dataset_dir)


def test_invalid_dates_are_rejected(dataset_dir: Path) -> None:
    rows = _valid_rows()["requests.csv"]
    rows[0]["request_date"] = "2026-02-30"
    _write(dataset_dir / "requests.csv", CSV_SCHEMAS["requests.csv"], rows)
    with pytest.raises(InputValidationError, match=r"request_date.*YYYY-MM-DD"):
        load_dataset(dataset_dir)


@pytest.mark.parametrize(
    ("filename", "column", "bad_value"),
    (
        ("financial_events.csv", "direction", "outgoing"),
        ("requests.csv", "request_type", "holiday"),
    ),
)
def test_invalid_enums_are_rejected(
    dataset_dir: Path, filename: str, column: str, bad_value: str
) -> None:
    rows = _valid_rows()[filename]
    rows[0][column] = bad_value
    _write(dataset_dir / filename, CSV_SCHEMAS[filename], rows)
    with pytest.raises(InputValidationError, match=rf"\[{column}\].*must be one of"):
        load_dataset(dataset_dir)


def test_broken_foreign_keys_are_rejected(dataset_dir: Path) -> None:
    rows = _valid_rows()["request_payment_options.csv"]
    rows[0]["request_id"] = "missing_request"
    _write(
        dataset_dir / "request_payment_options.csv",
        CSV_SCHEMAS["request_payment_options.csv"],
        rows,
    )
    with pytest.raises(InputValidationError, match=r"broken foreign key.*request_id"):
        load_dataset(dataset_dir)


def test_cross_user_evidence_is_rejected(dataset_dir: Path) -> None:
    rows = _valid_rows()["messages.csv"]
    rows[0]["request_id"] = "request_b"
    _write(dataset_dir / "messages.csv", CSV_SCHEMAS["messages.csv"], rows)
    with pytest.raises(InputValidationError, match=r"cross-user evidence leakage"):
        load_dataset(dataset_dir)


def test_input_order_and_decimal_types_are_preserved(dataset_dir: Path) -> None:
    dataset = load_dataset(dataset_dir)
    contexts = build_request_contexts(dataset)
    assert [context.request.request_id for context in contexts] == [
        "request_b",
        "request_a",
    ]
    assert [message.message_id for message in contexts[1].messages] == [
        "message_user",
        "message_request",
    ]
    assert contexts[1].request.requested_amount == Decimal("100.25")
    assert isinstance(contexts[1].request.requested_amount, Decimal)
    with pytest.raises(FrozenInstanceError):
        contexts[1].request.request_text = "mutated"  # type: ignore[misc]


def test_sample_outputs_are_separate_and_opt_in(dataset_dir: Path) -> None:
    rows = _valid_rows()["sample_requests.csv"]
    rows[0]["affordability_status"] = "secret_invalid_label"
    _write(
        dataset_dir / "sample_requests.csv", CSV_SCHEMAS["sample_requests.csv"], rows
    )

    dataset = load_dataset(dataset_dir, include_sample_outputs=False)
    assert dataset.sample_outputs == ()
    assert all(
        context.request.request_id != "sample_a"
        for context in build_request_contexts(dataset)
    )
    with pytest.raises(
        InputValidationError, match=r"affordability_status.*must be one of"
    ):
        load_dataset(dataset_dir, include_sample_outputs=True)
