"""Strict, deterministic CSV loading and request-context joins."""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Callable, Iterator, TypeVar

from buy_or_wait.domain import (
    AffordabilityStatus,
    Currency,
    Dataset,
    Direction,
    EventStatus,
    EventType,
    ExchangeRate,
    FinancialEvent,
    Flexibility,
    ImageRecord,
    MessageRecord,
    PaymentMethod,
    PaymentOption,
    Profile,
    Request,
    RequestContext,
    RequestType,
    SampleOutput,
    SourceType,
)
from buy_or_wait.validation.inputs import (
    InputValidationError,
    ensure_unique,
    fail,
    optional_text,
    parse_bool,
    parse_date,
    parse_datetime,
    parse_decimal,
    parse_enum,
    parse_int,
    parse_pipe_list,
    required_text,
    require_foreign_key,
    validate_headers,
)


PROFILE_HEADERS = (
    "user_id",
    "home_currency",
    "current_available_balance",
    "minimum_balance_to_keep",
    "financial_priorities",
    "expense_categories_to_protect",
    "expense_categories_user_is_willing_to_reduce",
    "expense_categories_user_is_willing_to_stop",
    "payment_methods_user_will_consider",
    "max_installment_months",
)
FINANCIAL_EVENT_HEADERS = (
    "event_id",
    "user_id",
    "event_type",
    "description",
    "category",
    "direction",
    "amount",
    "currency",
    "event_date",
    "settlement_date",
    "status",
    "linked_event_id",
    "flexibility",
    "minimum_allowed_amount",
)
EXCHANGE_RATE_HEADERS = ("rate_date", "from_currency", "to_currency", "rate")
REQUEST_HEADERS = (
    "request_id",
    "user_id",
    "request_date",
    "request_type",
    "requested_amount",
    "desired_completion_date",
    "allows_partial_payment",
    "request_text",
)
SAMPLE_OUTPUT_HEADERS = (
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
)
SAMPLE_REQUEST_HEADERS = REQUEST_HEADERS + SAMPLE_OUTPUT_HEADERS
PAYMENT_OPTION_HEADERS = (
    "payment_option_id",
    "request_id",
    "payment_method",
    "payment_amount",
    "number_of_payments",
    "first_payment_date",
    "payment_frequency_days",
    "financing_fee",
    "total_payable_amount",
)
MESSAGE_HEADERS = (
    "message_id",
    "user_id",
    "request_id",
    "related_event_id",
    "sent_at",
    "source_type",
    "message_text",
)
IMAGE_HEADERS = ("image_id", "user_id", "request_id", "related_event_id")

CSV_SCHEMAS: dict[str, tuple[str, ...]] = {
    "financial_profiles.csv": PROFILE_HEADERS,
    "financial_events.csv": FINANCIAL_EVENT_HEADERS,
    "exchange_rates.csv": EXCHANGE_RATE_HEADERS,
    "requests.csv": REQUEST_HEADERS,
    "sample_requests.csv": SAMPLE_REQUEST_HEADERS,
    "request_payment_options.csv": PAYMENT_OPTION_HEADERS,
    "messages.csv": MESSAGE_HEADERS,
    "images.csv": IMAGE_HEADERS,
}


def _rows(
    path: Path,
    expected_headers: tuple[str, ...],
    *,
    selected_headers: tuple[str, ...] | None = None,
) -> Iterator[tuple[int, dict[str, str]]]:
    if not path.is_file():
        fail(path, "required dataset file does not exist")
    try:
        handle = path.open("r", encoding="utf-8-sig", newline="")
    except OSError as exc:
        fail(path, f"could not open file: {exc}")
    with handle:
        reader = csv.reader(handle, strict=True)
        try:
            headers = next(reader)
        except StopIteration:
            fail(path, "CSV is empty; expected a header row")
        except csv.Error as exc:
            fail(path, f"invalid CSV header: {exc}")
        validate_headers(path, headers, expected_headers)
        row_number = 1
        while True:
            row_number += 1
            try:
                values = next(reader)
            except StopIteration:
                break
            except csv.Error as exc:
                fail(path, f"invalid CSV syntax: {exc}", row_number=row_number)
            if len(values) != len(expected_headers):
                fail(
                    path,
                    f"row has {len(values)} fields but {len(expected_headers)} are required",
                    row_number=row_number,
                )
            projected = selected_headers or expected_headers
            # In safe sample mode, output values are deliberately not associated
            # with column names or exposed beyond the CSV parser.
            yield (
                row_number,
                {
                    column: values[expected_headers.index(column)]
                    for column in projected
                },
            )


T = TypeVar("T")


def _load_with_unique_id(
    path: Path,
    headers: tuple[str, ...],
    identifier_name: str,
    parser: Callable[[Path, int, dict[str, str]], T],
) -> tuple[T, ...]:
    result: list[T] = []
    seen: dict[str, int] = {}
    for row_number, row in _rows(path, headers):
        identifier = required_text(
            path, row_number, identifier_name, row[identifier_name]
        )
        ensure_unique(path, identifier_name, identifier, seen, row_number)
        result.append(parser(path, row_number, row))
    return tuple(result)


def _required_decimal(
    path: Path,
    row: int,
    column: str,
    value: str,
    *,
    positive: bool = False,
    minimum: Decimal | None = None,
) -> Decimal:
    parsed = parse_decimal(
        path, row, column, value, strictly_positive=positive, minimum=minimum
    )
    assert parsed is not None
    return parsed


def _required_int(
    path: Path, row: int, column: str, value: str, *, minimum: int | None = None
) -> int:
    parsed = parse_int(path, row, column, value, minimum=minimum)
    assert parsed is not None
    return parsed


def _required_date(path: Path, row: int, column: str, value: str) -> date:
    parsed = parse_date(path, row, column, value)
    assert parsed is not None
    return parsed


def _parse_profile(path: Path, row_number: int, row: dict[str, str]) -> Profile:
    methods: list[PaymentMethod] = []
    for raw_method in parse_pipe_list(row["payment_methods_user_will_consider"]):
        method = parse_enum(
            path,
            row_number,
            "payment_methods_user_will_consider",
            raw_method,
            PaymentMethod,
        )
        if method not in {
            PaymentMethod.FULL_PAYMENT,
            PaymentMethod.PARTIAL_PAYMENT,
            PaymentMethod.INSTALLMENTS,
        }:
            fail(
                path,
                "profile may only consider full_payment, partial_payment, or installments",
                row_number=row_number,
                column="payment_methods_user_will_consider",
                value=raw_method,
            )
        if method in methods:
            fail(
                path,
                "payment method is listed more than once",
                row_number=row_number,
                column="payment_methods_user_will_consider",
                value=raw_method,
            )
        methods.append(method)
    if not methods:
        fail(
            path,
            "at least one payment method is required",
            row_number=row_number,
            column="payment_methods_user_will_consider",
        )
    max_months = parse_int(
        path,
        row_number,
        "max_installment_months",
        row["max_installment_months"],
        optional=True,
        minimum=1,
    )
    return Profile(
        user_id=required_text(path, row_number, "user_id", row["user_id"]),
        home_currency=parse_enum(
            path, row_number, "home_currency", row["home_currency"], Currency
        ),
        current_available_balance=_required_decimal(
            path,
            row_number,
            "current_available_balance",
            row["current_available_balance"],
            minimum=Decimal("0"),
        ),
        minimum_balance_to_keep=_required_decimal(
            path,
            row_number,
            "minimum_balance_to_keep",
            row["minimum_balance_to_keep"],
            minimum=Decimal("0"),
        ),
        financial_priorities=parse_pipe_list(row["financial_priorities"]),
        expense_categories_to_protect=parse_pipe_list(
            row["expense_categories_to_protect"]
        ),
        expense_categories_user_is_willing_to_reduce=parse_pipe_list(
            row["expense_categories_user_is_willing_to_reduce"]
        ),
        expense_categories_user_is_willing_to_stop=parse_pipe_list(
            row["expense_categories_user_is_willing_to_stop"]
        ),
        payment_methods_user_will_consider=tuple(methods),
        max_installment_months=max_months,
    )


def _parse_event(path: Path, row_number: int, row: dict[str, str]) -> FinancialEvent:
    return FinancialEvent(
        event_id=required_text(path, row_number, "event_id", row["event_id"]),
        user_id=required_text(path, row_number, "user_id", row["user_id"]),
        event_type=parse_enum(
            path, row_number, "event_type", row["event_type"], EventType
        ),
        description=required_text(path, row_number, "description", row["description"]),
        category=required_text(path, row_number, "category", row["category"]),
        direction=parse_enum(
            path, row_number, "direction", row["direction"], Direction
        ),
        amount=parse_decimal(
            path,
            row_number,
            "amount",
            row["amount"],
            optional=True,
            minimum=Decimal("0"),
        ),
        currency=parse_enum(path, row_number, "currency", row["currency"], Currency),
        event_date=_required_date(path, row_number, "event_date", row["event_date"]),
        settlement_date=parse_date(
            path, row_number, "settlement_date", row["settlement_date"], optional=True
        ),
        status=parse_enum(path, row_number, "status", row["status"], EventStatus),
        linked_event_id=optional_text(row["linked_event_id"]),
        flexibility=parse_enum(
            path, row_number, "flexibility", row["flexibility"], Flexibility
        ),
        minimum_allowed_amount=parse_decimal(
            path,
            row_number,
            "minimum_allowed_amount",
            row["minimum_allowed_amount"],
            optional=True,
            minimum=Decimal("0"),
        ),
    )


def _parse_rate(path: Path, row_number: int, row: dict[str, str]) -> ExchangeRate:
    from_currency = parse_enum(
        path, row_number, "from_currency", row["from_currency"], Currency
    )
    to_currency = parse_enum(
        path, row_number, "to_currency", row["to_currency"], Currency
    )
    if from_currency == to_currency:
        fail(
            path,
            "exchange-rate direction must use different currencies",
            row_number=row_number,
            column="to_currency",
            value=row["to_currency"],
        )
    return ExchangeRate(
        rate_date=_required_date(path, row_number, "rate_date", row["rate_date"]),
        from_currency=from_currency,
        to_currency=to_currency,
        rate=_required_decimal(path, row_number, "rate", row["rate"], positive=True),
    )


def _parse_request(path: Path, row_number: int, row: dict[str, str]) -> Request:
    request_date = _required_date(path, row_number, "request_date", row["request_date"])
    completion_date = _required_date(
        path, row_number, "desired_completion_date", row["desired_completion_date"]
    )
    if completion_date < request_date:
        fail(
            path,
            "must be on or after request_date",
            row_number=row_number,
            column="desired_completion_date",
            value=row["desired_completion_date"],
        )
    return Request(
        request_id=required_text(path, row_number, "request_id", row["request_id"]),
        user_id=required_text(path, row_number, "user_id", row["user_id"]),
        request_date=request_date,
        request_type=parse_enum(
            path, row_number, "request_type", row["request_type"], RequestType
        ),
        requested_amount=_required_decimal(
            path, row_number, "requested_amount", row["requested_amount"], positive=True
        ),
        desired_completion_date=completion_date,
        allows_partial_payment=parse_bool(
            path, row_number, "allows_partial_payment", row["allows_partial_payment"]
        ),
        request_text=required_text(
            path, row_number, "request_text", row["request_text"]
        ),
    )


def _load_requests(
    path: Path, *, samples: bool, include_sample_outputs: bool
) -> tuple[tuple[Request, ...], tuple[SampleOutput, ...]]:
    headers = SAMPLE_REQUEST_HEADERS if samples else REQUEST_HEADERS
    requests: list[Request] = []
    outputs: list[SampleOutput] = []
    seen: dict[str, int] = {}
    selected = headers if include_sample_outputs or not samples else REQUEST_HEADERS
    for row_number, row in _rows(path, headers, selected_headers=selected):
        request_id = required_text(path, row_number, "request_id", row["request_id"])
        ensure_unique(path, "request_id", request_id, seen, row_number)
        requests.append(_parse_request(path, row_number, row))
        if samples and include_sample_outputs:
            earliest = parse_date(
                path,
                row_number,
                "earliest_date_for_full_payment",
                row["earliest_date_for_full_payment"],
                optional=True,
            )
            outputs.append(
                SampleOutput(
                    request_id=request_id,
                    amount_safe_to_pay=_required_decimal(
                        path,
                        row_number,
                        "amount_safe_to_pay",
                        row["amount_safe_to_pay"],
                        minimum=Decimal("0"),
                    ),
                    affordability_status=parse_enum(
                        path,
                        row_number,
                        "affordability_status",
                        row["affordability_status"],
                        AffordabilityStatus,
                    ),
                    recommended_payment_method=parse_enum(
                        path,
                        row_number,
                        "recommended_payment_method",
                        row["recommended_payment_method"],
                        PaymentMethod,
                    ),
                    payment_plan=required_text(
                        path, row_number, "payment_plan", row["payment_plan"]
                    ),
                    earliest_date_for_full_payment=earliest,
                    spending_changes_needed=required_text(
                        path,
                        row_number,
                        "spending_changes_needed",
                        row["spending_changes_needed"],
                    ),
                    decision_explanation=required_text(
                        path,
                        row_number,
                        "decision_explanation",
                        row["decision_explanation"],
                    ),
                )
            )
    return tuple(requests), tuple(outputs)


def _parse_option(path: Path, row_number: int, row: dict[str, str]) -> PaymentOption:
    method = parse_enum(
        path, row_number, "payment_method", row["payment_method"], PaymentMethod
    )
    if method not in {PaymentMethod.FULL_PAYMENT, PaymentMethod.INSTALLMENTS}:
        fail(
            path,
            "seller option must be full_payment or installments",
            row_number=row_number,
            column="payment_method",
            value=row["payment_method"],
        )
    payments = _required_int(
        path, row_number, "number_of_payments", row["number_of_payments"], minimum=1
    )
    frequency = parse_int(
        path,
        row_number,
        "payment_frequency_days",
        row["payment_frequency_days"],
        optional=True,
        minimum=1,
    )
    if payments == 1 and frequency is not None:
        fail(
            path,
            "must be blank for a one-payment option",
            row_number=row_number,
            column="payment_frequency_days",
            value=row["payment_frequency_days"],
        )
    if payments > 1 and frequency is None:
        fail(
            path,
            "is required for a multi-payment option",
            row_number=row_number,
            column="payment_frequency_days",
        )
    if method is PaymentMethod.FULL_PAYMENT and payments != 1:
        fail(
            path,
            "full_payment must have exactly one payment",
            row_number=row_number,
            column="number_of_payments",
            value=row["number_of_payments"],
        )
    if method is PaymentMethod.INSTALLMENTS and payments < 2:
        fail(
            path,
            "installments must have at least two payments",
            row_number=row_number,
            column="number_of_payments",
            value=row["number_of_payments"],
        )
    return PaymentOption(
        payment_option_id=required_text(
            path, row_number, "payment_option_id", row["payment_option_id"]
        ),
        request_id=required_text(path, row_number, "request_id", row["request_id"]),
        payment_method=method,
        payment_amount=_required_decimal(
            path, row_number, "payment_amount", row["payment_amount"], positive=True
        ),
        number_of_payments=payments,
        first_payment_date=_required_date(
            path, row_number, "first_payment_date", row["first_payment_date"]
        ),
        payment_frequency_days=frequency,
        financing_fee=_required_decimal(
            path,
            row_number,
            "financing_fee",
            row["financing_fee"],
            minimum=Decimal("0"),
        ),
        total_payable_amount=_required_decimal(
            path,
            row_number,
            "total_payable_amount",
            row["total_payable_amount"],
            positive=True,
        ),
    )


def _parse_message(path: Path, row_number: int, row: dict[str, str]) -> MessageRecord:
    return MessageRecord(
        message_id=required_text(path, row_number, "message_id", row["message_id"]),
        user_id=required_text(path, row_number, "user_id", row["user_id"]),
        request_id=optional_text(row["request_id"]),
        related_event_id=optional_text(row["related_event_id"]),
        sent_at=parse_datetime(path, row_number, "sent_at", row["sent_at"]),
        source_type=parse_enum(
            path, row_number, "source_type", row["source_type"], SourceType
        ),
        message_text=required_text(
            path, row_number, "message_text", row["message_text"]
        ),
    )


def _parse_image(
    dataset_dir: Path,
) -> Callable[[Path, int, dict[str, str]], ImageRecord]:
    def parse(path: Path, row_number: int, row: dict[str, str]) -> ImageRecord:
        image_id = required_text(path, row_number, "image_id", row["image_id"])
        image_path = dataset_dir / "media" / "images" / f"{image_id}.png"
        if not image_path.is_file():
            fail(
                path,
                f"referenced image file does not exist: {image_path}",
                row_number=row_number,
                column="image_id",
                value=image_id,
            )
        return ImageRecord(
            image_id=image_id,
            user_id=required_text(path, row_number, "user_id", row["user_id"]),
            request_id=optional_text(row["request_id"]),
            related_event_id=optional_text(row["related_event_id"]),
            file_path=image_path,
        )

    return parse


def _validate_relationships(dataset_dir: Path, dataset: Dataset) -> None:
    profile_ids = {profile.user_id for profile in dataset.profiles}
    requests = dataset.evaluation_requests + dataset.sample_requests
    requests_by_id = {request.request_id: request for request in requests}
    event_by_id = {event.event_id: event for event in dataset.financial_events}
    event_ids = set(event_by_id)
    event_positions = {
        event.event_id: position
        for position, event in enumerate(dataset.financial_events)
    }

    for event in dataset.financial_events:
        require_foreign_key(
            dataset_dir / "financial_events.csv",
            None,
            "user_id",
            event.user_id,
            profile_ids,
            "profile user_id",
        )
        if event.linked_event_id is not None:
            require_foreign_key(
                dataset_dir / "financial_events.csv",
                None,
                "linked_event_id",
                event.linked_event_id,
                event_ids,
                "financial event_id",
            )
            linked = event_by_id.get(event.linked_event_id)
            if linked is not None and linked.user_id != event.user_id:
                fail(
                    dataset_dir / "financial_events.csv",
                    f"cross-user event link from {event.user_id} to {linked.user_id}",
                    column="linked_event_id",
                    value=event.linked_event_id,
                )
            if (
                linked is not None
                and event_positions[linked.event_id] >= event_positions[event.event_id]
            ):
                fail(
                    dataset_dir / "financial_events.csv",
                    f"linked_event_id on {event.event_id} must reference an earlier row",
                    column="linked_event_id",
                    value=event.linked_event_id,
                )

    for request in requests:
        require_foreign_key(
            dataset_dir
            / (
                "sample_requests.csv"
                if request in dataset.sample_requests
                else "requests.csv"
            ),
            None,
            "user_id",
            request.user_id,
            profile_ids,
            "profile user_id",
        )

    request_ids = set(requests_by_id)
    for option in dataset.payment_options:
        require_foreign_key(
            dataset_dir / "request_payment_options.csv",
            None,
            "request_id",
            option.request_id,
            request_ids,
            "request_id",
        )
    options_per_request: dict[str, int] = defaultdict(int)
    for option in dataset.payment_options:
        options_per_request[option.request_id] += 1
    for request in requests:
        option_count = options_per_request[request.request_id]
        if not 2 <= option_count <= 4:
            fail(
                dataset_dir / "request_payment_options.csv",
                f"request {request.request_id!r} must have 2 to 4 payment options; found {option_count}",
                column="request_id",
                value=request.request_id,
            )

    def validate_evidence(
        path: Path,
        record_id: str,
        user_id: str,
        request_id: str | None,
        event_id: str | None,
    ) -> None:
        require_foreign_key(
            path, None, "user_id", user_id, profile_ids, "profile user_id"
        )
        require_foreign_key(
            path, None, "request_id", request_id, request_ids, "request_id"
        )
        require_foreign_key(
            path, None, "related_event_id", event_id, event_ids, "financial event_id"
        )
        if request_id is not None and requests_by_id[request_id].user_id != user_id:
            fail(
                path,
                f"cross-user evidence leakage in {record_id}: request belongs to {requests_by_id[request_id].user_id}, record belongs to {user_id}",
                column="request_id",
                value=request_id,
            )
        if event_id is not None and event_by_id[event_id].user_id != user_id:
            fail(
                path,
                f"cross-user evidence leakage in {record_id}: event belongs to {event_by_id[event_id].user_id}, record belongs to {user_id}",
                column="related_event_id",
                value=event_id,
            )

    for message in dataset.messages:
        validate_evidence(
            dataset_dir / "messages.csv",
            message.message_id,
            message.user_id,
            message.request_id,
            message.related_event_id,
        )
    for image in dataset.images:
        validate_evidence(
            dataset_dir / "images.csv",
            image.image_id,
            image.user_id,
            image.request_id,
            image.related_event_id,
        )

    image_event_ids = {
        image.related_event_id
        for image in dataset.images
        if image.related_event_id is not None
    }
    for event in dataset.financial_events:
        if event.amount is None and event.event_id not in image_event_ids:
            fail(
                dataset_dir / "financial_events.csv",
                "blank amount must be linked from images.csv for later extraction",
                column="amount",
                value=event.event_id,
            )


def load_dataset(
    dataset_dir: Path | str, *, include_sample_outputs: bool = True
) -> Dataset:
    """Load and validate all participant-facing input files without modifying them."""

    root = Path(dataset_dir).resolve()
    if not root.is_dir():
        fail(root, "dataset directory does not exist")

    profiles = _load_with_unique_id(
        root / "financial_profiles.csv", PROFILE_HEADERS, "user_id", _parse_profile
    )
    events = _load_with_unique_id(
        root / "financial_events.csv", FINANCIAL_EVENT_HEADERS, "event_id", _parse_event
    )

    rates: list[ExchangeRate] = []
    rate_keys: dict[str, int] = {}
    for row_number, row in _rows(root / "exchange_rates.csv", EXCHANGE_RATE_HEADERS):
        rate = _parse_rate(root / "exchange_rates.csv", row_number, row)
        key = f"{rate.rate_date.isoformat()}:{rate.from_currency.value}:{rate.to_currency.value}"
        ensure_unique(
            root / "exchange_rates.csv",
            "rate_date/from_currency/to_currency",
            key,
            rate_keys,
            row_number,
        )
        rates.append(rate)

    evaluation_requests, _ = _load_requests(
        root / "requests.csv", samples=False, include_sample_outputs=False
    )
    sample_requests, sample_outputs = _load_requests(
        root / "sample_requests.csv",
        samples=True,
        include_sample_outputs=include_sample_outputs,
    )
    eval_ids = {request.request_id for request in evaluation_requests}
    overlap = next(
        (
            request.request_id
            for request in sample_requests
            if request.request_id in eval_ids
        ),
        None,
    )
    if overlap is not None:
        fail(
            root / "sample_requests.csv",
            "request_id must not overlap evaluation requests",
            column="request_id",
            value=overlap,
        )

    options = _load_with_unique_id(
        root / "request_payment_options.csv",
        PAYMENT_OPTION_HEADERS,
        "payment_option_id",
        _parse_option,
    )
    messages = _load_with_unique_id(
        root / "messages.csv", MESSAGE_HEADERS, "message_id", _parse_message
    )
    images = _load_with_unique_id(
        root / "images.csv", IMAGE_HEADERS, "image_id", _parse_image(root)
    )
    dataset = Dataset(
        profiles=profiles,
        financial_events=events,
        exchange_rates=tuple(rates),
        evaluation_requests=evaluation_requests,
        sample_requests=sample_requests,
        sample_outputs=sample_outputs,
        payment_options=options,
        messages=messages,
        images=images,
    )
    _validate_relationships(root, dataset)
    return dataset


def build_request_contexts(dataset: Dataset) -> tuple[RequestContext, ...]:
    """Join evaluation requests only, preserving every source file's row order."""

    profiles = {profile.user_id: profile for profile in dataset.profiles}
    events_by_user: dict[str, list[FinancialEvent]] = defaultdict(list)
    options_by_request: dict[str, list[PaymentOption]] = defaultdict(list)
    for event in dataset.financial_events:
        events_by_user[event.user_id].append(event)
    for option in dataset.payment_options:
        options_by_request[option.request_id].append(option)

    contexts: list[RequestContext] = []
    for request in dataset.evaluation_requests:
        profile = profiles.get(request.user_id)
        if profile is None:
            raise InputValidationError(
                Path("financial_profiles.csv"),
                f"request {request.request_id!r} has no profile for {request.user_id!r}",
            )
        user_events = tuple(events_by_user[request.user_id])
        event_ids = {event.event_id for event in user_events}
        messages = tuple(
            message
            for message in dataset.messages
            if message.user_id == request.user_id
            and (
                message.request_id is None
                or message.request_id == request.request_id
                or message.related_event_id in event_ids
            )
        )
        images = tuple(
            image
            for image in dataset.images
            if image.user_id == request.user_id
            and (
                image.request_id is None
                or image.request_id == request.request_id
                or image.related_event_id in event_ids
            )
        )
        contexts.append(
            RequestContext(
                request=request,
                profile=profile,
                financial_events=user_events,
                payment_options=tuple(options_by_request[request.request_id]),
                messages=messages,
                images=images,
                exchange_rates=dataset.exchange_rates,
            )
        )
    return tuple(contexts)


def load_request_contexts(dataset_dir: Path | str) -> tuple[RequestContext, ...]:
    """Convenience API that cannot expose solved sample labels to prediction code."""

    return build_request_contexts(
        load_dataset(dataset_dir, include_sample_outputs=False)
    )


__all__ = [
    "CSV_SCHEMAS",
    "InputValidationError",
    "build_request_contexts",
    "load_dataset",
    "load_request_contexts",
]
