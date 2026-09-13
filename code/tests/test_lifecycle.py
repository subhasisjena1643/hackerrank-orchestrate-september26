"""Phase 7 financial-event lifecycle coverage."""

from datetime import date, datetime, timezone
from decimal import Decimal

import pytest

from buy_or_wait.domain import (
    Currency,
    Direction,
    EventStatus,
    EventType,
    ExchangeRate,
    FinancialEvent,
    Flexibility,
    SourceType,
)
from buy_or_wait.evidence.messages import ExtractedMessageEvidence
from buy_or_wait.evidence.schemas import MessageEvidenceItem
from buy_or_wait.finance.lifecycle import (
    LifecycleResolutionError,
    ResolutionDisposition,
    ResolutionReason,
    resolve_event_lifecycles,
)


SNAPSHOT = date(2026, 9, 10)


def event(
    event_id: str,
    *,
    status: EventStatus = EventStatus.SETTLED,
    direction: Direction = Direction.DEBIT,
    amount: str | None = "10",
    when: date | None = date(2026, 9, 1),
    event_type: EventType = EventType.EXPENSE,
    linked: str | None = None,
    description: str = "Card purchase",
    category: str = "shopping",
    currency: Currency = Currency.USD,
) -> FinancialEvent:
    return FinancialEvent(
        event_id,
        "u1",
        event_type,
        description,
        category,
        direction,
        None if amount is None else Decimal(amount),
        currency,
        when or SNAPSHOT,
        when,
        status,
        linked,
        Flexibility.FIXED,
        None,
    )


def evidence(
    message_id: str,
    event_id: str,
    fact_type: str,
    *,
    source: SourceType = SourceType.BANK,
    sent: datetime = datetime(2026, 9, 5, tzinfo=timezone.utc),
    amount: str | None = None,
    currency: str | None = None,
    effective: date | None = None,
    settlement: date | None = None,
) -> ExtractedMessageEvidence:
    cash_state = {
        "event_cancelled": "cancelled",
        "event_settled": "confirmed_debit",
        "refund_pending": "pending_credit",
        "refund_settled": "confirmed_credit",
        "internal_transfer": "non_cash",
        "first_salary_confirmed": "confirmed_credit",
        "unconfirmed_income": "pending_credit",
    }.get(fact_type)
    item = MessageEvidenceItem(
        message_id=message_id,
        related_event_id=event_id,
        fact_type=fact_type,
        amount=amount,
        currency=currency,
        effective_date=effective,
        settlement_date=settlement,
        recurrence_scope="one_cycle" if fact_type == "temporary_salary" else None,
        cash_state=cash_state,
        confidence="high",
        evidence_quote=None,
        notes="test",
    )
    return ExtractedMessageEvidence(item, sent, source, "test")


def resolve(rows, *, facts=(), rates=(), home=Currency.USD, **kwargs):
    return resolve_event_lifecycles(
        rows,
        snapshot_date=SNAPSHOT,
        home_currency=home,
        exchange_rates=rates,
        message_evidence=facts,
        **kwargs,
    )


@pytest.mark.parametrize(
    ("status", "direction", "kind", "reason", "projected"),
    [
        (
            EventStatus.CANCELLED,
            Direction.DEBIT,
            EventType.EXPENSE,
            ResolutionReason.CANCELLED,
            "0",
        ),
        (
            EventStatus.FAILED,
            Direction.DEBIT,
            EventType.EXPENSE,
            ResolutionReason.FAILED,
            "0",
        ),
        (
            EventStatus.PENDING,
            Direction.CREDIT,
            EventType.REFUND,
            ResolutionReason.PENDING_CREDIT,
            "0",
        ),
        (
            EventStatus.PENDING,
            Direction.DEBIT,
            EventType.EXPENSE,
            ResolutionReason.PENDING_DEBIT,
            "-10",
        ),
        (
            EventStatus.SCHEDULED,
            Direction.DEBIT,
            EventType.EXPENSE,
            ResolutionReason.SCHEDULED_DEBIT,
            "-10",
        ),
        (
            EventStatus.SETTLED,
            Direction.DEBIT,
            EventType.EXPENSE,
            ResolutionReason.HISTORICAL_SETTLED,
            "0",
        ),
        (
            EventStatus.UNREALIZED,
            Direction.NON_CASH,
            EventType.INVESTMENT_VALUATION,
            ResolutionReason.UNREALIZED,
            "0",
        ),
    ],
)
def test_every_status(status, direction, kind, reason, projected):
    row = event("e", status=status, direction=direction, event_type=kind)
    result = resolve([row])[0]
    assert result.reason_code is reason
    assert result.projected_cash_effect == Decimal(projected)


def test_snapshot_is_not_replayed_but_history_can_support_recurrence():
    result = resolve([event("e")])[0]
    assert result.signed_cash_effect == Decimal("-10")
    assert result.projected_cash_effect == 0
    assert result.disposition is ResolutionDisposition.HISTORICAL_CASH_EVIDENCE
    assert result.supports_recurrence


def test_cancelled_and_failed_never_teach_recurrence():
    results = resolve(
        [
            event("c", status=EventStatus.CANCELLED),
            event("f", status=EventStatus.FAILED),
        ]
    )
    assert all(
        not row.supports_recurrence and row.signed_cash_effect == 0 for row in results
    )


def test_scheduled_salary_is_confirmed_and_uses_settlement_date():
    row = event(
        "salary",
        status=EventStatus.SCHEDULED,
        direction=Direction.CREDIT,
        event_type=EventType.INCOME,
        category="salary",
        description="Next confirmed salary",
        when=date(2026, 9, 15),
    )
    result = resolve([row])[0]
    assert result.reason_code is ResolutionReason.CONFIRMED_SALARY
    assert result.cash_flow_date == date(2026, 9, 15)
    assert result.projected_cash_effect == 10


def test_unconfirmed_scheduled_credit_is_zero():
    row = event(
        "bonus",
        status=EventStatus.SCHEDULED,
        direction=Direction.CREDIT,
        event_type=EventType.INCOME,
        description="Possible bonus",
        when=date(2026, 9, 15),
    )
    result = resolve([row])[0]
    assert result.reason_code is ResolutionReason.UNCONFIRMED_CREDIT
    assert result.projected_cash_effect == 0


def test_cancelled_authorization_and_settled_purchase_count_purchase_once():
    auth = event(
        "auth",
        status=EventStatus.CANCELLED,
        when=date(2026, 9, 1),
        description="Card authorization",
    )
    purchase = event(
        "purchase",
        linked="auth",
        when=date(2026, 9, 2),
        description="Settled card purchase",
    )
    results = resolve([auth, purchase])
    assert [r.signed_cash_effect for r in results] == [Decimal(0), Decimal("-10")]


def test_settled_purchase_and_refund_retain_both_real_effects():
    purchase = event("purchase")
    refund = event(
        "refund",
        direction=Direction.CREDIT,
        event_type=EventType.REFUND,
        linked="purchase",
        description="Settled refund",
    )
    results = resolve([purchase, refund])
    assert [r.signed_cash_effect for r in results] == [Decimal("-10"), Decimal("10")]


def test_link_alone_does_not_prove_duplication():
    first = event("first")
    second = event(
        "second",
        status=EventStatus.PENDING,
        linked="first",
        when=date(2026, 9, 12),
        description="Second card charge",
    )
    assert resolve([first, second])[1].projected_cash_effect == Decimal("-10")


def test_verified_own_account_transfer_nets_linked_lifecycle_and_no_recurrence():
    debit = event("out", description="Transfer out")
    credit = event(
        "in", direction=Direction.CREDIT, linked="out", description="Transfer in"
    )
    results = resolve(
        [debit, credit], facts=[evidence("m", "out", "internal_transfer")]
    )
    assert all(r.reason_code is ResolutionReason.INTERNAL_TRANSFER for r in results)
    assert all(r.signed_cash_effect == 0 and not r.supports_recurrence for r in results)


def test_proven_duplicate_is_suppressed_once():
    first = event("first", status=EventStatus.SCHEDULED, when=date(2026, 9, 12))
    duplicate = event(
        "duplicate",
        status=EventStatus.SCHEDULED,
        linked="first",
        when=date(2026, 9, 12),
    )
    result = resolve([first, duplicate], proven_duplicate_event_ids={"duplicate"})
    assert result[0].projected_cash_effect == -10
    assert result[1].reason_code is ResolutionReason.PROVEN_DUPLICATE
    assert result[1].projected_cash_effect == 0


def test_possible_duplicate_uses_safer_debit_interpretation():
    first = event("first")
    possible = event(
        "possible",
        status=EventStatus.PENDING,
        linked="first",
        when=date(2026, 9, 12),
        description="Possible duplicate card charge",
    )
    result = resolve([first, possible])[1]
    assert result.reason_code is ResolutionReason.UNRESOLVED_DUPLICATE
    assert result.projected_cash_effect == -10
    assert not result.supports_recurrence


def test_message_amount_and_date_amendments_apply_without_inference():
    salary = event(
        "salary",
        status=EventStatus.SCHEDULED,
        direction=Direction.CREDIT,
        event_type=EventType.INCOME,
        category="salary",
        description="Next confirmed salary",
        when=date(2026, 9, 15),
        amount="100",
    )
    facts = [
        evidence(
            "m1",
            "salary",
            "salary_amount_amendment",
            amount="120",
            currency="USD",
            effective=date(2026, 9, 1),
        ),
        evidence("m2", "salary", "salary_date_amendment", effective=date(2026, 9, 20)),
    ]
    result = resolve([salary], facts=facts)[0]
    assert (result.effective_amount, result.cash_flow_date) == (
        Decimal("120"),
        date(2026, 9, 20),
    )
    assert result.reason_code is ResolutionReason.EXPLICIT_AMENDMENT
    assert result.projected_cash_effect == 120


def test_newer_same_source_wins_and_input_order_does_not_matter():
    row = event("e", status=EventStatus.SCHEDULED, when=date(2026, 9, 12))
    older = evidence(
        "m1", "e", "event_cancelled", sent=datetime(2026, 9, 1, tzinfo=timezone.utc)
    )
    newer = evidence(
        "m2",
        "e",
        "event_settled",
        sent=datetime(2026, 9, 2, tzinfo=timezone.utc),
        settlement=date(2026, 9, 12),
    )
    a = resolve([row], facts=[older, newer])[0]
    b = resolve([row], facts=[newer, older])[0]
    assert a == b
    assert a.effective_status is EventStatus.SETTLED
    assert a.reason_code is ResolutionReason.NEWER_SAME_SOURCE


@pytest.mark.parametrize(
    ("fact_type", "expected", "reason"),
    [
        (
            "event_cancelled",
            EventStatus.CANCELLED,
            ResolutionReason.EXPLICIT_CANCELLATION,
        ),
        ("event_settled", EventStatus.SETTLED, ResolutionReason.EXPLICIT_SETTLEMENT),
    ],
)
def test_explicit_message_status_overrides_schedule(fact_type, expected, reason):
    row = event("e", status=EventStatus.SCHEDULED, when=date(2026, 9, 12))
    result = resolve([row], facts=[evidence("m", "e", fact_type)])[0]
    assert result.effective_status is expected
    assert result.reason_code is reason


def test_settled_fact_beats_pending_estimate():
    row = event("refund", direction=Direction.CREDIT, event_type=EventType.REFUND)
    result = resolve(
        [row],
        facts=[evidence("m", "refund", "refund_pending", source=SourceType.MERCHANT)],
    )[0]
    assert result.effective_status is EventStatus.SETTLED
    assert result.reason_code is ResolutionReason.SETTLED_OVER_ESTIMATE


@pytest.mark.parametrize(
    ("direction", "expected"),
    [(Direction.DEBIT, EventStatus.SETTLED), (Direction.CREDIT, EventStatus.CANCELLED)],
)
def test_unresolved_cross_source_conflict_is_financially_safer(direction, expected):
    row = event(
        "e", status=EventStatus.SCHEDULED, direction=direction, when=date(2026, 9, 12)
    )
    facts = [
        evidence("cancel", "e", "event_cancelled", source=SourceType.MERCHANT),
        evidence("settle", "e", "event_settled", source=SourceType.BANK),
    ]
    result = resolve([row], facts=facts)[0]
    assert result.effective_status is expected
    assert result.reason_code is ResolutionReason.SAFER_INTERPRETATION


def test_cross_currency_uses_exact_settlement_date_rate():
    row = event(
        "fx",
        status=EventStatus.PENDING,
        when=date(2026, 9, 12),
        amount="10",
        currency=Currency.USD,
    )
    rate = ExchangeRate(date(2026, 9, 12), Currency.USD, Currency.EUR, Decimal("0.9"))
    result = resolve([row], rates=[rate], home=Currency.EUR)[0]
    assert result.amount_home_currency == Decimal("9.0")
    assert result.projected_cash_effect == Decimal("-9.0")
    assert any(p.kind.value == "exchange_rate" for p in result.provenance)


def test_unrealized_valuation_is_non_cash_but_settled_sale_is_actual_cash():
    purchase = event("buy", event_type=EventType.INVESTMENT_PURCHASE)
    valuation = event(
        "value",
        status=EventStatus.UNREALIZED,
        direction=Direction.NON_CASH,
        event_type=EventType.INVESTMENT_VALUATION,
        linked="buy",
    )
    sale = event(
        "sale",
        direction=Direction.CREDIT,
        event_type=EventType.INVESTMENT_SALE,
        linked="buy",
    )
    results = resolve([purchase, valuation, sale])
    assert results[1].signed_cash_effect == 0
    assert results[2].signed_cash_effect == 10
    assert results[2].reason_code is ResolutionReason.SETTLED_INVESTMENT_SALE
    assert not results[2].supports_recurrence


def test_blank_cash_amount_requires_prior_image_resolution():
    with pytest.raises(LifecycleResolutionError, match="prior evidence"):
        resolve(
            [
                event(
                    "e", status=EventStatus.PENDING, amount=None, when=date(2026, 9, 12)
                )
            ]
        )
