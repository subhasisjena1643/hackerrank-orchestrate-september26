"""Phase 8 future cash-flow construction coverage."""

from dataclasses import replace
from datetime import date
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
)
from buy_or_wait.evidence.schemas import MessageEvidenceItem
from buy_or_wait.finance.cashflows import (
    CashFlowReason,
    construct_future_cash_flows,
)
from buy_or_wait.finance.lifecycle import resolve_event_lifecycles
from buy_or_wait.finance.recurrence_policies import BASELINE_POLICY, SameDayOrdering
from buy_or_wait.domain import Payment
from buy_or_wait.finance.forecast import (
    calculate_baseline_forecast,
    optimized_safe_payment_on_date,
    replay_forecast,
)


REQUEST = date(2026, 4, 1)


def event(
    event_id,
    when,
    amount="100",
    *,
    status=EventStatus.SETTLED,
    direction=Direction.DEBIT,
    category="rent",
    description="Rent",
    kind=EventType.EXPENSE,
):
    return FinancialEvent(
        event_id,
        "u",
        kind,
        description,
        category,
        direction,
        Decimal(amount),
        Currency.USD,
        when,
        when,
        status,
        None,
        Flexibility.FIXED,
        None,
    )


def resolve(rows):
    return resolve_event_lifecycles(
        rows, snapshot_date=REQUEST, home_currency=Currency.USD
    )


def history():
    return [event(f"r{i}", date(2026, i, 10)) for i in (1, 2, 3)]


def test_explicit_future_row_suppresses_inferred_occurrence():
    rows = history() + [
        event("scheduled", date(2026, 4, 10), "120", status=EventStatus.SCHEDULED)
    ]
    projection = construct_future_cash_flows(
        resolve(rows), request_date=REQUEST, policy=BASELINE_POLICY
    )
    april = [
        flow for flow in projection.cash_flows if flow.flow_date == date(2026, 4, 10)
    ]
    assert len(april) == 1
    assert april[0].amount == Decimal("120") and not april[0].synthetic
    assert april[0].reason_code == CashFlowReason.EXPLICIT_SUPPRESSES_INFERRED
    assert len(projection.suppressed) == 1


def test_confirmed_salary_anchor_is_not_double_counted_despite_description_change():
    rows = [
        event(
            "first",
            date(2026, 3, 15),
            "60",
            status=EventStatus.SETTLED,
            direction=Direction.CREDIT,
            category="salary",
            description="Prorated first salary",
            kind=EventType.INCOME,
        ),
        event(
            "next",
            date(2026, 4, 15),
            "100",
            status=EventStatus.SCHEDULED,
            direction=Direction.CREDIT,
            category="salary",
            description="Next confirmed salary",
            kind=EventType.INCOME,
        ),
    ]
    projection = construct_future_cash_flows(
        resolve(rows), request_date=REQUEST, policy=BASELINE_POLICY, horizon_days=60
    )
    april = [
        flow for flow in projection.cash_flows if flow.flow_date == date(2026, 4, 15)
    ]
    assert len(april) == 1
    assert april[0].cash_flow_id == "explicit:next"
    assert len(projection.suppressed) == 1


def test_authoritative_pending_debit_and_confirmed_salary_are_included():
    rows = [
        event("pending", date(2026, 4, 5), "25", status=EventStatus.PENDING),
        event(
            "salary",
            date(2026, 4, 15),
            "500",
            status=EventStatus.SCHEDULED,
            direction=Direction.CREDIT,
            category="salary",
            description="Confirmed salary",
            kind=EventType.INCOME,
        ),
        event(
            "refund",
            date(2026, 4, 7),
            "30",
            status=EventStatus.PENDING,
            direction=Direction.CREDIT,
            category="shopping",
            kind=EventType.REFUND,
        ),
    ]
    flows = construct_future_cash_flows(
        resolve(rows), request_date=REQUEST, policy=BASELINE_POLICY
    ).cash_flows
    assert [(x.cash_flow_id, x.signed_amount) for x in flows] == [
        ("explicit:pending", Decimal("-25")),
        ("explicit:salary", Decimal("500")),
    ]


def test_message_only_confirmed_salary_becomes_one_dated_credit():
    fact = MessageEvidenceItem(
        message_id="message_salary",
        related_event_id=None,
        fact_type="first_salary_confirmed",
        amount="100",
        currency="EUR",
        effective_date=None,
        settlement_date=date(2026, 4, 15),
        recurrence_scope="recurring",
        cash_state="confirmed_credit",
        confidence="high",
        evidence_quote="first salary will be EUR 100",
        notes="Explicit confirmed first salary.",
    )
    projection = construct_future_cash_flows(
        (),
        request_date=REQUEST,
        policy=BASELINE_POLICY,
        message_evidence=(fact,),
        user_id="u",
        home_currency=Currency.USD,
        exchange_rates=(
            ExchangeRate(date(2026, 4, 15), Currency.EUR, Currency.USD, Decimal("1.1")),
        ),
    )
    assert len(projection.cash_flows) == 1
    flow = projection.cash_flows[0]
    assert flow.cash_flow_id == "message-confirmed:message_salary"
    assert flow.flow_date == date(2026, 4, 15)
    assert flow.amount == Decimal("110.0")
    assert flow.reason_code == CashFlowReason.AUTHORITATIVE_MESSAGE_CREDIT


def test_horizon_is_inclusive_and_provenance_is_retained():
    rows = history()
    projection = construct_future_cash_flows(
        resolve(rows), request_date=REQUEST, policy=BASELINE_POLICY
    )
    assert projection.horizon_end == date(2026, 6, 30)
    assert projection.cash_flows[-1].flow_date == date(2026, 6, 10)
    assert all(flow.reason_codes and flow.provenance for flow in projection.cash_flows)


def test_same_day_ordering_is_global_policy():
    rows = [
        event("debit", date(2026, 4, 5), status=EventStatus.SCHEDULED),
        event(
            "credit",
            date(2026, 4, 5),
            status=EventStatus.SCHEDULED,
            direction=Direction.CREDIT,
            category="salary",
            description="Confirmed salary",
            kind=EventType.INCOME,
        ),
    ]
    resolved = resolve(rows)
    debit_first = construct_future_cash_flows(
        resolved, request_date=REQUEST, policy=BASELINE_POLICY
    ).cash_flows
    credit_first = construct_future_cash_flows(
        resolved,
        request_date=REQUEST,
        policy=replace(
            BASELINE_POLICY, same_day_ordering=SameDayOrdering.CREDITS_BEFORE_DEBITS
        ),
    ).cash_flows
    assert [x.direction for x in debit_first] == [Direction.DEBIT, Direction.CREDIT]
    assert [x.direction for x in credit_first] == [Direction.CREDIT, Direction.DEBIT]


def test_cash_flow_production_requires_selected_policy():
    with pytest.raises(ValueError, match="explicit selected"):
        construct_future_cash_flows((), request_date=REQUEST, policy=None)


def flow(flow_id, when, amount, direction=Direction.DEBIT):
    source = event(
        flow_id,
        when,
        str(amount),
        status=EventStatus.SCHEDULED,
        direction=direction,
        category="salary" if direction is Direction.CREDIT else "rent",
        description="Confirmed salary" if direction is Direction.CREDIT else "Rent",
        kind=EventType.INCOME if direction is Direction.CREDIT else EventType.EXPENSE,
    )
    return construct_future_cash_flows(
        resolve([source]), request_date=REQUEST, policy=BASELINE_POLICY
    ).cash_flows[0]


def baseline(balance="100", minimum="20", requested="50", flows=(), horizon=90):
    return calculate_baseline_forecast(
        current_available_balance=Decimal(balance),
        minimum_balance_to_keep=Decimal(minimum),
        request_date=REQUEST,
        requested_amount=Decimal(requested),
        cash_flows=flows,
        policy=BASELINE_POLICY,
        horizon_days=horizon,
    )


def test_safe_full_payment_today_and_capped_safe_amount():
    result = baseline(balance="100", minimum="20", requested="50")
    assert result.amount_safe_to_pay == Decimal("50")
    assert result.earliest_date_for_full_payment == REQUEST
    assert len(result.ledger) == 91


@pytest.mark.parametrize(
    ("balance", "debit", "expected"),
    [
        ("20", "0", "0"),
        ("100", "50", "30"),
        ("100", "10", "50"),
    ],
)
def test_safe_amount_zero_partial_and_capped(balance, debit, expected):
    flows = () if debit == "0" else (flow("future", REQUEST.replace(day=2), debit),)
    assert baseline(balance=balance, flows=flows).amount_safe_to_pay == Decimal(
        expected
    )


def test_salary_settlement_creates_later_safe_date():
    result = baseline(
        balance="50",
        minimum="20",
        requested="50",
        flows=(flow("salary2", date(2026, 4, 10), "30", Direction.CREDIT),),
    )
    assert result.earliest_date_for_full_payment == date(2026, 4, 10)


def test_pending_debit_lowers_capacity():
    result = baseline(flows=(flow("pending2", date(2026, 4, 5), "40"),))
    assert result.amount_safe_to_pay == Decimal("40")


def test_day_90_is_inclusive_and_can_fail():
    last = REQUEST + __import__("datetime").timedelta(days=90)
    result = baseline(
        balance="100", minimum="20", requested="50", flows=(flow("last", last, "40"),)
    )
    assert result.amount_safe_to_pay == Decimal("40")
    replayed = replay_forecast(
        starting_balance=Decimal("100"),
        minimum_balance_to_keep=Decimal("20"),
        request_date=REQUEST,
        cash_flows=(flow("last2", last, "40"),),
        payments=(Payment(REQUEST, Decimal("50")),),
        policy=BASELINE_POLICY,
    )
    assert not replayed.is_safe and replayed.failure_date == last


def test_same_day_debit_payment_credit_order_checks_each_step():
    rows = (
        flow("d", date(2026, 4, 5), "90"),
        flow("c", date(2026, 4, 5), "100", Direction.CREDIT),
    )
    replayed = replay_forecast(
        starting_balance=Decimal("100"),
        minimum_balance_to_keep=Decimal("20"),
        request_date=REQUEST,
        cash_flows=rows,
        payments=(Payment(date(2026, 4, 5), Decimal("30")),),
        policy=BASELINE_POLICY,
    )
    entries = replayed.ledger[4].entries
    assert [entry.entry_id for entry in entries] == [
        "explicit:d",
        "explicit:c",
        "payment:0",
    ]
    assert not replayed.is_safe
    evaluation_policy = replace(
        BASELINE_POLICY, same_day_ordering=SameDayOrdering.CREDITS_BEFORE_DEBITS
    )
    alternative = replay_forecast(
        starting_balance=Decimal("100"),
        minimum_balance_to_keep=Decimal("20"),
        request_date=REQUEST,
        cash_flows=rows,
        payments=(Payment(date(2026, 4, 5), Decimal("30")),),
        policy=evaluation_policy,
    )
    assert [entry.entry_id for entry in alternative.ledger[4].entries] == [
        "explicit:c",
        "explicit:d",
        "payment:0",
    ]
    assert alternative.is_safe


def test_no_earliest_date():
    assert (
        baseline(
            balance="30", minimum="20", requested="50"
        ).earliest_date_for_full_payment
        is None
    )


def test_decimal_precision_for_large_idr_values():
    result = baseline(
        balance="9999999999999999.99",
        minimum="123456789.01",
        requested="9000000000000000.12",
    )
    assert result.amount_safe_to_pay == Decimal("9000000000000000.12")


def test_optimized_capacity_matches_independent_replay():
    rows = (flow("rent2", date(2026, 4, 5), "40"),)
    result = baseline(flows=rows)
    optimized = optimized_safe_payment_on_date(
        baseline_replay=result.baseline_replay,
        payment_date=REQUEST,
        minimum_balance_to_keep=Decimal("20"),
        requested_amount=Decimal("50"),
    )
    replayed = replay_forecast(
        starting_balance=Decimal("100"),
        minimum_balance_to_keep=Decimal("20"),
        request_date=REQUEST,
        cash_flows=rows,
        payments=(Payment(REQUEST, optimized),),
        policy=BASELINE_POLICY,
    )
    assert optimized == result.amount_safe_to_pay and replayed.is_safe
