"""Phase 8 recurrence and forecast-policy coverage."""

from dataclasses import replace
from datetime import date, datetime, timezone
from decimal import Decimal
import hashlib

import pytest

from buy_or_wait.domain import (
    Currency,
    Direction,
    EventStatus,
    EventType,
    FinancialEvent,
    Flexibility,
    SourceType,
)
from buy_or_wait.evidence.messages import ExtractedMessageEvidence
from buy_or_wait.evidence.schemas import MessageEvidenceItem
from buy_or_wait.finance.lifecycle import resolve_event_lifecycles
from buy_or_wait.finance.recurrence import (
    Cadence,
    detect_cadence,
    infer_recurrence_series,
    project_recurrence_series,
)
from buy_or_wait.finance.recurrence_policies import (
    Aggregation,
    AmountEstimator,
    AmountRounding,
    BASELINE_POLICY,
    CadenceAnchor,
    HistoryWindow,
    POLICY_GRID,
    SameDayOrdering,
    estimate_amount,
    select_history,
)


REQUEST = date(2026, 4, 1)


def event(
    event_id,
    when,
    amount="100",
    *,
    description="Regular bill",
    category="utilities",
    direction=Direction.DEBIT,
    kind=EventType.EXPENSE,
    status=EventStatus.SETTLED,
    flexibility=Flexibility.FIXED,
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
        flexibility,
        None,
    )


def resolved(rows):
    return resolve_event_lifecycles(
        rows, snapshot_date=REQUEST, home_currency=Currency.USD
    )


def fact(
    message_id,
    fact_type,
    *,
    event_id=None,
    amount=None,
    effective=None,
    settlement=None,
    scope=None,
):
    cash = {
        "first_salary_confirmed": "confirmed_credit",
        "unconfirmed_income": "pending_credit",
    }.get(fact_type)
    return MessageEvidenceItem(
        message_id=message_id,
        related_event_id=event_id,
        fact_type=fact_type,
        amount=amount,
        currency="USD" if amount else None,
        effective_date=effective,
        settlement_date=settlement,
        recurrence_scope=scope,
        cash_state=cash,
        confidence="high",
        evidence_quote=None,
        notes="test",
    )


@pytest.mark.parametrize(
    ("dates", "expected"),
    [
        ([date(2026, 1, 1), date(2026, 1, 8), date(2026, 1, 15)], Cadence.WEEKLY),
        ([date(2026, 1, 1), date(2026, 1, 15), date(2026, 1, 29)], Cadence.BIWEEKLY),
        ([date(2026, 1, 1), date(2026, 1, 22), date(2026, 2, 12)], Cadence.THREE_WEEK),
        ([date(2026, 1, 31), date(2026, 2, 28), date(2026, 3, 31)], Cadence.MONTHLY),
    ],
)
def test_all_supported_cadences(dates, expected):
    assert detect_cadence(dates)[0] is expected


def test_robust_clustering_and_insufficient_history():
    assert (
        detect_cadence(
            [date(2026, 1, 1), date(2026, 1, 8), date(2026, 1, 15), date(2026, 2, 20)]
        )[0]
        is Cadence.WEEKLY
    )
    assert detect_cadence([date(2026, 1, 1), date(2026, 1, 8)]) is None


def test_month_end_projection_is_calendar_aware():
    rows = [
        event(f"e{i}", day, category="rent")
        for i, day in enumerate(
            [date(2026, 1, 31), date(2026, 2, 28), date(2026, 3, 31)]
        )
    ]
    series = infer_recurrence_series(
        resolved(rows), request_date=REQUEST, policy=BASELINE_POLICY
    )[0]
    projected = project_recurrence_series(
        series, start_date=REQUEST, end_date=date(2026, 6, 30), policy=BASELINE_POLICY
    )
    assert [item.flow_date for item in projected] == [
        date(2026, 4, 30),
        date(2026, 5, 31),
        date(2026, 6, 30),
    ]


def test_variable_descriptions_form_one_category_series_but_one_off_does_not():
    names = ["Neighbourhood grocer", "Fresh food shop", "Grocery delivery"]
    rows = [
        event(
            f"g{i}",
            date(2026, 3, 1 + 7 * i),
            str(10 + i),
            description=name,
            category="groceries",
        )
        for i, name in enumerate(names)
    ]
    rows.append(
        event(
            "bulk",
            date(2026, 3, 22),
            "500",
            description="Large bulk pantry purchase",
            category="groceries",
        )
    )
    series = infer_recurrence_series(
        resolved(rows), request_date=REQUEST, policy=BASELINE_POLICY
    )
    assert len(series) == 1
    assert series[0].source_event_ids == ("g0", "g1", "g2")


def test_two_independent_salaries_in_same_category_remain_separate():
    rows = []
    for month in (1, 2, 3):
        rows.extend(
            (
                event(
                    f"primary-{month}",
                    date(2026, month, 15),
                    "1000",
                    description="Primary household salary",
                    category="salary",
                    direction=Direction.CREDIT,
                    kind=EventType.INCOME,
                ),
                event(
                    f"second-{month}",
                    date(2026, month, 20),
                    "400",
                    description="Second household income",
                    category="salary",
                    direction=Direction.CREDIT,
                    kind=EventType.INCOME,
                ),
            )
        )
    series = infer_recurrence_series(
        resolved(rows), request_date=REQUEST, policy=BASELINE_POLICY
    )
    assert len(series) == 2
    assert {item.key.description_family for item in series} == {
        "salary:primary household",
        "salary:second household",
    }


def test_final_payroll_terminates_supported_recurrence():
    rows = [
        event(
            f"salary-{month}",
            date(2026, month, 15),
            description="Final employer payroll" if month == 3 else "Payroll credit",
            category="salary",
            direction=Direction.CREDIT,
            kind=EventType.INCOME,
        )
        for month in (1, 2, 3)
    ]
    assert not infer_recurrence_series(
        resolved(rows), request_date=REQUEST, policy=BASELINE_POLICY
    )


def test_one_time_irregular_gig_income_does_not_join_payroll_series():
    rows = [
        event(
            f"salary-{month}",
            date(2026, month, 15),
            description="Payroll credit",
            category="salary",
            direction=Direction.CREDIT,
            kind=EventType.INCOME,
        )
        for month in (1, 2, 3)
    ]
    rows.append(
        event(
            "gig",
            date(2026, 3, 22),
            "250",
            description="Task marketplace payout",
            category="salary",
            direction=Direction.CREDIT,
            kind=EventType.INCOME,
        )
    )
    series = infer_recurrence_series(
        resolved(rows), request_date=REQUEST, policy=BASELINE_POLICY
    )
    assert len(series) == 1
    assert series[0].source_event_ids == ("salary-1", "salary-2", "salary-3")


def test_isolated_salary_amount_is_one_occurrence_not_new_series_base():
    amounts = ("1000", "1000", "600")
    rows = [
        event(
            f"salary-{month}",
            date(2026, month, 15),
            amounts[month - 1],
            description="Payroll credit",
            category="salary",
            direction=Direction.CREDIT,
            kind=EventType.INCOME,
        )
        for month in (1, 2, 3)
    ]
    series = infer_recurrence_series(
        resolved(rows), request_date=REQUEST, policy=BASELINE_POLICY
    )[0]
    assert series.projected_amount == Decimal("1000.00")


def test_confirmed_first_and_next_salary_establish_supported_cadence():
    rows = [
        event(
            "first",
            date(2026, 3, 15),
            "600",
            description="Prorated first salary",
            category="salary",
            direction=Direction.CREDIT,
            kind=EventType.INCOME,
        ),
        event(
            "next",
            date(2026, 4, 15),
            "1000",
            description="Next confirmed salary",
            category="salary",
            direction=Direction.CREDIT,
            kind=EventType.INCOME,
            status=EventStatus.SCHEDULED,
        ),
    ]
    series = infer_recurrence_series(
        resolved(rows), request_date=REQUEST, policy=BASELINE_POLICY
    )[0]
    projected = project_recurrence_series(
        series, start_date=REQUEST, end_date=date(2026, 5, 31), policy=BASELINE_POLICY
    )
    assert [(item.flow_date, item.amount) for item in projected] == [
        (date(2026, 4, 15), Decimal("1000.00")),
        (date(2026, 5, 15), Decimal("1000.00")),
    ]


def test_resumed_salary_restarts_at_effective_occurrence_without_backfill():
    rows = [
        event(
            f"salary-{month}",
            date(2026, month, 15),
            description="Payroll credit",
            category="salary",
            direction=Direction.CREDIT,
            kind=EventType.INCOME,
        )
        for month in (1, 2, 3)
    ]
    rows.append(
        event(
            "resume",
            date(2026, 6, 15),
            description="Confirmed resumed salary",
            category="salary",
            direction=Direction.CREDIT,
            kind=EventType.INCOME,
            status=EventStatus.SCHEDULED,
        )
    )
    series = infer_recurrence_series(
        resolved(rows), request_date=REQUEST, policy=BASELINE_POLICY
    )[0]
    projected = project_recurrence_series(
        series, start_date=REQUEST, end_date=date(2026, 7, 31), policy=BASELINE_POLICY
    )
    assert [item.flow_date for item in projected] == [date(2026, 7, 15)]


def test_fixed_recurring_bill_uses_latest_authoritative_amount() -> None:
    rows = [
        event(f"r{i}", date(2026, i, 15), amount, description="Rent", category="rent")
        for i, amount in ((1, "90"), (2, "100"), (3, "110"))
    ]
    series = infer_recurrence_series(
        resolved(rows), request_date=REQUEST, policy=BASELINE_POLICY
    )[0]
    assert series.variable_amount is False
    assert series.projected_amount == Decimal("110.00")


def test_amendment_recency_is_source_scoped_then_financially_conservative() -> None:
    rows = [
        event(f"r{i}", date(2026, i, 15), "100", description="Rent", category="rent")
        for i in (1, 2, 3)
    ]

    def extracted(message_id: str, amount: str, day: int, source: SourceType):
        item = fact(
            message_id,
            "recurring_expense_amendment",
            event_id="r3",
            amount=amount,
            effective=REQUEST,
            scope="recurring",
        )
        return ExtractedMessageEvidence(
            item,
            datetime(2026, 3, day, tzinfo=timezone.utc),
            source,
            "model",
        )

    evidence = (
        extracted("old_bank", "130", 1, SourceType.BANK),
        extracted("new_bank", "110", 3, SourceType.BANK),
        extracted("merchant", "120", 2, SourceType.MERCHANT),
    )
    series = infer_recurrence_series(
        resolve_event_lifecycles(
            rows,
            snapshot_date=REQUEST,
            home_currency=Currency.USD,
            message_evidence=evidence,
        ),
        request_date=REQUEST,
        policy=BASELINE_POLICY,
        message_evidence=evidence,
    )[0]
    projected = project_recurrence_series(
        series,
        start_date=REQUEST,
        end_date=date(2026, 4, 30),
        policy=BASELINE_POLICY,
    )
    assert projected[0].amount == Decimal("120.00")
    assert projected[0].source_message_ids == ("merchant",)


@pytest.mark.parametrize(
    ("estimator", "expected"),
    [
        (AmountEstimator.RECENT_ARITHMETIC_MEAN, "3"),
        (AmountEstimator.MEDIAN, "3"),
        (AmountEstimator.TRIMMED_MEAN_20, "3"),
        (AmountEstimator.LATEST, "5"),
        (AmountEstimator.PERCENTILE_60, "3.4"),
        (AmountEstimator.PERCENTILE_75, "4"),
    ],
)
def test_every_amount_estimator(estimator, expected):
    policy = replace(
        BASELINE_POLICY,
        amount_estimator=estimator,
        amount_rounding=AmountRounding.EXACT_DECIMAL,
    )
    assert estimate_amount(map(Decimal, ["1", "2", "3", "4", "5"]), policy) == Decimal(
        expected
    )


@pytest.mark.parametrize(
    ("window", "count"),
    [
        (HistoryWindow.LAST_2, 2),
        (HistoryWindow.LAST_3, 3),
        (HistoryWindow.LAST_4, 4),
        (HistoryWindow.RECENT_90_DAYS, 4),
    ],
)
def test_every_history_window(window, count):
    class Observation:
        def __init__(self, when):
            self.when = when

    values = [
        Observation(date(2025, 1, 1)),
        Observation(date(2026, 1, 1)),
        Observation(date(2026, 2, 1)),
        Observation(date(2026, 3, 1)),
        Observation(date(2026, 4, 1)),
    ]
    assert (
        len(select_history(values, replace(BASELINE_POLICY, history_window=window)))
        == count
    )


def test_aggregation_and_rounding_policies():
    rows = []
    for month in (1, 2, 3):
        rows.extend(
            [
                event(
                    f"a{month}",
                    date(2026, month, 5),
                    "10.005",
                    description="Cafe",
                    category="dining",
                ),
                event(
                    f"b{month}",
                    date(2026, month, 5),
                    "20",
                    description="Restaurant",
                    category="dining",
                ),
            ]
        )
    per = infer_recurrence_series(
        resolved(rows),
        request_date=REQUEST,
        policy=replace(BASELINE_POLICY, aggregation=Aggregation.PER_OCCURRENCE),
    )[0]
    total = infer_recurrence_series(
        resolved(rows),
        request_date=REQUEST,
        policy=replace(BASELINE_POLICY, aggregation=Aggregation.CALENDAR_PERIOD_TOTAL),
    )[0]
    assert per.projected_amount == Decimal("16.67")
    assert total.projected_amount == Decimal("30.01")
    exact = replace(BASELINE_POLICY, amount_rounding=AmountRounding.EXACT_DECIMAL)
    assert estimate_amount([Decimal("1.005"), Decimal("2.00")], exact) == Decimal(
        "1.5025"
    )
    assert estimate_amount(
        [Decimal("1.005"), Decimal("2.00")], BASELINE_POLICY
    ) == Decimal("1.50")


def test_cadence_anchor_alternatives():
    assert {policy.cadence_anchor for policy in POLICY_GRID} == set(CadenceAnchor)
    assert {policy.aggregation for policy in POLICY_GRID} == set(Aggregation)
    assert {policy.same_day_ordering for policy in POLICY_GRID} == set(SameDayOrdering)


def test_policy_canonical_hash_and_grid_are_complete():
    payload = BASELINE_POLICY.serialize_canonical()
    assert payload == (
        '{"aggregation":"per_occurrence","amount_estimator":'
        '"recent_arithmetic_mean","amount_rounding":"round_half_up_2",'
        '"cadence_anchor":"calendar_day_of_month","history_window":'
        '"last_3_supported_cycles","same_day_ordering":'
        '"debits_before_credits","schema_version":"forecast-policy-v1"}'
    )
    assert BASELINE_POLICY.sha256 == hashlib.sha256(payload.encode()).hexdigest()
    assert (
        BASELINE_POLICY.sha256
        == "743bad6adb94c55064191e2888e7cc2c1433d11ae008e6ae04a0cee7cdd5199b"
    )
    assert len(BASELINE_POLICY.sha256) == 64
    assert len(POLICY_GRID) == 6 * 4 * 2 * 2 * 2 * 2
    assert len({policy.sha256 for policy in POLICY_GRID}) == len(POLICY_GRID)


def test_amount_and_date_amendments_apply_from_effective_cycle():
    rows = [
        event(
            f"s{i}",
            date(2026, i, 15),
            "1000",
            description="Salary",
            category="salary",
            direction=Direction.CREDIT,
            kind=EventType.INCOME,
        )
        for i in (1, 2, 3)
    ]
    facts = [
        fact(
            "raise",
            "salary_amount_amendment",
            amount="1200",
            effective=date(2026, 5, 1),
            scope="recurring",
        ),
        fact(
            "date",
            "salary_date_amendment",
            settlement=date(2026, 4, 20),
            scope="one_cycle",
        ),
    ]
    series = infer_recurrence_series(
        resolved(rows),
        request_date=REQUEST,
        policy=BASELINE_POLICY,
        message_evidence=facts,
    )[0]
    flows = project_recurrence_series(
        series, start_date=REQUEST, end_date=date(2026, 6, 30), policy=BASELINE_POLICY
    )
    assert [(x.flow_date, x.amount) for x in flows] == [
        (date(2026, 4, 20), Decimal("1000.00")),
        (date(2026, 5, 15), Decimal("1200.00")),
        (date(2026, 6, 15), Decimal("1200.00")),
    ]


def test_temporary_salary_one_cycle_then_reverts():
    rows = [
        event(
            f"s{i}",
            date(2026, i, 15),
            "1000",
            description="Salary",
            category="salary",
            direction=Direction.CREDIT,
            kind=EventType.INCOME,
        )
        for i in (1, 2, 3)
    ]
    temporary = fact(
        "temp",
        "temporary_salary",
        amount="700",
        effective=date(2026, 4, 1),
        scope="one_cycle",
    )
    series = infer_recurrence_series(
        resolved(rows),
        request_date=REQUEST,
        policy=BASELINE_POLICY,
        message_evidence=[temporary],
    )[0]
    flows = project_recurrence_series(
        series, start_date=REQUEST, end_date=date(2026, 5, 31), policy=BASELINE_POLICY
    )
    assert [x.amount for x in flows] == [Decimal("700.00"), Decimal("1000.00")]


def test_seasonal_income_end_and_first_salary_scope():
    rows = [
        event(
            f"s{i}",
            date(2026, i, 15),
            "1000",
            description="Salary",
            category="salary",
            direction=Direction.CREDIT,
            kind=EventType.INCOME,
        )
        for i in (1, 2, 3)
    ]
    ended = fact("ended", "salary_amount_amendment", effective=REQUEST, scope="ended")
    assert not infer_recurrence_series(
        resolved(rows),
        request_date=REQUEST,
        policy=BASELINE_POLICY,
        message_evidence=[ended],
    )
    first = fact("first", "first_salary_confirmed", event_id="s1", scope="one_cycle")
    assert not infer_recurrence_series(
        resolved(rows[:3]),
        request_date=REQUEST,
        policy=BASELINE_POLICY,
        message_evidence=[first],
    )


def test_one_offs_and_unconfirmed_income_never_recur():
    oneoffs = [
        event(
            f"b{i}",
            date(2026, i, 1),
            description="One-off bonus",
            category="salary",
            direction=Direction.CREDIT,
            kind=EventType.INCOME,
        )
        for i in (1, 2, 3)
    ]
    assert not infer_recurrence_series(
        resolved(oneoffs), request_date=REQUEST, policy=BASELINE_POLICY
    )
    regular = [
        event(
            f"u{i}",
            date(2026, i, 15),
            description="Contract income",
            category="salary",
            direction=Direction.CREDIT,
            kind=EventType.INCOME,
        )
        for i in (1, 2, 3)
    ]
    unconfirmed = [
        fact(f"m{i}", "unconfirmed_income", event_id=f"u{i}") for i in (1, 2, 3)
    ]
    assert not infer_recurrence_series(
        resolve_event_lifecycles(
            regular,
            snapshot_date=REQUEST,
            home_currency=Currency.USD,
            message_evidence=unconfirmed,
        ),
        request_date=REQUEST,
        policy=BASELINE_POLICY,
        message_evidence=unconfirmed,
    )


@pytest.mark.parametrize(
    "description",
    [
        "Pending fuel authorization",
        "Card reversal",
        "Duplicate representation",
        "Investment sale proceeds",
        "Annual setup deposit",
    ],
)
def test_lifecycle_artifact_and_one_off_descriptions_do_not_recur(description):
    rows = [
        event(
            f"x{i}",
            date(2026, 3, 1 + 7 * i),
            description=description,
            category="transport",
        )
        for i in range(3)
    ]
    assert not infer_recurrence_series(
        resolved(rows), request_date=REQUEST, policy=BASELINE_POLICY
    )


def test_explicit_policy_is_required():
    with pytest.raises(ValueError, match="explicit selected"):
        infer_recurrence_series((), request_date=REQUEST, policy=None)
