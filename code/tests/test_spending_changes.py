"""Phase 10 spending-change generation and independent verification."""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from decimal import Decimal
from pathlib import Path
import sys

import pytest

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from buy_or_wait.data_loader import load_dataset
from buy_or_wait.domain import (
    Currency,
    Direction,
    EventStatus,
    EventType,
    FinancialEvent,
    Flexibility,
    PaymentMethod,
    Profile,
    SpendingChange,
    SpendingChangeType,
)
from buy_or_wait.finance.cashflows import construct_future_cash_flows
from buy_or_wait.finance.forecast import replay_forecast
from buy_or_wait.finance.lifecycle import resolve_event_lifecycles
from buy_or_wait.finance.recurrence import infer_recurrence_series
from buy_or_wait.finance.recurrence_policies import BASELINE_POLICY
from buy_or_wait.finance.spending_changes import (
    SpendingChangeValidationError,
    generate_spending_change_candidates,
    verify_spending_changes,
)

REQUEST = date(2026, 4, 1)


def profile(*, protected=(), reduce=("streaming",), stop=("streaming",)):
    return Profile(
        "u",
        Currency.USD,
        Decimal("500"),
        Decimal("0"),
        (),
        tuple(protected),
        tuple(reduce),
        tuple(stop),
        (PaymentMethod.FULL_PAYMENT,),
        None,
    )


def event(
    event_id,
    month,
    *,
    category="streaming",
    flexibility=Flexibility.REDUCIBLE_OR_STOPPABLE,
    floor="40.00",
    description="Streaming plan",
):
    return FinancialEvent(
        event_id,
        "u",
        EventType.SUBSCRIPTION,
        description,
        category,
        Direction.DEBIT,
        Decimal("100"),
        Currency.USD,
        date(2026, month, 15),
        date(2026, month, 15),
        EventStatus.SETTLED,
        None,
        flexibility,
        Decimal(floor) if floor is not None else None,
    )


def inputs(*, flexibility=Flexibility.REDUCIBLE_OR_STOPPABLE, floor="40.00"):
    events = tuple(
        event(f"e{i}", i, flexibility=flexibility, floor=floor) for i in (1, 2, 3)
    )
    resolved = resolve_event_lifecycles(
        events, snapshot_date=REQUEST, home_currency=Currency.USD
    )
    series = infer_recurrence_series(
        resolved, request_date=REQUEST, policy=BASELINE_POLICY
    )
    return resolved, series


def actions(candidates):
    return [candidate.changes for candidate in candidates]


def test_protected_category_precedence_and_profile_permission():
    resolved, series = inputs()
    assert not generate_spending_change_candidates(
        profile(protected=("streaming",)), series, resolved, request_date=REQUEST
    )
    assert not generate_spending_change_candidates(
        profile(reduce=(), stop=()), series, resolved, request_date=REQUEST
    )
    with pytest.raises(SpendingChangeValidationError, match="protected"):
        verify_spending_changes(
            (SpendingChange(SpendingChangeType.STOP, "e3"),),
            profile(protected=("streaming",)),
            recurring_series=series,
            resolved_events=resolved,
            request_date=REQUEST,
        )


@pytest.mark.parametrize(
    ("flexibility", "expected"),
    [
        (Flexibility.FIXED, set()),
        (Flexibility.STOPPABLE, {SpendingChangeType.STOP}),
        (Flexibility.REDUCIBLE, {SpendingChangeType.REDUCE_TO}),
        (
            Flexibility.REDUCIBLE_OR_STOPPABLE,
            {SpendingChangeType.STOP, SpendingChangeType.REDUCE_TO},
        ),
    ],
)
def test_all_flexibility_values(flexibility, expected):
    resolved, series = inputs(flexibility=flexibility)
    generated = generate_spending_change_candidates(
        profile(), series, resolved, request_date=REQUEST
    )
    assert {value[0].change_type for value in actions(generated)} == expected


def test_reduction_floor_format_missing_floor_and_external_bounds():
    resolved, series = inputs(floor="40.00")
    generated = generate_spending_change_candidates(
        profile(stop=()), series, resolved, request_date=REQUEST
    )
    reduction = generated[0].changes[0]
    assert reduction.new_amount == Decimal("40.00")
    assert reduction.new_amount.as_tuple().exponent == -2
    for amount in (Decimal("39.99"), Decimal("100.01")):
        with pytest.raises(SpendingChangeValidationError, match="floor|baseline"):
            verify_spending_changes(
                (replace(reduction, new_amount=amount),),
                profile(stop=()),
                recurring_series=series,
                resolved_events=resolved,
                request_date=REQUEST,
            )
    no_floor_rows, no_floor_series = inputs(floor=None)
    assert not generate_spending_change_candidates(
        profile(stop=()), no_floor_series, no_floor_rows, request_date=REQUEST
    )
    with pytest.raises(SpendingChangeValidationError, match="floor"):
        verify_spending_changes(
            (SpendingChange(SpendingChangeType.REDUCE_TO, "e3", Decimal("20")),),
            profile(stop=()),
            recurring_series=no_floor_series,
            resolved_events=no_floor_rows,
            request_date=REQUEST,
        )


def test_external_stop_cannot_carry_a_replacement_amount():
    resolved, series = inputs()
    with pytest.raises(SpendingChangeValidationError, match="cannot specify"):
        verify_spending_changes(
            (
                SpendingChange(
                    SpendingChangeType.STOP,
                    "e3",
                    Decimal("0"),
                ),
            ),
            profile(),
            recurring_series=series,
            resolved_events=resolved,
            request_date=REQUEST,
        )


def test_one_to_three_limit_no_duplicate_target_and_same_event_conflict():
    events = []
    categories = ("streaming", "gym", "cloud_storage", "music_subscription")
    for index, category in enumerate(categories):
        events.extend(
            event(
                f"{category}_{month}",
                month,
                category=category,
                flexibility=Flexibility.STOPPABLE,
                floor=None,
                description=category,
            )
            for month in (1, 2, 3)
        )
    resolved = resolve_event_lifecycles(
        events, snapshot_date=REQUEST, home_currency=Currency.USD
    )
    series = infer_recurrence_series(
        resolved, request_date=REQUEST, policy=BASELINE_POLICY
    )
    candidates = generate_spending_change_candidates(
        profile(reduce=(), stop=categories), series, resolved, request_date=REQUEST
    )
    assert {len(candidate.changes) for candidate in candidates} == {1, 2, 3}
    assert all(
        len({item.event_id for item in candidate.changes}) == len(candidate.changes)
        for candidate in candidates
    )
    with pytest.raises(SpendingChangeValidationError, match="one to three"):
        verify_spending_changes(
            tuple(
                SpendingChange(SpendingChangeType.STOP, f"{category}_3")
                for category in categories
            ),
            profile(reduce=(), stop=categories),
            recurring_series=series,
            resolved_events=resolved,
            request_date=REQUEST,
        )
    basic_rows, basic_series = inputs()
    with pytest.raises(SpendingChangeValidationError, match="conflicts"):
        verify_spending_changes(
            (
                SpendingChange(SpendingChangeType.STOP, "e3"),
                SpendingChange(SpendingChangeType.REDUCE_TO, "e3", Decimal("40")),
            ),
            profile(),
            recurring_series=basic_series,
            resolved_events=basic_rows,
            request_date=REQUEST,
        )


def test_non_recurring_rejected_latest_id_and_source_metadata_retained():
    resolved, series = inputs()
    candidates = generate_spending_change_candidates(
        profile(stop=()), series, resolved, request_date=REQUEST
    )
    candidate = candidates[0]
    assert candidate.changes[0].event_id == "e3"
    assert candidate.sources[0].description == "Streaming plan"
    assert candidate.sources[0].category == "streaming"
    assert candidate.sources[0].source_event_ids == ("e1", "e2", "e3")
    with pytest.raises(SpendingChangeValidationError, match="latest source"):
        verify_spending_changes(
            (SpendingChange(SpendingChangeType.STOP, "e2"),),
            profile(),
            recurring_series=series,
            resolved_events=resolved,
            request_date=REQUEST,
        )
    with pytest.raises(SpendingChangeValidationError, match="latest source"):
        verify_spending_changes(
            (SpendingChange(SpendingChangeType.STOP, "e3"),),
            profile(),
            recurring_series=(),
            resolved_events=resolved,
            request_date=REQUEST,
        )


def test_forecast_savings_apply_only_to_future_series_occurrences():
    resolved, series = inputs()
    change = generate_spending_change_candidates(
        profile(stop=()), series, resolved, request_date=REQUEST
    )[0].changes
    flows = construct_future_cash_flows(
        resolved, request_date=REQUEST, policy=BASELINE_POLICY
    ).cash_flows
    baseline = replay_forecast(
        starting_balance=Decimal("500"),
        minimum_balance_to_keep=Decimal("0"),
        request_date=REQUEST,
        cash_flows=flows,
        policy=BASELINE_POLICY,
    )
    changed = replay_forecast(
        starting_balance=Decimal("500"),
        minimum_balance_to_keep=Decimal("0"),
        request_date=REQUEST,
        cash_flows=flows,
        policy=BASELINE_POLICY,
        changes=change,
    )
    assert changed.ledger[-1].closing_balance - baseline.ledger[
        -1
    ].closing_balance == Decimal("180.00")
    generated = generate_spending_change_candidates(
        profile(stop=()), series, resolved, request_date=REQUEST
    )
    assert all(item.effective_from == REQUEST for item in generated[0].verified_changes)


def test_generated_actions_include_all_solved_change_examples_without_ids_or_labels():
    dataset = load_dataset(Path(__file__).resolve().parents[2] / "dataset")
    profiles = {item.user_id: item for item in dataset.profiles}
    requests = {item.request_id: item for item in dataset.sample_requests}
    solved = [
        item
        for item in dataset.sample_outputs
        if item.spending_changes_needed != "none"
    ]
    assert solved
    for expected in solved:
        request = requests[expected.request_id]
        user_profile = profiles[request.user_id]
        events = tuple(
            item for item in dataset.financial_events if item.user_id == request.user_id
        )
        resolved = resolve_event_lifecycles(
            events,
            snapshot_date=request.request_date,
            home_currency=user_profile.home_currency,
            exchange_rates=dataset.exchange_rates,
            allow_unresolved_amounts=True,
        )
        series = infer_recurrence_series(
            resolved, request_date=request.request_date, policy=BASELINE_POLICY
        )
        generated = generate_spending_change_candidates(
            user_profile, series, resolved, request_date=request.request_date
        )
        generated_actions = {
            tuple(
                (item.change_type.value, item.event_id, item.new_amount)
                for item in candidate.changes
            )
            for candidate in generated
        }
        expected_actions = tuple(
            (
                parts[0],
                parts[1],
                Decimal(parts[2]) if len(parts) == 3 else None,
            )
            for action in expected.spending_changes_needed.split("|")
            for parts in (action.split(":"),)
        )
        assert expected_actions in generated_actions
