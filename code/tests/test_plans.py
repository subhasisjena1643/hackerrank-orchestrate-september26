"""Phase 11 candidate eligibility, replay, ranking, and fallback coverage."""

from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal
import pytest

from buy_or_wait.domain import (
    AffordabilityStatus,
    CandidatePlan,
    Currency,
    Direction,
    EventStatus,
    EventType,
    FinancialEvent,
    Flexibility,
    Payment,
    PaymentMethod,
    PaymentOption,
    Profile,
    Request,
    RequestType,
    SpendingChange,
    SpendingChangeType,
)
from buy_or_wait.finance.cashflows import ProjectedCashFlow
from buy_or_wait.finance.decision import (
    candidate_ranking_key,
    rank_candidate_plans,
    select_final_decision,
)
from buy_or_wait.finance.forecast import ReplayResult, calculate_baseline_forecast
from buy_or_wait.finance.plans import (
    PlanEligibilityError,
    VerifiedPlanCandidate,
    build_installment_schedule,
    generate_candidate_plans,
    verify_candidate_plan,
)
from buy_or_wait.finance.lifecycle import resolve_event_lifecycles
from buy_or_wait.finance.recurrence import SeriesKey, infer_recurrence_series
from buy_or_wait.finance.recurrence_policies import BASELINE_POLICY
from buy_or_wait.finance.spending_changes import (
    generate_spending_change_candidates,
)


TODAY = date(2026, 9, 1)


def request(*, partial=True, deadline=TODAY + timedelta(days=90)):
    return Request(
        "request_1",
        "u",
        TODAY,
        RequestType.OTHER,
        Decimal("100"),
        deadline,
        partial,
        "x",
    )


def profile(*, balance="250", methods=None, months=4):
    return Profile(
        "u", Currency.USD, Decimal(balance), Decimal("20"), (), (), ("fun",),
        ("fun",),
        tuple(methods or (PaymentMethod.FULL_PAYMENT, PaymentMethod.PARTIAL_PAYMENT,
                          PaymentMethod.INSTALLMENTS)), months,
    )


def flow(flow_id, when, amount, direction=Direction.CREDIT, sources=()):
    key = SeriesKey(
        "u",
        direction,
        "salary" if direction is Direction.CREDIT else "fun",
        EventType.INCOME
        if direction is Direction.CREDIT
        else EventType.EXPENSE,
        Currency.USD,
        Flexibility.FIXED,
        flow_id,
    )
    return ProjectedCashFlow(
        flow_id,
        when,
        Decimal(amount),
        direction,
        key.category,
        flow_id,
        tuple(sources),
        (),
        (),
        ("test",),
        False,
        key,
    )


def baseline(req, prof, flows=()):
    return calculate_baseline_forecast(
        current_available_balance=prof.current_available_balance,
        minimum_balance_to_keep=prof.minimum_balance_to_keep,
        request_date=req.request_date,
        requested_amount=req.requested_amount,
        cash_flows=flows,
        policy=BASELINE_POLICY,
    )


def option(
    option_id="payment_option_2",
    *,
    frequency=30,
    count=2,
    amount="55",
    fee="10",
    total="110",
    first=TODAY,
    request_id="request_1",
):
    return PaymentOption(
        option_id,
        request_id,
        PaymentMethod.INSTALLMENTS,
        Decimal(amount),
        count,
        first,
        frequency,
        Decimal(fee),
        Decimal(total),
    )


def generate(
    req=None,
    prof=None,
    flows=(),
    options=(),
    changes=(),
    series=(),
    resolved=(),
):
    req, prof = req or request(), prof or profile()
    return generate_candidate_plans(
        request=req, profile=prof, baseline=baseline(req, prof, flows),
        payment_options=options, cash_flows=flows, policy=BASELINE_POLICY,
        spending_change_candidates=changes,
        recurring_series=series,
        resolved_events=resolved,
    )


def change_context(prof, amount="70"):
    events = tuple(
        FinancialEvent(
            f"event_{index}",
            "u",
            EventType.SUBSCRIPTION,
            "Fun subscription",
            "fun",
            Direction.DEBIT,
            Decimal(amount),
            Currency.USD,
            date(2026, month, 1),
            date(2026, month, 1),
            EventStatus.SETTLED,
            None,
            Flexibility.STOPPABLE,
            None,
        )
        for index, month in enumerate((6, 7, 8), start=1)
    )
    resolved = resolve_event_lifecycles(
        events,
        snapshot_date=TODAY,
        home_currency=Currency.USD,
    )
    series = infer_recurrence_series(
        resolved,
        request_date=TODAY,
        policy=BASELINE_POLICY,
    )
    candidates = generate_spending_change_candidates(
        prof,
        series,
        resolved,
        request_date=TODAY,
    )
    return resolved, series, candidates


def methods(rows):
    return [row.payment_method for row in rows]


def test_full_now_and_method_preference_boundary():
    rows = generate()
    full = next(row for row in rows if row.payment_method is PaymentMethod.FULL_PAYMENT)
    assert full.affordability_status is AffordabilityStatus.AFFORDABLE_NOW
    assert full.payments == (Payment(TODAY, Decimal("100")),)
    assert generate(prof=profile(methods=(PaymentMethod.INSTALLMENTS,))) == ()


def test_full_now_rejects_inconsistent_baseline_earliest_date():
    req, prof = request(), profile()
    base = replace(
        baseline(req, prof),
        earliest_date_for_full_payment=TODAY + timedelta(days=1),
    )
    plan = CandidatePlan(
        PaymentMethod.FULL_PAYMENT,
        (Payment(TODAY, Decimal("100")),),
        Decimal("100"),
    )
    with pytest.raises(PlanEligibilityError, match="earliest date"):
        verify_candidate_plan(
            plan,
            AffordabilityStatus.AFFORDABLE_NOW,
            request=req,
            profile=prof,
            baseline=base,
            payment_options=(),
            cash_flows=(),
            policy=BASELINE_POLICY,
        )


def test_full_now_with_changes_replays_and_keeps_baseline_safe_amount():
    prof, req = profile(balance="150"), request()
    resolved, series, variants = change_context(prof)
    change = variants[0].changes[0]
    debit = flow(
        "expense",
        TODAY + timedelta(days=1),
        "70",
        Direction.DEBIT,
        (change.event_id,),
    )
    base = baseline(req, prof, (debit,))
    rows = generate(
        req,
        prof,
        (debit,),
        changes=variants,
        series=series,
        resolved=resolved,
    )
    selected = select_final_decision(request=req, baseline=base, candidates=rows)
    assert selected.affordability_status is AffordabilityStatus.AFFORDABLE_WITH_PLAN
    assert selected.amount_safe_to_pay == Decimal("60")
    assert selected.spending_changes_needed == (change,)


@pytest.mark.parametrize(
    ("allows", "accepted", "safe", "expected"),
    [(False, True, "50", False), (True, False, "50", False),
     (True, True, "0", False), (True, True, "100", False)],
)
def test_partial_eligibility_boundaries(allows, accepted, safe, expected):
    req = request(partial=allows)
    accepted_methods = (
        (PaymentMethod.PARTIAL_PAYMENT,)
        if accepted
        else (PaymentMethod.FULL_PAYMENT,)
    )
    prof = profile(balance=str(Decimal(safe) + 20), methods=accepted_methods)
    assert (PaymentMethod.PARTIAL_PAYMENT in methods(generate(req, prof))) is expected


def test_partial_deadline_equality_exact_two_payment_sum_and_replay():
    payday = TODAY + timedelta(days=10)
    credit = flow("salary", payday, "50")
    req = request(deadline=payday)
    prof = profile(balance="70", methods=(PaymentMethod.PARTIAL_PAYMENT,))
    rows = generate(req, prof, (credit,))
    partial = next(
        row for row in rows if row.payment_method is PaymentMethod.PARTIAL_PAYMENT
    )
    assert partial.payments == (
        Payment(TODAY, Decimal("50")),
        Payment(payday, Decimal("50")),
    )
    assert sum((p.amount for p in partial.payments), Decimal(0)) == req.requested_amount
    assert partial.replay.is_safe
    missed = replace(
        req, desired_completion_date=payday - timedelta(days=1)
    )
    assert PaymentMethod.PARTIAL_PAYMENT not in methods(
        generate(missed, prof, (credit,))
    )


def test_external_partial_with_wrong_split_is_rejected():
    payday = TODAY + timedelta(days=10)
    credit = flow("salary", payday, "50")
    req = request(deadline=payday)
    prof = profile(balance="70", methods=(PaymentMethod.PARTIAL_PAYMENT,))
    bad = CandidatePlan(
        PaymentMethod.PARTIAL_PAYMENT,
        (Payment(TODAY, Decimal("49")), Payment(payday, Decimal("51"))),
        Decimal("100"),
    )
    with pytest.raises(PlanEligibilityError, match="exact two-payment"):
        verify_candidate_plan(
            bad,
            AffordabilityStatus.AFFORDABLE_WITH_PLAN,
            request=req,
            profile=prof,
            baseline=baseline(req, prof, (credit,)),
            payment_options=(),
            cash_flows=(credit,),
            policy=BASELINE_POLICY,
        )


@pytest.mark.parametrize("frequency", [28, 30, 31])
def test_installment_schedule_uses_exact_day_frequency_and_amount(frequency):
    opt = option(
        frequency=frequency,
        count=3,
        amount="36.67",
        fee="10.01",
        total="110.01",
    )
    schedule = build_installment_schedule(opt)
    assert [p.payment_date for p in schedule] == [
        TODAY,
        TODAY + timedelta(days=frequency),
        TODAY + timedelta(days=2 * frequency),
    ]
    assert [p.amount for p in schedule] == [Decimal("36.67")] * 3


def test_installment_preference_request_max_term_total_and_deadline_boundaries():
    opt = option()
    assert PaymentMethod.INSTALLMENTS in methods(generate(options=(opt,)))
    excluded = profile(methods=(PaymentMethod.FULL_PAYMENT,))
    assert PaymentMethod.INSTALLMENTS not in methods(
        generate(prof=excluded, options=(opt,))
    )
    assert PaymentMethod.INSTALLMENTS not in methods(
        generate(prof=profile(months=1), options=(opt,))
    )
    assert PaymentMethod.INSTALLMENTS in methods(
        generate(prof=profile(months=2), options=(opt,))
    )
    assert PaymentMethod.INSTALLMENTS not in methods(
        generate(prof=profile(months=None), options=(opt,))
    )
    wrong_request = replace(opt, request_id="request_2")
    assert PaymentMethod.INSTALLMENTS not in methods(
        generate(options=(wrong_request,))
    )
    wrong_total = replace(opt, total_payable_amount=Decimal("109"))
    assert PaymentMethod.INSTALLMENTS not in methods(
        generate(options=(wrong_total,))
    )
    wrong_fee = replace(opt, financing_fee=Decimal("9"))
    assert PaymentMethod.INSTALLMENTS not in methods(
        generate(options=(wrong_fee,))
    )
    late = replace(opt, first_payment_date=TODAY + timedelta(days=61))
    assert PaymentMethod.INSTALLMENTS not in methods(generate(options=(late,)))
    equality = replace(opt, first_payment_date=TODAY + timedelta(days=60))
    assert PaymentMethod.INSTALLMENTS in methods(generate(options=(equality,)))


def test_installment_can_be_made_safe_by_valid_change():
    prof = profile(balance="150", methods=(PaymentMethod.INSTALLMENTS,))
    resolved, series, variants = change_context(prof, "80")
    change = variants[0].changes[0]
    debit = flow(
        "expense",
        TODAY + timedelta(days=2),
        "80",
        Direction.DEBIT,
        (change.event_id,),
    )
    rows = generate(
        prof=prof,
        flows=(debit,),
        options=(option(),),
        changes=variants,
        series=series,
        resolved=resolved,
    )
    installment = next(
        row for row in rows if row.payment_method is PaymentMethod.INSTALLMENTS
    )
    assert installment.spending_changes == (change,) and installment.replay.is_safe
    assert all(row.spending_changes for row in rows)


def test_forged_spending_change_is_independently_rejected():
    prof = profile(balance="150")
    forged = SpendingChange(SpendingChangeType.STOP, "event_999")
    debit = flow(
        "expense",
        TODAY + timedelta(days=1),
        "70",
        Direction.DEBIT,
        (forged.event_id,),
    )
    plan = CandidatePlan(
        PaymentMethod.FULL_PAYMENT,
        (Payment(TODAY, Decimal("100")),),
        Decimal("100"),
        (forged,),
    )
    with pytest.raises(PlanEligibilityError, match="invalid spending changes"):
        verify_candidate_plan(
            plan,
            AffordabilityStatus.AFFORDABLE_WITH_PLAN,
            request=request(),
            profile=prof,
            baseline=baseline(request(), prof, (debit,)),
            payment_options=(),
            cash_flows=(debit,),
            policy=BASELINE_POLICY,
        )


def test_generation_does_not_silence_invalid_forecast_configuration():
    req, prof = request(), profile()
    with pytest.raises(ValueError, match="explicit selected"):
        generate_candidate_plans(
            request=req,
            profile=prof,
            baseline=baseline(req, prof),
            payment_options=(),
            cash_flows=(),
            policy=None,
        )


def test_wait_eligibility_and_no_optional_changes():
    payday = TODAY + timedelta(days=10)
    credit = flow("salary", payday, "50")
    req = request(deadline=payday)
    prof = profile(balance="70", methods=(PaymentMethod.FULL_PAYMENT,))
    wait = next(
        row
        for row in generate(req, prof, (credit,))
        if row.payment_method is PaymentMethod.WAIT
    )
    assert wait.affordability_status is AffordabilityStatus.AFFORDABLE_LATER
    assert wait.payments == (Payment(payday, Decimal("100")),)
    assert not wait.spending_changes
    missed = replace(req, desired_completion_date=payday - timedelta(days=1))
    assert PaymentMethod.WAIT not in methods(generate(missed, prof, (credit,)))
    partial_only = profile(
        balance="70", methods=(PaymentMethod.PARTIAL_PAYMENT,)
    )
    assert PaymentMethod.WAIT not in methods(
        generate(req, partial_only, (credit,))
    )


def ranked_candidate(
    *,
    complete=True,
    changes=(),
    total="100",
    first=TODAY,
    payments=1,
    option_id=None,
    savings="0",
):
    plan = CandidatePlan(
        PaymentMethod.INSTALLMENTS,
        tuple(
            Payment(first + timedelta(days=index), Decimal("1"))
            for index in range(payments)
        ),
        Decimal(total),
        tuple(changes),
        option_id,
    )
    return VerifiedPlanCandidate(
        plan,
        AffordabilityStatus.AFFORDABLE_WITH_PLAN,
        ReplayResult((), Decimal("20"), True),
        complete,
        Decimal(savings),
    )


def test_ranking_precedence_and_change_before_cost():
    change = SpendingChange(SpendingChangeType.STOP, "event_2")
    incomplete = ranked_candidate(complete=False, total="1")
    no_change = ranked_candidate(total="110", option_id="option_9")
    changed = ranked_candidate(changes=(change,), total="1", option_id="option_1")
    assert rank_candidate_plans((incomplete, changed, no_change))[0] is no_change


def test_ranking_cost_first_date_payment_count_and_lowest_option_id():
    expensive = ranked_candidate(
        total="101", first=TODAY - timedelta(days=2), option_id="option_1"
    )
    late = ranked_candidate(
        total="100", first=TODAY + timedelta(days=1), option_id="option_1"
    )
    many = ranked_candidate(total="100", payments=2, option_id="option_1")
    high_id = ranked_candidate(total="100", option_id="option_10")
    low_id = ranked_candidate(total="100", option_id="option_2")
    assert rank_candidate_plans((expensive, late, many, high_id, low_id))[0] is low_id
    assert candidate_ranking_key(low_id)[:6] < candidate_ranking_key(high_id)[:6]


def test_non_option_sentinel_is_used_only_after_first_five_ranking_keys():
    option_candidate = ranked_candidate(option_id="option_2")
    non_option_candidate = ranked_candidate(option_id=None)
    assert rank_candidate_plans((non_option_candidate, option_candidate))[0] is (
        option_candidate
    )


def test_selection_is_deterministic_when_option_input_order_changes():
    req = request()
    prof = profile(methods=(PaymentMethod.INSTALLMENTS,))
    low = option("payment_option_2")
    high = option("payment_option_10")
    selected_ids = []
    for options in ((high, low), (low, high)):
        rows = generate(req, prof, options=options)
        selected_ids.append(rank_candidate_plans(rows)[0].payment_option_id)
    assert selected_ids == ["payment_option_2", "payment_option_2"]


def test_final_ties_use_change_count_savings_then_numeric_event_id():
    c2 = SpendingChange(SpendingChangeType.STOP, "event_2")
    c10 = SpendingChange(SpendingChangeType.STOP, "event_10")
    two = ranked_candidate(changes=(c2, c10), savings="1")
    costly = ranked_candidate(changes=(c2,), savings="2")
    low = ranked_candidate(changes=(c2,), savings="1")
    high = ranked_candidate(changes=(c10,), savings="1")
    assert rank_candidate_plans((two, costly, high, low))[0] is low


def test_fallback_only_when_no_safe_eligible_candidate_survives():
    req, prof = request(), profile(balance="20", methods=(PaymentMethod.INSTALLMENTS,))
    base = baseline(req, prof)
    decision = select_final_decision(
        request=req, baseline=base, candidates=generate(req, prof)
    )
    assert decision.affordability_status is AffordabilityStatus.NOT_AFFORDABLE
    assert decision.recommended_payment_method is PaymentMethod.NOT_RECOMMENDED
    assert decision.payment_plan == () and decision.amount_safe_to_pay == Decimal(0)
