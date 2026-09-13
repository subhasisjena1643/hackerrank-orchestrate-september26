"""Strict output-plan validation backed by an independent forecast replay."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import re

from buy_or_wait.domain import (
    AffordabilityStatus,
    CandidatePlan,
    Decision,
    PaymentMethod,
    PaymentOption,
    Profile,
    Request,
)
from buy_or_wait.finance.cashflows import ProjectedCashFlow
from buy_or_wait.finance.forecast import (
    BaselineForecast,
    HORIZON_DAYS,
    ReplayResult,
    calculate_baseline_forecast,
)
from buy_or_wait.finance.lifecycle import ResolvedEvent
from buy_or_wait.finance.plans import (
    PlanEligibilityError,
    build_installment_schedule,
    verify_candidate_plan,
)
from buy_or_wait.finance.recurrence import RecurringSeries
from buy_or_wait.finance.recurrence_policies import ForecastPolicy
from buy_or_wait.finance.spending_changes import (
    SpendingChangeValidationError,
    VerifiedSpendingChange,
    verify_spending_changes,
)


class DecisionValidationError(ValueError):
    """A final decision violates an output or replay invariant."""


@dataclass(frozen=True, slots=True)
class DecisionValidationContext:
    request: Request
    profile: Profile
    baseline: BaselineForecast
    payment_options: tuple[PaymentOption, ...]
    cash_flows: tuple[ProjectedCashFlow, ...]
    policy: ForecastPolicy
    recurring_series: tuple[RecurringSeries, ...] = ()
    resolved_events: tuple[ResolvedEvent, ...] = ()
    horizon_days: int = HORIZON_DAYS


@dataclass(frozen=True, slots=True)
class ValidatedDecisionPlan:
    replay: ReplayResult | None
    selected_option: PaymentOption | None
    verified_changes: tuple[VerifiedSpendingChange, ...]


_STATUS_METHODS = {
    AffordabilityStatus.AFFORDABLE_NOW: frozenset({PaymentMethod.FULL_PAYMENT}),
    AffordabilityStatus.AFFORDABLE_WITH_PLAN: frozenset(
        {
            PaymentMethod.FULL_PAYMENT,
            PaymentMethod.PARTIAL_PAYMENT,
            PaymentMethod.INSTALLMENTS,
        }
    ),
    AffordabilityStatus.AFFORDABLE_LATER: frozenset({PaymentMethod.WAIT}),
    AffordabilityStatus.NOT_AFFORDABLE: frozenset({PaymentMethod.NOT_RECOMMENDED}),
}


def validate_decision_plan(
    decision: Decision,
    context: DecisionValidationContext,
) -> ValidatedDecisionPlan:
    """Validate one final decision and independently replay every actual payment."""
    request, profile, baseline = context.request, context.profile, context.baseline
    if decision.request_id != request.request_id:
        raise DecisionValidationError("decision request_id does not match its request")
    if request.user_id != profile.user_id:
        raise DecisionValidationError("request and profile belong to different users")
    try:
        recomputed_baseline = calculate_baseline_forecast(
            current_available_balance=profile.current_available_balance,
            minimum_balance_to_keep=profile.minimum_balance_to_keep,
            request_date=request.request_date,
            requested_amount=request.requested_amount,
            cash_flows=context.cash_flows,
            policy=context.policy,
            horizon_days=context.horizon_days,
        )
    except (TypeError, ValueError) as error:
        raise DecisionValidationError(
            f"cannot independently recompute baseline: {error}"
        ) from error
    if baseline != recomputed_baseline:
        raise DecisionValidationError(
            "validation context baseline does not match independent recomputation"
        )
    safe = decision.amount_safe_to_pay
    if not isinstance(safe, Decimal) or not safe.is_finite():
        raise DecisionValidationError("amount_safe_to_pay must be a finite Decimal")
    if not Decimal(0) <= safe <= request.requested_amount:
        raise DecisionValidationError("amount_safe_to_pay is outside request bounds")
    if safe != baseline.amount_safe_to_pay:
        raise DecisionValidationError("amount_safe_to_pay is not the baseline value")
    if (
        decision.earliest_date_for_full_payment
        != baseline.earliest_date_for_full_payment
    ):
        raise DecisionValidationError("earliest full-payment date is not baseline")
    allowed = _STATUS_METHODS.get(decision.affordability_status)
    if allowed is None or decision.recommended_payment_method not in allowed:
        raise DecisionValidationError(
            "invalid affordability-status/payment-method pair"
        )
    if (
        decision.affordability_status is AffordabilityStatus.AFFORDABLE_NOW
        and decision.earliest_date_for_full_payment != request.request_date
    ):
        raise DecisionValidationError("affordable_now requires request-date earliest")

    method = decision.recommended_payment_method
    if method is PaymentMethod.NOT_RECOMMENDED:
        if decision.payment_plan:
            raise DecisionValidationError("not_recommended requires payment_plan none")
        if decision.spending_changes_needed:
            raise DecisionValidationError(
                "not_recommended cannot require spending changes"
            )
        return ValidatedDecisionPlan(None, None, ())
    if not decision.payment_plan:
        raise DecisionValidationError("a recommended method requires a payment plan")
    if any(
        not isinstance(payment.amount, Decimal)
        or not payment.amount.is_finite()
        or payment.amount <= 0
        for payment in decision.payment_plan
    ):
        raise DecisionValidationError("payment amounts must be finite and positive")
    if (
        tuple(sorted(decision.payment_plan, key=lambda row: row.payment_date))
        != decision.payment_plan
    ):
        raise DecisionValidationError("payment plan must be chronological")

    option = _matching_installment_option(decision, context.payment_options, request)
    candidate = CandidatePlan(
        method,
        decision.payment_plan,
        option.total_payable_amount if option else request.requested_amount,
        decision.spending_changes_needed,
        option.payment_option_id if option else None,
    )
    try:
        checked = verify_candidate_plan(
            candidate,
            decision.affordability_status,
            request=request,
            profile=profile,
            baseline=baseline,
            payment_options=context.payment_options,
            cash_flows=context.cash_flows,
            policy=context.policy,
            recurring_series=context.recurring_series,
            resolved_events=context.resolved_events,
            horizon_days=context.horizon_days,
        )
        verified_changes = (
            verify_spending_changes(
                decision.spending_changes_needed,
                profile,
                recurring_series=context.recurring_series,
                resolved_events=context.resolved_events,
                request_date=request.request_date,
            )
            if decision.spending_changes_needed
            else ()
        )
    except (PlanEligibilityError, SpendingChangeValidationError) as error:
        raise DecisionValidationError(str(error)) from error
    return ValidatedDecisionPlan(checked.replay, option, verified_changes)


def _matching_installment_option(
    decision: Decision,
    options: tuple[PaymentOption, ...],
    request: Request,
) -> PaymentOption | None:
    if decision.recommended_payment_method is not PaymentMethod.INSTALLMENTS:
        return None
    matches: list[PaymentOption] = []
    for option in options:
        if (
            option.request_id != request.request_id
            or option.payment_method is not PaymentMethod.INSTALLMENTS
        ):
            continue
        try:
            schedule = build_installment_schedule(option)
        except PlanEligibilityError:
            continue
        if schedule == decision.payment_plan:
            matches.append(option)
    if not matches:
        raise DecisionValidationError(
            "installment plan does not exactly match a supplied option"
        )
    return min(matches, key=lambda row: _numeric_suffix(row.payment_option_id))


def _numeric_suffix(value: str) -> int:
    match = re.search(r"(\d+)$", value)
    if match is None:
        raise DecisionValidationError(
            "payment_option_id must end in a numeric identifier"
        )
    return int(match.group(1))


verify_decision_plan = validate_decision_plan
validate_plan = validate_decision_plan

__all__ = [
    "DecisionValidationContext",
    "DecisionValidationError",
    "ValidatedDecisionPlan",
    "validate_decision_plan",
    "validate_plan",
    "verify_decision_plan",
]
