"""Phase 11 payment-plan generation and independent replay verification."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
import re
from typing import Iterable

from buy_or_wait.domain import (
    AffordabilityStatus,
    CandidatePlan,
    Payment,
    PaymentMethod,
    PaymentOption,
    Profile,
    Request,
    SpendingChange,
)
from buy_or_wait.finance.cashflows import ProjectedCashFlow
from buy_or_wait.finance.forecast import (
    BaselineForecast,
    HORIZON_DAYS,
    ReplayResult,
    replay_forecast,
)
from buy_or_wait.finance.lifecycle import ResolvedEvent
from buy_or_wait.finance.recurrence import RecurringSeries
from buy_or_wait.finance.recurrence_policies import ForecastPolicy
from buy_or_wait.finance.spending_changes import (
    SpendingChangeCandidate,
    SpendingChangeValidationError,
    verify_spending_changes,
)


class PlanEligibilityError(ValueError):
    """A plan does not exactly satisfy its source facts or eligibility rules."""


@dataclass(frozen=True, slots=True)
class VerifiedPlanCandidate:
    """A structurally eligible candidate that passed an independent replay."""

    plan: CandidatePlan
    affordability_status: AffordabilityStatus
    replay: ReplayResult
    completes_full_request_by_deadline: bool
    lifestyle_savings: Decimal = Decimal(0)

    @property
    def payment_method(self) -> PaymentMethod:
        return self.plan.payment_method

    @property
    def payments(self) -> tuple[Payment, ...]:
        return self.plan.payments

    @property
    def spending_changes(self) -> tuple[SpendingChange, ...]:
        return self.plan.spending_changes

    @property
    def total_payable_amount(self) -> Decimal:
        return self.plan.total_payable_amount

    @property
    def payment_option_id(self) -> str | None:
        return self.plan.payment_option_id

    @property
    def first_payment_date(self) -> date:
        return self.plan.payments[0].payment_date


def build_installment_schedule(option: PaymentOption) -> tuple[Payment, ...]:
    """Reproduce an option exactly; in particular, never adjust its final amount."""
    if option.payment_method is not PaymentMethod.INSTALLMENTS:
        raise PlanEligibilityError("option is not an installment option")
    if option.number_of_payments < 2 or option.payment_frequency_days is None:
        raise PlanEligibilityError("installments require at least two dated payments")
    if option.payment_frequency_days <= 0:
        raise PlanEligibilityError("payment frequency must be positive")
    return tuple(
        Payment(
            option.first_payment_date
            + timedelta(days=index * option.payment_frequency_days),
            option.payment_amount,
        )
        for index in range(option.number_of_payments)
    )


def verify_candidate_plan(
    candidate: CandidatePlan,
    affordability_status: AffordabilityStatus,
    *,
    request: Request,
    profile: Profile,
    baseline: BaselineForecast,
    payment_options: Iterable[PaymentOption],
    cash_flows: Iterable[ProjectedCashFlow],
    policy: ForecastPolicy | None,
    recurring_series: Iterable[RecurringSeries] = (),
    resolved_events: Iterable[ResolvedEvent] = (),
    horizon_days: int = HORIZON_DAYS,
) -> VerifiedPlanCandidate:
    """Fail closed on eligibility, exact schedules, and a fresh forecast replay."""
    payments = candidate.payments
    changes = candidate.spending_changes
    requested = request.requested_amount
    safe_today = baseline.amount_safe_to_pay
    accepted = frozenset(profile.payment_methods_user_will_consider)

    if request.user_id != profile.user_id:
        raise PlanEligibilityError("request and profile belong to different users")
    if not Decimal(0) <= safe_today <= requested:
        raise PlanEligibilityError("baseline safe amount is outside request bounds")
    if not payments or any(payment.amount <= 0 for payment in payments):
        raise PlanEligibilityError("candidate payments must be non-empty and positive")
    if tuple(sorted(payments, key=lambda item: item.payment_date)) != payments:
        raise PlanEligibilityError("candidate payments must be chronological")
    if len(changes) > 3 or len({item.event_id for item in changes}) != len(changes):
        raise PlanEligibilityError("candidate has invalid spending-change targets")
    if (
        not isinstance(candidate.total_payable_amount, Decimal)
        or not candidate.total_payable_amount.is_finite()
        or candidate.total_payable_amount <= 0
    ):
        raise PlanEligibilityError("total payable must be finite")
    lifestyle = Decimal(0)
    if changes:
        try:
            verified_changes = verify_spending_changes(
                changes,
                profile,
                recurring_series=tuple(recurring_series),
                resolved_events=tuple(resolved_events),
                request_date=request.request_date,
            )
        except SpendingChangeValidationError as error:
            raise PlanEligibilityError(f"invalid spending changes: {error}") from error
        lifestyle = sum(
            (item.saving_per_occurrence for item in verified_changes), Decimal(0)
        )

    method = candidate.payment_method
    if method is PaymentMethod.FULL_PAYMENT:
        _verify_full(candidate, affordability_status, request, baseline, accepted)
    elif method is PaymentMethod.PARTIAL_PAYMENT:
        _verify_partial(candidate, affordability_status, request, baseline, accepted)
    elif method is PaymentMethod.INSTALLMENTS:
        _verify_installments(
            candidate,
            affordability_status,
            request,
            profile,
            tuple(payment_options),
            accepted,
        )
    elif method is PaymentMethod.WAIT:
        _verify_wait(candidate, affordability_status, request, baseline, accepted)
    else:
        raise PlanEligibilityError("fallback is not a payable candidate")

    horizon_end = request.request_date + timedelta(days=horizon_days)
    if payments[-1].payment_date > request.desired_completion_date:
        raise PlanEligibilityError("candidate completes after the requested deadline")
    if any(
        not request.request_date <= payment.payment_date <= horizon_end
        for payment in payments
    ):
        raise PlanEligibilityError(
            "candidate payment falls outside the forecast horizon"
        )

    replay = replay_forecast(
        starting_balance=profile.current_available_balance,
        minimum_balance_to_keep=profile.minimum_balance_to_keep,
        request_date=request.request_date,
        cash_flows=tuple(cash_flows),
        payments=payments,
        changes=changes,
        policy=policy,
        horizon_days=horizon_days,
    )
    if not replay.is_safe:
        raise PlanEligibilityError("candidate fails independent forecast replay")
    return VerifiedPlanCandidate(
        candidate, affordability_status, replay, True, lifestyle
    )


def generate_candidate_plans(
    *,
    request: Request,
    profile: Profile,
    baseline: BaselineForecast,
    payment_options: Iterable[PaymentOption],
    cash_flows: Iterable[ProjectedCashFlow],
    policy: ForecastPolicy | None,
    spending_change_candidates: Iterable[SpendingChangeCandidate] = (),
    recurring_series: Iterable[RecurringSeries] = (),
    resolved_events: Iterable[ResolvedEvent] = (),
    horizon_days: int = HORIZON_DAYS,
) -> tuple[VerifiedPlanCandidate, ...]:
    """Generate A-E independently; omit every candidate that fails verification."""
    options = tuple(payment_options)
    flows = tuple(cash_flows)
    change_variants = tuple(spending_change_candidates)
    series = tuple(recurring_series)
    resolved = tuple(resolved_events)
    result: list[VerifiedPlanCandidate] = []

    def add(
        plan: CandidatePlan,
        status: AffordabilityStatus,
    ) -> None:
        try:
            result.append(
                verify_candidate_plan(
                    plan,
                    status,
                    request=request,
                    profile=profile,
                    baseline=baseline,
                    payment_options=options,
                    cash_flows=flows,
                    policy=policy,
                    recurring_series=series,
                    resolved_events=resolved,
                    horizon_days=horizon_days,
                )
            )
        except PlanEligibilityError:
            return

    full_now = CandidatePlan(
        PaymentMethod.FULL_PAYMENT,
        (Payment(request.request_date, request.requested_amount),),
        request.requested_amount,
    )
    add(full_now, AffordabilityStatus.AFFORDABLE_NOW)
    for variant in change_variants:
        add(
            CandidatePlan(
                PaymentMethod.FULL_PAYMENT,
                full_now.payments,
                request.requested_amount,
                tuple(variant.changes),
            ),
            AffordabilityStatus.AFFORDABLE_WITH_PLAN,
        )

    safe_today = baseline.amount_safe_to_pay
    earliest = baseline.earliest_date_for_full_payment
    if earliest is not None:
        remainder = request.requested_amount - safe_today
        add(
            CandidatePlan(
                PaymentMethod.PARTIAL_PAYMENT,
                (
                    Payment(request.request_date, safe_today),
                    Payment(earliest, remainder),
                ),
                request.requested_amount,
            ),
            AffordabilityStatus.AFFORDABLE_WITH_PLAN,
        )
        add(
            CandidatePlan(
                PaymentMethod.WAIT,
                (Payment(earliest, request.requested_amount),),
                request.requested_amount,
            ),
            AffordabilityStatus.AFFORDABLE_LATER,
        )

    installment_variants: tuple[SpendingChangeCandidate | None, ...] = (
        None,
        *change_variants,
    )
    for option in options:
        if option.payment_method is not PaymentMethod.INSTALLMENTS:
            continue
        try:
            schedule = build_installment_schedule(option)
        except PlanEligibilityError:
            continue
        for variant in installment_variants:
            changes = () if variant is None else tuple(variant.changes)
            add(
                CandidatePlan(
                    PaymentMethod.INSTALLMENTS,
                    schedule,
                    option.total_payable_amount,
                    changes,
                    option.payment_option_id,
                ),
                AffordabilityStatus.AFFORDABLE_WITH_PLAN,
            )
    return tuple(result)


def _verify_full(
    candidate: CandidatePlan,
    status: AffordabilityStatus,
    request: Request,
    baseline: BaselineForecast,
    accepted: frozenset[PaymentMethod],
) -> None:
    if PaymentMethod.FULL_PAYMENT not in accepted:
        raise PlanEligibilityError("user does not accept full payment")
    expected = (Payment(request.request_date, request.requested_amount),)
    if (
        candidate.payments != expected
        or candidate.total_payable_amount != request.requested_amount
    ):
        raise PlanEligibilityError(
            "full-now candidate must pay the exact request today"
        )
    if candidate.payment_option_id is not None:
        raise PlanEligibilityError("full-now candidate is not seller-option based")
    if status is AffordabilityStatus.AFFORDABLE_NOW:
        if candidate.spending_changes:
            raise PlanEligibilityError("affordable-now full payment cannot use changes")
        if baseline.amount_safe_to_pay != request.requested_amount:
            raise PlanEligibilityError("full request is not safe today in baseline")
        if baseline.earliest_date_for_full_payment != request.request_date:
            raise PlanEligibilityError(
                "affordable-now earliest date must be request date"
            )
    elif status is AffordabilityStatus.AFFORDABLE_WITH_PLAN:
        if not 1 <= len(candidate.spending_changes) <= 3:
            raise PlanEligibilityError(
                "full payment with changes needs one to three actions"
            )
        if baseline.amount_safe_to_pay >= request.requested_amount:
            raise PlanEligibilityError(
                "spending changes do not make an unsafe plan safe"
            )
    else:
        raise PlanEligibilityError("invalid status for full-now candidate")


def _verify_partial(
    candidate: CandidatePlan,
    status: AffordabilityStatus,
    request: Request,
    baseline: BaselineForecast,
    accepted: frozenset[PaymentMethod],
) -> None:
    safe = baseline.amount_safe_to_pay
    earliest = baseline.earliest_date_for_full_payment
    if not request.allows_partial_payment:
        raise PlanEligibilityError("request does not allow partial payment")
    if PaymentMethod.PARTIAL_PAYMENT not in accepted:
        raise PlanEligibilityError("user does not accept partial payment")
    if status is not AffordabilityStatus.AFFORDABLE_WITH_PLAN:
        raise PlanEligibilityError("partial payment must be affordable_with_plan")
    if not Decimal(0) < safe < request.requested_amount or earliest is None:
        raise PlanEligibilityError("baseline does not permit a partial plan")
    expected = (
        Payment(request.request_date, safe),
        Payment(earliest, request.requested_amount - safe),
    )
    if candidate.payments != expected or sum(
        (item.amount for item in candidate.payments), Decimal(0)
    ) != request.requested_amount:
        raise PlanEligibilityError("partial plan must be the exact two-payment split")
    if earliest > request.desired_completion_date:
        raise PlanEligibilityError("partial plan misses the completion deadline")
    if candidate.spending_changes or candidate.payment_option_id is not None:
        raise PlanEligibilityError("partial payment cannot use changes or an option ID")
    if candidate.total_payable_amount != request.requested_amount:
        raise PlanEligibilityError("partial plan total must equal requested amount")


def _verify_installments(
    candidate: CandidatePlan,
    status: AffordabilityStatus,
    request: Request,
    profile: Profile,
    options: tuple[PaymentOption, ...],
    accepted: frozenset[PaymentMethod],
) -> None:
    if PaymentMethod.INSTALLMENTS not in accepted:
        raise PlanEligibilityError("user does not accept installments")
    if status is not AffordabilityStatus.AFFORDABLE_WITH_PLAN:
        raise PlanEligibilityError("installments must be affordable_with_plan")
    if profile.max_installment_months is None:
        raise PlanEligibilityError("user has no installment term allowance")
    option = next(
        (
            item
            for item in options
            if item.payment_option_id == candidate.payment_option_id
        ),
        None,
    )
    if option is None or option.request_id != request.request_id:
        raise PlanEligibilityError(
            "installment option is not supplied for this request"
        )
    if option.payment_method is not PaymentMethod.INSTALLMENTS:
        raise PlanEligibilityError("matched option is not installments")
    if option.number_of_payments > profile.max_installment_months:
        raise PlanEligibilityError("installment term exceeds max_installment_months")
    expected = build_installment_schedule(option)
    exact_total = option.payment_amount * option.number_of_payments
    if candidate.payments != expected:
        raise PlanEligibilityError("installment schedule does not exactly match option")
    if exact_total != option.total_payable_amount:
        raise PlanEligibilityError("installment amounts do not equal total payable")
    if option.total_payable_amount != request.requested_amount + option.financing_fee:
        raise PlanEligibilityError("installment total does not preserve financing fee")
    if candidate.total_payable_amount != option.total_payable_amount:
        raise PlanEligibilityError("candidate total differs from supplied option")
    if expected[-1].payment_date > request.desired_completion_date:
        raise PlanEligibilityError("installment option completes after deadline")
    _numeric_suffix(option.payment_option_id, "payment_option_id")


def _verify_wait(
    candidate: CandidatePlan,
    status: AffordabilityStatus,
    request: Request,
    baseline: BaselineForecast,
    accepted: frozenset[PaymentMethod],
) -> None:
    earliest = baseline.earliest_date_for_full_payment
    if PaymentMethod.FULL_PAYMENT not in accepted:
        raise PlanEligibilityError("user does not accept the eventual full payment")
    if status is not AffordabilityStatus.AFFORDABLE_LATER:
        raise PlanEligibilityError("wait candidate must be affordable_later")
    if earliest is None or not request.request_date < earliest:
        raise PlanEligibilityError("wait requires a later baseline-safe full date")
    if earliest > request.desired_completion_date:
        raise PlanEligibilityError("wait date is after the requested deadline")
    if candidate.payments != (Payment(earliest, request.requested_amount),):
        raise PlanEligibilityError("wait must make one exact full payment")
    if candidate.total_payable_amount != request.requested_amount:
        raise PlanEligibilityError("wait total must equal requested amount")
    if candidate.spending_changes or candidate.payment_option_id is not None:
        raise PlanEligibilityError("wait cannot use changes or an option ID")


def _numeric_suffix(value: str, label: str) -> int:
    match = re.search(r"(\d+)$", value)
    if match is None:
        raise PlanEligibilityError(f"{label} must end in a numeric identifier")
    return int(match.group(1))


generate_candidates = generate_candidate_plans
verify_candidate = verify_candidate_plan

__all__ = [
    "PlanEligibilityError",
    "VerifiedPlanCandidate",
    "build_installment_schedule",
    "generate_candidate_plans",
    "generate_candidates",
    "verify_candidate",
    "verify_candidate_plan",
]
