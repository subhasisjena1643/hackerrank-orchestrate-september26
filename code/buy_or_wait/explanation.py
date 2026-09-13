"""Deterministic, source-grounded decision explanations."""

from __future__ import annotations

from dataclasses import replace
from decimal import Decimal
from typing import Iterable

from buy_or_wait.domain import (
    Decision,
    ForecastResult,
    PaymentMethod,
    PaymentOption,
    Profile,
    Request,
    RequestType,
)
from buy_or_wait.finance.forecast import ReplayResult
from buy_or_wait.finance.plans import PlanEligibilityError, build_installment_schedule
from buy_or_wait.finance.spending_changes import VerifiedSpendingChange
from buy_or_wait.money import format_decimal


class ExplanationError(ValueError):
    """A deterministic explanation cannot be grounded in verified inputs."""


def build_decision_explanation(
    decision: Decision,
    *,
    request: Request,
    profile: Profile,
    replay: ReplayResult | None = None,
    forecast_result: ForecastResult | None = None,
    selected_option: PaymentOption | None = None,
    verified_changes: Iterable[VerifiedSpendingChange] = (),
) -> str:
    """Render one concise explanation without model-authored prose."""
    if decision.request_id != request.request_id or request.user_id != profile.user_id:
        raise ExplanationError(
            "decision, request, and profile must describe one request"
        )
    changes = tuple(verified_changes)
    if tuple(item.change for item in changes) != decision.spending_changes_needed:
        if decision.spending_changes_needed:
            raise ExplanationError(
                "every spending change needs verified source provenance"
            )
        if changes:
            raise ExplanationError("unexpected spending-change provenance")
    if replay is not None and forecast_result is not None:
        raise ExplanationError("supply one verified forecast result, not two")
    verified_forecast = replay if replay is not None else forecast_result
    if verified_forecast is not None and not verified_forecast.is_safe:
        raise ExplanationError("a recommended payment plan must have a safe replay")

    currency = profile.home_currency.value
    minimum = _money(profile.minimum_balance_to_keep)
    requested = _money(request.requested_amount)
    method = decision.recommended_payment_method
    prefix = _change_prefix(changes, currency)

    if method is PaymentMethod.FULL_PAYMENT:
        _require_payments(decision, 1)
        text = (
            f"{prefix}Pay {currency} {requested} today. This keeps the "
            f"{currency} {minimum} minimum protected over the next 90 days."
        )
    elif method is PaymentMethod.INSTALLMENTS:
        if selected_option is None:
            raise ExplanationError(
                "installment explanation requires its selected option"
            )
        if selected_option.request_id != request.request_id:
            raise ExplanationError("selected option belongs to another request")
        try:
            expected_schedule = build_installment_schedule(selected_option)
        except PlanEligibilityError as error:
            raise ExplanationError("selected installment option is invalid") from error
        if expected_schedule != decision.payment_plan:
            raise ExplanationError("decision does not match the selected option")
        _require_payments(decision, selected_option.number_of_payments)
        amount = _money(selected_option.payment_amount, preserve_precision=True)
        start = selected_option.first_payment_date.isoformat()
        text = (
            f"{prefix}Use {selected_option.number_of_payments} installments of "
            f"{currency} {amount}, starting {start}. This keeps the {currency} "
            f"{minimum} minimum protected."
        )
    elif method is PaymentMethod.PARTIAL_PAYMENT:
        _require_payments(decision, 2)
        first, second = decision.payment_plan
        text = (
            f"Pay {currency} {_money(first.amount)} today and {currency} "
            f"{_money(second.amount)} on {second.payment_date.isoformat()}. This "
            f"completes the request and keeps the {currency} {minimum} minimum "
            "protected."
        )
    elif method is PaymentMethod.WAIT:
        _require_payments(decision, 1)
        when = decision.payment_plan[0].payment_date.isoformat()
        text = (
            f"Pay {currency} {requested} in full on {when}. Paying earlier would "
            f"risk the {currency} {minimum} minimum."
        )
    elif method is PaymentMethod.NOT_RECOMMENDED:
        if decision.payment_plan or decision.spending_changes_needed:
            raise ExplanationError("not-recommended decisions cannot contain a plan")
        text = (
            f"Do not proceed by {request.desired_completion_date.isoformat()}: no "
            f"eligible available option protects the {currency} {minimum} minimum."
        )
        if decision.amount_safe_to_pay > 0:
            text += (
                f" Only {currency} {_money(decision.amount_safe_to_pay)} is safe "
                "to pay today."
            )
    else:  # pragma: no cover - closed enum
        raise ExplanationError(f"unsupported payment method: {method}")

    if request.request_type is RequestType.INVESTMENT:
        text += " This assesses affordability only, not investment performance."
    return text


def with_decision_explanation(
    decision: Decision,
    *,
    request: Request,
    profile: Profile,
    replay: ReplayResult | None = None,
    forecast_result: ForecastResult | None = None,
    selected_option: PaymentOption | None = None,
    verified_changes: Iterable[VerifiedSpendingChange] = (),
) -> Decision:
    """Return the immutable decision with its deterministic explanation attached."""
    return replace(
        decision,
        decision_explanation=build_decision_explanation(
            decision,
            request=request,
            profile=profile,
            replay=replay,
            forecast_result=forecast_result,
            selected_option=selected_option,
            verified_changes=verified_changes,
        ),
    )


def _change_prefix(changes: tuple[VerifiedSpendingChange, ...], currency: str) -> str:
    actions: list[str] = []
    for item in changes:
        description = _plain_description(item.source.description)
        if item.change.new_amount is None:
            actions.append(f"stop {description}")
        else:
            actions.append(
                f"reduce {description} to {currency} {_money(item.change.new_amount)}"
            )
    if not actions:
        return ""
    return "First " + "; then ".join(actions) + ". Then "


def _require_payments(decision: Decision, expected: int) -> None:
    if len(decision.payment_plan) != expected:
        raise ExplanationError(f"payment method requires exactly {expected} payments")


def _money(value: Decimal, *, preserve_precision: bool = False) -> str:
    try:
        return format_decimal(value, preserve_precision=preserve_precision)
    except ValueError as error:
        raise ExplanationError("explanation amount must be a finite Decimal") from error


def _plain_description(value: str) -> str:
    cleaned = " ".join(value.split())
    if not cleaned or any(character in cleaned for character in "|\r\n"):
        raise ExplanationError("change description is not safe for an explanation")
    return cleaned


generate_explanation = build_decision_explanation
explain_decision = build_decision_explanation

__all__ = [
    "ExplanationError",
    "build_decision_explanation",
    "explain_decision",
    "generate_explanation",
    "with_decision_explanation",
]
