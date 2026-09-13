"""Phase 11 deterministic candidate ranking and final decision selection."""

from __future__ import annotations

from datetime import date
import re
from typing import Iterable

from buy_or_wait.domain import AffordabilityStatus, Decision, PaymentMethod, Request
from buy_or_wait.finance.forecast import BaselineForecast
from buy_or_wait.finance.plans import VerifiedPlanCandidate


# Used only at official tie-break 6. Real numeric option IDs sort before a
# non-option candidate when all first five criteria are identical.
NON_OPTION_ID_SENTINEL = 2**63 - 1


def candidate_ranking_key(candidate: VerifiedPlanCandidate) -> tuple[object, ...]:
    """Six official criteria, then only the documented deterministic ties."""
    plan = candidate.plan
    option_number = (
        NON_OPTION_ID_SENTINEL
        if plan.payment_option_id is None
        else _numeric_suffix(plan.payment_option_id)
    )
    event_order = tuple(
        sorted(_event_id_key(change.event_id) for change in plan.spending_changes)
    )
    return (
        0 if candidate.completes_full_request_by_deadline else 1,
        0 if not plan.spending_changes else 1,
        plan.total_payable_amount,
        plan.payments[0].payment_date if plan.payments else date.max,
        len(plan.payments),
        option_number,
        len(plan.spending_changes),
        candidate.lifestyle_savings,
        event_order,
    )


def rank_candidate_plans(
    candidates: Iterable[VerifiedPlanCandidate],
) -> tuple[VerifiedPlanCandidate, ...]:
    """Retain safe eligible candidates and apply the exact ranking order."""
    eligible = (
        candidate
        for candidate in candidates
        if candidate.replay.is_safe and candidate.completes_full_request_by_deadline
    )
    return tuple(sorted(eligible, key=candidate_ranking_key))


def select_final_decision(
    *,
    request: Request,
    baseline: BaselineForecast,
    candidates: Iterable[VerifiedPlanCandidate],
) -> Decision:
    """Choose the best candidate, using fallback only when none survives."""
    ranked = rank_candidate_plans(candidates)
    if not ranked:
        return Decision(
            request.request_id,
            baseline.amount_safe_to_pay,
            AffordabilityStatus.NOT_AFFORDABLE,
            PaymentMethod.NOT_RECOMMENDED,
            (),
            baseline.earliest_date_for_full_payment,
            (),
            "No safe eligible payment plan was found within the forecast period.",
        )
    best = ranked[0]
    plan = best.plan
    method = plan.payment_method.value.replace("_", " ")
    minimum = format(best.replay.minimum_projected_balance, "f")
    return Decision(
        request.request_id,
        baseline.amount_safe_to_pay,
        best.affordability_status,
        plan.payment_method,
        plan.payments,
        baseline.earliest_date_for_full_payment,
        plan.spending_changes,
        "The verified "
        f"{method} plan keeps the projected balance at or above {minimum}.",
    )


def _numeric_suffix(value: str) -> int:
    match = re.search(r"(\d+)$", value)
    if match is None:
        raise ValueError("payment_option_id must end in a numeric identifier")
    return int(match.group(1))


def _event_id_key(value: str) -> tuple[int, str, str]:
    match = re.search(r"(\d+)$", value)
    number = NON_OPTION_ID_SENTINEL if match is None else int(match.group(1))
    return number, value.casefold(), value


rank_candidates = rank_candidate_plans
select_decision = select_final_decision

__all__ = [
    "NON_OPTION_ID_SENTINEL",
    "candidate_ranking_key",
    "rank_candidate_plans",
    "rank_candidates",
    "select_decision",
    "select_final_decision",
]
