"""Build auditable future non-request cash flows for the 90-day horizon."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Iterable

from buy_or_wait.domain import Currency, Direction, EventType, ExchangeRate, Flexibility
from buy_or_wait.evidence.messages import ExtractedMessageEvidence
from buy_or_wait.evidence.schemas import MessageEvidenceItem
from buy_or_wait.finance.lifecycle import (
    ProvenanceKind,
    ResolvedEvent,
    ResolutionDisposition,
    ResolutionProvenance,
)
from buy_or_wait.finance.recurrence import (
    InferredOccurrence,
    SeriesKey,
    infer_recurrence_series,
    project_recurrences,
    semantic_description_family,
)
from buy_or_wait.finance.recurrence_policies import (
    ForecastPolicy,
    SameDayOrdering,
    require_selected_policy,
)
from buy_or_wait.money import convert_to_home_currency


class CashFlowReason(StrEnum):
    AUTHORITATIVE_FUTURE_EVENT = "authoritative_future_event"
    INFERRED_SUPPORTED_RECURRENCE = "inferred_supported_recurrence"
    EXPLICIT_SUPPRESSES_INFERRED = "explicit_future_row_suppresses_inferred_duplicate"
    AUTHORITATIVE_MESSAGE_CREDIT = "authoritative_confirmed_message_credit"


@dataclass(frozen=True, slots=True)
class ProjectedCashFlow:
    cash_flow_id: str
    flow_date: date
    amount: Decimal
    direction: Direction
    category: str
    description: str
    source_event_ids: tuple[str, ...]
    source_message_ids: tuple[str, ...]
    provenance: tuple[ResolutionProvenance, ...]
    reason_codes: tuple[str, ...]
    synthetic: bool
    series_key: SeriesKey

    @property
    def signed_amount(self) -> Decimal:
        return self.amount if self.direction is Direction.CREDIT else -self.amount

    @property
    def reason_code(self) -> str:
        return self.reason_codes[-1]


@dataclass(frozen=True, slots=True)
class SuppressedCashFlow:
    inferred_cash_flow_id: str
    explicit_cash_flow_id: str
    flow_date: date
    reason_code: str = CashFlowReason.EXPLICIT_SUPPRESSES_INFERRED.value


@dataclass(frozen=True, slots=True)
class CashFlowProjection:
    request_date: date
    horizon_end: date
    policy_hash: str
    cash_flows: tuple[ProjectedCashFlow, ...]
    suppressed: tuple[SuppressedCashFlow, ...]


def construct_future_cash_flows(
    resolved_events: Iterable[ResolvedEvent],
    *,
    request_date: date,
    policy: ForecastPolicy | None,
    message_evidence: Iterable[object] = (),
    user_id: str | None = None,
    home_currency: Currency | None = None,
    exchange_rates: Iterable[ExchangeRate] = (),
    horizon_days: int = 90,
) -> CashFlowProjection:
    """Combine explicit lifecycle rows and inferred recurrence, fail closed."""
    selected_policy = require_selected_policy(policy)
    if horizon_days < 0:
        raise ValueError("horizon_days must be non-negative")
    rows, evidence = tuple(resolved_events), tuple(message_evidence)
    end = request_date + timedelta(days=horizon_days)
    explicit = [
        _explicit(row)
        for row in rows
        if row.disposition is ResolutionDisposition.PROJECTED_CASH_FLOW
        and row.cash_flow_date is not None
        and request_date <= row.cash_flow_date <= end
    ]
    explicit.extend(
        _confirmed_message_credits(
            evidence,
            request_date=request_date,
            horizon_end=end,
            user_id=user_id,
            home_currency=home_currency,
            exchange_rates=tuple(exchange_rates),
            existing=explicit,
        )
    )
    inferred_series = infer_recurrence_series(
        rows,
        request_date=request_date,
        policy=selected_policy,
        message_evidence=evidence,  # type: ignore[arg-type]
    )
    inferred = [
        _inferred(item)
        for item in project_recurrences(
            inferred_series,
            start_date=request_date,
            end_date=end,
            policy=selected_policy,
        )
    ]
    explicit_by_match = {
        (item.flow_date, item.series_key): index for index, item in enumerate(explicit)
    }
    retained, suppressed = [], []
    for item in inferred:
        index = explicit_by_match.get((item.flow_date, item.series_key))
        if index is None:
            retained.append(item)
            continue
        target = explicit[index]
        explicit[index] = replace(
            target,
            source_event_ids=tuple(
                dict.fromkeys((*target.source_event_ids, *item.source_event_ids))
            ),
            source_message_ids=tuple(
                dict.fromkeys((*target.source_message_ids, *item.source_message_ids))
            ),
            provenance=(
                *target.provenance,
                ResolutionProvenance(
                    ProvenanceKind.RULE,
                    item.cash_flow_id,
                    "explicit row retained; matching inferred occurrence suppressed",
                ),
            ),
            reason_codes=(
                *target.reason_codes,
                CashFlowReason.EXPLICIT_SUPPRESSES_INFERRED.value,
            ),
        )
        suppressed.append(
            SuppressedCashFlow(item.cash_flow_id, target.cash_flow_id, item.flow_date)
        )
    flows = sort_cash_flows((*explicit, *retained), selected_policy.same_day_ordering)
    return CashFlowProjection(
        request_date, end, selected_policy.sha256, flows, tuple(suppressed)
    )


def build_future_cash_flows(
    resolved_events: Iterable[ResolvedEvent],
    *,
    request_date: date,
    policy: ForecastPolicy | None,
    message_evidence: Iterable[object] = (),
    user_id: str | None = None,
    home_currency: Currency | None = None,
    exchange_rates: Iterable[ExchangeRate] = (),
    horizon_days: int = 90,
) -> tuple[ProjectedCashFlow, ...]:
    return construct_future_cash_flows(
        resolved_events,
        request_date=request_date,
        policy=policy,
        message_evidence=message_evidence,
        user_id=user_id,
        home_currency=home_currency,
        exchange_rates=exchange_rates,
        horizon_days=horizon_days,
    ).cash_flows


build_cash_flows = build_future_cash_flows
build_future_non_request_cash_flows = build_future_cash_flows
FutureCashFlow = ProjectedCashFlow


def sort_cash_flows(
    values: Iterable[ProjectedCashFlow], ordering: SameDayOrdering
) -> tuple[ProjectedCashFlow, ...]:
    debit_rank = 0 if ordering is SameDayOrdering.DEBITS_BEFORE_CREDITS else 1
    return tuple(
        sorted(
            values,
            key=lambda item: (
                item.flow_date,
                debit_rank if item.direction is Direction.DEBIT else 1 - debit_rank,
                item.cash_flow_id,
            ),
        )
    )


def _key(row: ResolvedEvent) -> SeriesKey:
    event = row.source_event
    return SeriesKey(
        event.user_id,
        event.direction,
        event.category,
        event.event_type,
        event.currency,
        event.flexibility,
        semantic_description_family(event.description, event.category),
    )


def _explicit(row: ResolvedEvent) -> ProjectedCashFlow:
    assert row.cash_flow_date is not None and row.amount_home_currency is not None
    return ProjectedCashFlow(
        f"explicit:{row.event_id}",
        row.cash_flow_date,
        row.amount_home_currency,
        row.source_event.direction,
        row.source_event.category,
        row.source_event.description,
        row.source_event_ids,
        row.source_message_ids,
        row.provenance,
        (row.reason_code.value, CashFlowReason.AUTHORITATIVE_FUTURE_EVENT.value),
        False,
        _key(row),
    )


def _confirmed_message_credits(
    evidence: tuple[object, ...],
    *,
    request_date: date,
    horizon_end: date,
    user_id: str | None,
    home_currency: Currency | None,
    exchange_rates: tuple[ExchangeRate, ...],
    existing: list[ProjectedCashFlow],
) -> tuple[ProjectedCashFlow, ...]:
    """Materialize only explicitly confirmed, dated message-only salary credits."""

    result: list[ProjectedCashFlow] = []
    for value in evidence:
        item = (
            value.item
            if isinstance(value, ExtractedMessageEvidence)
            else value
            if isinstance(value, MessageEvidenceItem)
            else None
        )
        if item is None or item.fact_type != "first_salary_confirmed":
            continue
        if (
            item.related_event_id is not None
            or item.cash_state != "confirmed_credit"
            or item.amount is None
            or item.currency is None
            or item.settlement_date is None
        ):
            continue
        if user_id is None or home_currency is None:
            raise ValueError(
                "message-confirmed salary requires user_id and home_currency"
            )
        when = item.settlement_date
        if not request_date <= when <= horizon_end:
            continue
        currency = Currency(item.currency)
        amount = convert_to_home_currency(
            Decimal(item.amount),
            currency,
            home_currency,
            when,
            exchange_rates,
            event_id=item.message_id,
        )
        if any(
            flow.direction is Direction.CREDIT
            and flow.flow_date == when
            and flow.amount == amount
            for flow in (*existing, *result)
        ):
            continue
        key = SeriesKey(
            user_id,
            Direction.CREDIT,
            "salary",
            EventType.INCOME,
            currency,
            Flexibility.FIXED,
            semantic_description_family("Payroll credit", "salary"),
        )
        result.append(
            ProjectedCashFlow(
                f"message-confirmed:{item.message_id}",
                when,
                amount,
                Direction.CREDIT,
                "salary",
                "Confirmed salary from message evidence",
                (),
                (item.message_id,),
                (
                    ResolutionProvenance(
                        ProvenanceKind.MESSAGE,
                        item.message_id,
                        "explicit confirmed salary credit",
                    ),
                ),
                (CashFlowReason.AUTHORITATIVE_MESSAGE_CREDIT.value,),
                False,
                key,
            )
        )
    return tuple(result)


def _inferred(item: InferredOccurrence) -> ProjectedCashFlow:
    provenance = (
        tuple(
            ResolutionProvenance(
                ProvenanceKind.EVENT, event_id, "supports inferred recurrence"
            )
            for event_id in item.source_event_ids
        )
        + tuple(
            ResolutionProvenance(
                ProvenanceKind.MESSAGE, message_id, "effective recurrence amendment"
            )
            for message_id in item.source_message_ids
        )
        + (
            ResolutionProvenance(
                ProvenanceKind.RULE, item.occurrence_id, item.reason_code
            ),
        )
    )
    return ProjectedCashFlow(
        item.occurrence_id,
        item.flow_date,
        item.amount,
        item.series_key.direction,
        item.series_key.category,
        f"Inferred {item.series_key.description_family}",
        item.source_event_ids,
        item.source_message_ids,
        provenance,
        (CashFlowReason.INFERRED_SUPPORTED_RECURRENCE.value,),
        True,
        item.series_key,
    )


__all__ = [
    "CashFlowProjection",
    "CashFlowReason",
    "ProjectedCashFlow",
    "SuppressedCashFlow",
    "FutureCashFlow",
    "build_cash_flows",
    "build_future_cash_flows",
    "build_future_non_request_cash_flows",
    "construct_future_cash_flows",
    "sort_cash_flows",
]
