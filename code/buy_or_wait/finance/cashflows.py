"""Build auditable future non-request cash flows for the 90-day horizon."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Iterable

from buy_or_wait.domain import Direction
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


class CashFlowReason(StrEnum):
    AUTHORITATIVE_FUTURE_EVENT = "authoritative_future_event"
    INFERRED_SUPPORTED_RECURRENCE = "inferred_supported_recurrence"
    EXPLICIT_SUPPRESSES_INFERRED = "explicit_future_row_suppresses_inferred_duplicate"


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
    horizon_days: int = 90,
) -> tuple[ProjectedCashFlow, ...]:
    return construct_future_cash_flows(
        resolved_events,
        request_date=request_date,
        policy=policy,
        message_evidence=message_evidence,
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
