"""Generate and independently verify recurring spending changes (Phase 10)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from itertools import combinations
from typing import Iterable, Sequence

from buy_or_wait.domain import (
    Direction,
    EventStatus,
    Flexibility,
    Profile,
    SpendingChange,
    SpendingChangeType,
)
from buy_or_wait.finance.lifecycle import (
    ResolvedEvent,
    ResolutionDisposition,
    ResolutionProvenance,
)
from buy_or_wait.finance.recurrence import RecurringSeries

MAX_SPENDING_CHANGES = 3
_STOPPABLE = {Flexibility.STOPPABLE, Flexibility.REDUCIBLE_OR_STOPPABLE}
_REDUCIBLE = {Flexibility.REDUCIBLE, Flexibility.REDUCIBLE_OR_STOPPABLE}


class SpendingChangeValidationError(ValueError):
    """A proposed spending change is not authorized by source facts."""


@dataclass(frozen=True, slots=True)
class SpendingChangeSource:
    event_id: str
    description: str
    category: str
    flexibility: Flexibility
    baseline_amount: Decimal
    minimum_allowed_amount: Decimal | None
    source_event_ids: tuple[str, ...]
    provenance_ids: tuple[str, ...]
    provenance: tuple[ResolutionProvenance, ...]


@dataclass(frozen=True, slots=True)
class VerifiedSpendingChange:
    change: SpendingChange
    source: SpendingChangeSource
    effective_from: date
    saving_per_occurrence: Decimal


@dataclass(frozen=True, slots=True)
class SpendingChangeCandidate:
    verified_changes: tuple[VerifiedSpendingChange, ...]

    @property
    def changes(self) -> tuple[SpendingChange, ...]:
        return tuple(item.change for item in self.verified_changes)

    @property
    def actions(self) -> tuple[SpendingChange, ...]:
        return self.changes

    @property
    def sources(self) -> tuple[SpendingChangeSource, ...]:
        return tuple(item.source for item in self.verified_changes)

    @property
    def maximum_saving_per_occurrence(self) -> Decimal:
        return sum(
            (item.saving_per_occurrence for item in self.verified_changes), Decimal(0)
        )


def generate_spending_change_candidates(
    profile: Profile,
    recurring_series: Iterable[RecurringSeries],
    resolved_events: Iterable[ResolvedEvent],
    *,
    request_date: date,
) -> tuple[SpendingChangeCandidate, ...]:
    """Enumerate deterministic combinations without choosing a payment plan."""
    series, rows = tuple(recurring_series), tuple(resolved_events)
    sources = _sources(profile, series, rows, request_date)
    protected = _categories(profile.expense_categories_to_protect)
    reduce = _categories(profile.expense_categories_user_is_willing_to_reduce)
    stop = _categories(profile.expense_categories_user_is_willing_to_stop)
    atomic: list[SpendingChange] = []
    for source in sources:
        category = _category(source.category)
        if category in protected or source.baseline_amount <= 0:
            continue
        if source.flexibility in _STOPPABLE and category in stop:
            atomic.append(SpendingChange(SpendingChangeType.STOP, source.event_id))
        floor = source.minimum_allowed_amount
        if (
            source.flexibility in _REDUCIBLE
            and category in reduce
            and floor is not None
            and floor < source.baseline_amount
        ):
            atomic.append(
                SpendingChange(SpendingChangeType.REDUCE_TO, source.event_id, floor)
            )
    atomic.sort(key=_change_key)
    result: list[SpendingChangeCandidate] = []
    for size in range(1, min(MAX_SPENDING_CHANGES, len(atomic)) + 1):
        for selected in combinations(atomic, size):
            if len({change.event_id for change in selected}) != size:
                continue
            verified = verify_spending_changes(
                selected,
                profile,
                recurring_series=series,
                resolved_events=rows,
                request_date=request_date,
            )
            result.append(SpendingChangeCandidate(verified))
    return tuple(result)


def verify_spending_change(
    change: SpendingChange,
    profile: Profile,
    *,
    recurring_series: Iterable[RecurringSeries],
    resolved_events: Iterable[ResolvedEvent],
    request_date: date,
) -> VerifiedSpendingChange:
    return verify_spending_changes(
        (change,),
        profile,
        recurring_series=recurring_series,
        resolved_events=resolved_events,
        request_date=request_date,
    )[0]


def verify_spending_changes(
    changes: Iterable[SpendingChange],
    profile: Profile,
    *,
    recurring_series: Iterable[RecurringSeries],
    resolved_events: Iterable[ResolvedEvent],
    request_date: date,
) -> tuple[VerifiedSpendingChange, ...]:
    """Fail closed for external actions, including ones generation omits."""
    proposed = tuple(changes)
    if not 1 <= len(proposed) <= MAX_SPENDING_CHANGES:
        raise SpendingChangeValidationError("one to three actions are required")
    ids = [item.event_id for item in proposed]
    if any(not item for item in ids) or len(set(ids)) != len(ids):
        raise SpendingChangeValidationError(
            "duplicate targets and same-event stop/reduce conflicts are invalid"
        )
    source_by_id = {
        item.event_id: item
        for item in _sources(
            profile, tuple(recurring_series), tuple(resolved_events), request_date
        )
    }
    protected = _categories(profile.expense_categories_to_protect)
    reduce = _categories(profile.expense_categories_user_is_willing_to_reduce)
    stop = _categories(profile.expense_categories_user_is_willing_to_stop)
    result: list[VerifiedSpendingChange] = []
    for change in proposed:
        source = source_by_id.get(change.event_id)
        if source is None:
            raise SpendingChangeValidationError(
                f"{change.event_id} is not the latest source of a recurring expense"
            )
        category = _category(source.category)
        if category in protected:
            raise SpendingChangeValidationError(
                f"{source.category} is protected"
            )
        if change.change_type is SpendingChangeType.STOP:
            if change.new_amount is not None:
                raise SpendingChangeValidationError("stop cannot specify new_amount")
            if source.flexibility not in _STOPPABLE or category not in stop:
                raise SpendingChangeValidationError("stop is not permitted")
            saving = source.baseline_amount
        elif change.change_type is SpendingChangeType.REDUCE_TO:
            amount = _decimal(change.new_amount, "new_amount")
            floor = source.minimum_allowed_amount
            if floor is None:
                raise SpendingChangeValidationError("reduction floor is missing")
            if source.flexibility not in _REDUCIBLE or category not in reduce:
                raise SpendingChangeValidationError("reduction is not permitted")
            if amount < floor or amount > source.baseline_amount:
                raise SpendingChangeValidationError(
                    "reduction must be between floor and baseline"
                )
            saving = source.baseline_amount - amount
        else:
            raise SpendingChangeValidationError("unsupported change type")
        result.append(VerifiedSpendingChange(change, source, request_date, saving))
    return tuple(result)


def _sources(
    profile: Profile,
    series: Sequence[RecurringSeries],
    rows: Sequence[ResolvedEvent],
    request_date: date,
) -> tuple[SpendingChangeSource, ...]:
    by_id = {row.event_id: row for row in rows}
    if len(by_id) != len(rows):
        raise SpendingChangeValidationError("duplicate resolved event_id")
    result: list[SpendingChangeSource] = []
    targets: set[str] = set()
    for item in series:
        if item.key.user_id != profile.user_id:
            raise SpendingChangeValidationError("series belongs to another user")
        if item.key.direction is not Direction.DEBIT:
            continue
        if not item.observations:
            raise SpendingChangeValidationError("series has no observations")
        latest = max(item.observations, key=lambda value: (value.when, value.event_id))
        row = by_id.get(latest.event_id)
        event = row.source_event if row else None
        if (
            row is None
            or event is None
            or row.effective_status is not EventStatus.SETTLED
            or row.disposition is not ResolutionDisposition.HISTORICAL_CASH_EVIDENCE
            or not row.supports_recurrence
            or row.cash_flow_date is None
            or row.cash_flow_date >= request_date
            or event.direction is not Direction.DEBIT
            or event.user_id != profile.user_id
            or _category(event.category) != _category(item.key.category)
            or event.flexibility is not item.key.flexibility
        ):
            raise SpendingChangeValidationError("invalid recurring expense source")
        if event.event_id in targets:
            raise SpendingChangeValidationError("duplicate recurring-series target")
        targets.add(event.event_id)
        baseline = _decimal(item.projected_amount, "baseline")
        floor = _floor_in_home_currency(row, profile)
        provenance_ids = tuple(
            dict.fromkeys(
                value
                for observation in item.observations
                for value in observation.provenance_ids
            )
        )
        result.append(
            SpendingChangeSource(
                event.event_id,
                event.description,
                event.category,
                event.flexibility,
                baseline,
                floor,
                item.source_event_ids,
                provenance_ids,
                row.provenance,
            )
        )
    return tuple(sorted(result, key=lambda value: value.event_id))


def _floor_in_home_currency(row: ResolvedEvent, profile: Profile) -> Decimal | None:
    floor = row.source_event.minimum_allowed_amount
    if floor is None:
        return None
    floor = _decimal(floor, "minimum_allowed_amount")
    if row.effective_currency is profile.home_currency:
        return floor  # Preserve the source Decimal exponent/format.
    if row.effective_amount is None or row.amount_home_currency is None:
        raise SpendingChangeValidationError("cannot convert reduction floor")
    if row.effective_amount == 0:
        if floor == 0:
            return Decimal(0)
        raise SpendingChangeValidationError("cannot convert reduction floor")
    return floor * (row.amount_home_currency / row.effective_amount)


def _decimal(value: Decimal | None, name: str) -> Decimal:
    if not isinstance(value, Decimal) or not value.is_finite() or value < 0:
        raise SpendingChangeValidationError(f"{name} must be a non-negative Decimal")
    return value


def _category(value: str) -> str:
    return value.strip().casefold()


def _categories(values: Iterable[str]) -> frozenset[str]:
    return frozenset(_category(value) for value in values)


def _change_key(change: SpendingChange) -> tuple[str, str, str]:
    amount = "" if change.new_amount is None else format(change.new_amount, "f")
    return change.event_id, change.change_type.value, amount


enumerate_spending_change_candidates = generate_spending_change_candidates

__all__ = [
    "MAX_SPENDING_CHANGES",
    "SpendingChangeCandidate",
    "SpendingChangeSource",
    "SpendingChangeValidationError",
    "VerifiedSpendingChange",
    "enumerate_spending_change_candidates",
    "generate_spending_change_candidates",
    "verify_spending_change",
    "verify_spending_changes",
]
