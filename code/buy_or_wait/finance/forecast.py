"""Auditable daily balance forecasting and independent payment replay."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from decimal import Decimal
from enum import StrEnum
from typing import Iterable, Mapping, TypeAlias

from buy_or_wait.domain import Direction, Payment, SpendingChange, SpendingChangeType
from buy_or_wait.finance.cashflows import ProjectedCashFlow
from buy_or_wait.finance.lifecycle import ResolutionProvenance
from buy_or_wait.finance.recurrence_policies import (
    ForecastPolicy,
    SameDayOrdering,
    require_selected_policy,
)


HORIZON_DAYS = 90
LedgerStep: TypeAlias = tuple[
    str,
    "LedgerEntryKind",
    Decimal,
    Direction,
    tuple[ResolutionProvenance, ...],
]


class LedgerEntryKind(StrEnum):
    CASH_FLOW = "cash_flow"
    PAYMENT = "payment"


@dataclass(frozen=True, slots=True)
class LedgerEntry:
    entry_id: str
    kind: LedgerEntryKind
    amount: Decimal
    direction: Direction
    balance_after: Decimal
    provenance: tuple[ResolutionProvenance, ...] = ()

    @property
    def signed_amount(self) -> Decimal:
        return self.amount if self.direction is Direction.CREDIT else -self.amount


@dataclass(frozen=True, slots=True)
class DailyLedgerRow:
    ledger_date: date
    opening_balance: Decimal
    entries: tuple[LedgerEntry, ...]
    closing_balance: Decimal
    running_minimum: Decimal

    @property
    def flows(self) -> tuple[LedgerEntry, ...]:
        return self.entries


@dataclass(frozen=True, slots=True)
class ReplayResult:
    ledger: tuple[DailyLedgerRow, ...]
    minimum_projected_balance: Decimal
    is_safe: bool
    failure_date: date | None = None
    failure_entry_id: str | None = None

    @property
    def daily_balances(self) -> tuple[tuple[date, Decimal], ...]:
        return tuple((row.ledger_date, row.closing_balance) for row in self.ledger)


@dataclass(frozen=True, slots=True)
class BaselineForecast:
    baseline_replay: ReplayResult
    amount_safe_to_pay: Decimal
    earliest_date_for_full_payment: date | None

    @property
    def ledger(self) -> tuple[DailyLedgerRow, ...]:
        return self.baseline_replay.ledger


def replay_forecast(
    *,
    starting_balance: Decimal,
    minimum_balance_to_keep: Decimal,
    request_date: date,
    cash_flows: Iterable[ProjectedCashFlow],
    policy: ForecastPolicy | None,
    payments: Iterable[Payment] = (),
    changes: Iterable[SpendingChange] | Mapping[str, Decimal | None] = (),
    horizon_days: int = HORIZON_DAYS,
) -> ReplayResult:
    """Independently replay arbitrary payments/changes over an inclusive horizon."""
    selected = require_selected_policy(policy)
    start = _finite_nonnegative(starting_balance, "starting_balance")
    minimum = _finite_nonnegative(minimum_balance_to_keep, "minimum_balance_to_keep")
    if horizon_days < 0:
        raise ValueError("horizon_days must be non-negative")
    end = request_date + timedelta(days=horizon_days)
    flows, payment_rows = tuple(cash_flows), tuple(payments)
    _validate_inputs(flows, payment_rows, request_date, end)
    change_map = _normalize_changes(changes)
    flows_by_date: dict[date, list[ProjectedCashFlow]] = {}
    for flow in flows:
        flows_by_date.setdefault(flow.flow_date, []).append(flow)
    payments_by_date: dict[date, list[tuple[int, Payment]]] = {}
    for ordinal, payment in enumerate(payment_rows):
        payments_by_date.setdefault(payment.payment_date, []).append((ordinal, payment))

    balance, running_minimum = start, start
    failure_date: date | None = request_date if start < minimum else None
    failure_entry: str | None = "opening_balance" if failure_date else None
    ledger: list[DailyLedgerRow] = []
    for offset in range(horizon_days + 1):
        day, opening = request_date + timedelta(days=offset), balance
        entries: list[LedgerEntry] = []
        for entry_id, kind, amount, direction, provenance in _ordered_steps(
            flows_by_date.get(day, ()),
            payments_by_date.get(day, ()),
            selected.same_day_ordering,
            change_map,
        ):
            balance += amount if direction is Direction.CREDIT else -amount
            running_minimum = min(running_minimum, balance)
            entries.append(
                LedgerEntry(entry_id, kind, amount, direction, balance, provenance)
            )
            if failure_date is None and balance < minimum:
                failure_date, failure_entry = day, entry_id
        ledger.append(
            DailyLedgerRow(day, opening, tuple(entries), balance, running_minimum)
        )
    return ReplayResult(
        tuple(ledger),
        running_minimum,
        failure_date is None,
        failure_date,
        failure_entry,
    )


def build_daily_ledger(**kwargs: object) -> tuple[DailyLedgerRow, ...]:
    return replay_forecast(**kwargs).ledger  # type: ignore[arg-type]


def calculate_baseline_forecast(
    *,
    current_available_balance: Decimal,
    minimum_balance_to_keep: Decimal,
    request_date: date,
    requested_amount: Decimal,
    cash_flows: Iterable[ProjectedCashFlow],
    policy: ForecastPolicy | None,
    horizon_days: int = HORIZON_DAYS,
) -> BaselineForecast:
    """Calculate pre-change safe capacity and first independently safe full date."""
    requested = _finite_nonnegative(requested_amount, "requested_amount")
    flows = tuple(cash_flows)
    baseline = replay_forecast(
        starting_balance=current_available_balance,
        minimum_balance_to_keep=minimum_balance_to_keep,
        request_date=request_date,
        cash_flows=flows,
        policy=policy,
        horizon_days=horizon_days,
    )
    capacity = baseline.minimum_projected_balance - minimum_balance_to_keep
    safe = min(requested, max(Decimal(0), capacity))
    earliest = earliest_safe_full_payment_date(
        current_available_balance=current_available_balance,
        minimum_balance_to_keep=minimum_balance_to_keep,
        request_date=request_date,
        requested_amount=requested,
        cash_flows=flows,
        policy=policy,
        horizon_days=horizon_days,
    )
    return BaselineForecast(baseline, safe, earliest)


def earliest_safe_full_payment_date(
    *,
    current_available_balance: Decimal,
    minimum_balance_to_keep: Decimal,
    request_date: date,
    requested_amount: Decimal,
    cash_flows: Iterable[ProjectedCashFlow],
    policy: ForecastPolicy | None,
    horizon_days: int = HORIZON_DAYS,
) -> date | None:
    """Return the first date whose one-payment full replay is safe."""
    selected = require_selected_policy(policy)
    requested = _finite_nonnegative(requested_amount, "requested_amount")
    flows = tuple(cash_flows)
    for offset in range(horizon_days + 1):
        candidate = request_date + timedelta(days=offset)
        result = replay_forecast(
            starting_balance=current_available_balance,
            minimum_balance_to_keep=minimum_balance_to_keep,
            request_date=request_date,
            cash_flows=flows,
            policy=selected,
            payments=(Payment(candidate, requested),),
            horizon_days=horizon_days,
        )
        if result.is_safe:
            return candidate
    return None


def optimized_safe_payment_on_date(
    *,
    baseline_replay: ReplayResult,
    payment_date: date,
    minimum_balance_to_keep: Decimal,
    requested_amount: Decimal,
) -> Decimal:
    """Suffix-min shortcut for cross-checking against the replay verifier."""
    all_rows = baseline_replay.ledger
    rows = tuple(row for row in all_rows if row.ledger_date >= payment_date)
    if not rows:
        raise ValueError("payment_date is outside the replay horizon")
    payment_row = rows[0]
    if payment_row.running_minimum < minimum_balance_to_keep:
        return Decimal(0)
    balances = [payment_row.closing_balance]
    for row in rows[1:]:
        balances.append(row.opening_balance)
        balances.extend(entry.balance_after for entry in row.entries)
    capacity = min(balances) - minimum_balance_to_keep
    return min(requested_amount, max(Decimal(0), capacity))


def _ordered_steps(
    flows: Iterable[ProjectedCashFlow],
    payments: Iterable[tuple[int, Payment]],
    ordering: SameDayOrdering,
    changes: Mapping[str, Decimal | None],
) -> tuple[LedgerStep, ...]:
    debits: list[LedgerStep] = []
    credits: list[LedgerStep] = []
    for flow in sorted(flows, key=lambda item: item.cash_flow_id):
        amount = _changed_amount(flow, changes)
        if amount == 0:
            continue
        step = (
            flow.cash_flow_id,
            LedgerEntryKind.CASH_FLOW,
            amount,
            flow.direction,
            flow.provenance,
        )
        (credits if flow.direction is Direction.CREDIT else debits).append(step)
    payment_steps: list[LedgerStep] = [
        (
            f"payment:{ordinal}",
            LedgerEntryKind.PAYMENT,
            payment.amount,
            Direction.DEBIT,
            (),
        )
        for ordinal, payment in sorted(payments, key=lambda item: item[0])
    ]
    if ordering is SameDayOrdering.DEBITS_BEFORE_CREDITS:
        return tuple((*debits, *credits, *payment_steps))
    return tuple((*credits, *debits, *payment_steps))


def _normalize_changes(
    values: Iterable[SpendingChange] | Mapping[str, Decimal | None],
) -> dict[str, Decimal | None]:
    if isinstance(values, Mapping):
        result = dict(values)
    else:
        result = {}
        for change in values:
            if change.event_id in result:
                raise ValueError(f"duplicate spending change: {change.event_id}")
            if change.change_type is SpendingChangeType.STOP:
                if change.new_amount is not None:
                    raise ValueError("stop change cannot specify new_amount")
                result[change.event_id] = None
            elif change.change_type is SpendingChangeType.REDUCE_TO:
                if change.new_amount is None:
                    raise ValueError("reduce_to change requires new_amount")
                result[change.event_id] = _finite_nonnegative(
                    change.new_amount, "new_amount"
                )
    for event_id, amount in result.items():
        if not event_id:
            raise ValueError("spending change event id must not be empty")
        if amount is not None:
            _finite_nonnegative(amount, "new_amount")
    return result


def _changed_amount(
    flow: ProjectedCashFlow, changes: Mapping[str, Decimal | None]
) -> Decimal:
    matches = [changes[source] for source in flow.source_event_ids if source in changes]
    if not matches:
        return flow.amount
    if len(matches) > 1 and len(set(matches)) > 1:
        raise ValueError(f"conflicting changes for {flow.cash_flow_id}")
    replacement = matches[0]
    if replacement is None:
        return Decimal(0)
    if replacement > flow.amount:
        raise ValueError(f"spending change increases {flow.cash_flow_id}")
    return replacement


def _validate_inputs(
    flows: tuple[ProjectedCashFlow, ...],
    payments: tuple[Payment, ...],
    start: date,
    end: date,
) -> None:
    for flow in flows:
        if not start <= flow.flow_date <= end:
            raise ValueError(f"cash flow outside forecast horizon: {flow.cash_flow_id}")
        _finite_nonnegative(flow.amount, f"cash flow {flow.cash_flow_id}")
        if flow.direction not in {Direction.DEBIT, Direction.CREDIT}:
            raise ValueError(f"non-cash flow in forecast: {flow.cash_flow_id}")
    for payment in payments:
        if not start <= payment.payment_date <= end:
            raise ValueError("payment outside forecast horizon")
        _finite_nonnegative(payment.amount, "payment amount")


def _finite_nonnegative(value: Decimal, name: str) -> Decimal:
    result = Decimal(value)
    if not result.is_finite() or result < 0:
        raise ValueError(f"{name} must be a finite non-negative Decimal")
    return result


forecast_baseline = calculate_baseline_forecast
replay = replay_forecast


__all__ = [
    "BaselineForecast",
    "DailyLedgerRow",
    "HORIZON_DAYS",
    "LedgerEntry",
    "LedgerEntryKind",
    "ReplayResult",
    "build_daily_ledger",
    "calculate_baseline_forecast",
    "earliest_safe_full_payment_date",
    "forecast_baseline",
    "optimized_safe_payment_on_date",
    "replay",
    "replay_forecast",
]
