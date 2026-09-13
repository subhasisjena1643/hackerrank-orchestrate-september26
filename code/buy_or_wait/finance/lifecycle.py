"""Deterministic financial-event lifecycle resolution (Phase 7 only)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from decimal import Decimal
from enum import StrEnum
from typing import Iterable, Mapping, Protocol, Sequence, TypeAlias

from buy_or_wait.domain import (
    Currency,
    Direction,
    EventStatus,
    EventType,
    ExchangeRate,
    FinancialEvent,
    SourceType,
)
from buy_or_wait.evidence.messages import ExtractedMessageEvidence
from buy_or_wait.evidence.schemas import MessageEvidenceItem
from buy_or_wait.money import MoneyError, convert_to_home_currency


class LifecycleResolutionError(ValueError):
    """Lifecycle inputs cannot be resolved safely."""


class ResolutionDisposition(StrEnum):
    HISTORICAL_CASH_EVIDENCE = "historical_cash_evidence"
    PROJECTED_CASH_FLOW = "projected_cash_flow"
    EXCLUDED = "excluded"
    REQUIRES_AMOUNT = "requires_amount"


class ResolutionReason(StrEnum):
    HISTORICAL_SETTLED = "historical_settled_evidence"
    FUTURE_SETTLED = "future_settled_cash"
    SETTLED_INVESTMENT_SALE = "settled_investment_sale_cash"
    CANCELLED = "cancelled_zero"
    FAILED = "failed_zero"
    PENDING_CREDIT = "pending_credit_zero"
    PENDING_DEBIT = "pending_debit_reserved"
    SCHEDULED_DEBIT = "scheduled_debit_included"
    CONFIRMED_SALARY = "confirmed_scheduled_salary"
    UNCONFIRMED_CREDIT = "unconfirmed_scheduled_credit_zero"
    UNREALIZED = "unrealized_non_cash"
    NON_CASH = "non_cash_direction"
    INTERNAL_TRANSFER = "verified_internal_transfer_net_zero"
    PROVEN_DUPLICATE = "proven_duplicate_suppressed"
    UNRESOLVED_DUPLICATE = "unresolved_duplicate_safer_interpretation"
    EXPLICIT_CANCELLATION = "explicit_cancellation"
    EXPLICIT_SETTLEMENT = "explicit_settlement"
    EXPLICIT_AMENDMENT = "explicit_amendment"
    NEWER_SAME_SOURCE = "newer_record_same_source"
    SETTLED_OVER_ESTIMATE = "settled_event_over_estimate"
    SAFER_INTERPRETATION = "financially_safer_interpretation"
    AMOUNT_UNAVAILABLE = "amount_requires_prior_evidence_resolution"


ReasonCode = ResolutionReason


class ProvenanceKind(StrEnum):
    EVENT = "financial_event"
    MESSAGE = "message"
    IMAGE = "image"
    EXCHANGE_RATE = "exchange_rate"
    LINK = "lifecycle_link"
    RULE = "resolution_rule"


@dataclass(frozen=True, slots=True)
class ResolutionProvenance:
    kind: ProvenanceKind
    source_id: str
    action: str
    observed_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class ResolvedEvent:
    """Resolved row; only projected_cash_effect may change the snapshot balance."""

    source_event: FinancialEvent
    effective_status: EventStatus
    effective_amount: Decimal | None
    effective_currency: Currency
    cash_flow_date: date | None
    amount_home_currency: Decimal | None
    signed_cash_effect: Decimal
    projected_cash_effect: Decimal
    disposition: ResolutionDisposition
    supports_recurrence: bool
    reason_code: ResolutionReason
    provenance: tuple[ResolutionProvenance, ...]

    @property
    def event_id(self) -> str:
        return self.source_event.event_id

    @property
    def user_id(self) -> str:
        return self.source_event.user_id

    @property
    def affects_projected_balance(self) -> bool:
        return self.disposition is ResolutionDisposition.PROJECTED_CASH_FLOW

    @property
    def source_event_ids(self) -> tuple[str, ...]:
        linked = self.source_event.linked_event_id
        return (linked, self.event_id) if linked else (self.event_id,)

    @property
    def source_message_ids(self) -> tuple[str, ...]:
        return tuple(
            p.source_id for p in self.provenance if p.kind is ProvenanceKind.MESSAGE
        )


MessageEvidence: TypeAlias = ExtractedMessageEvidence | MessageEvidenceItem


class ImageResolutionLike(Protocol):
    @property
    def event_id(self) -> str:
        raise NotImplementedError

    @property
    def image_id(self) -> str:
        raise NotImplementedError

    @property
    def linked_event_amount(self) -> Decimal:
        raise NotImplementedError

    @property
    def currency(self) -> Currency:
        raise NotImplementedError

    @property
    def relevant_date(self) -> date | None:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class _Fact:
    item: MessageEvidenceItem
    at: datetime
    source: str


@dataclass(frozen=True, slots=True)
class _Meta:
    newer: bool = False
    settled_over_estimate: bool = False
    safer: bool = False
    cancellation: bool = False
    settlement: bool = False
    amendment: bool = False


_AMENDMENTS = frozenset(
    {
        "salary_amount_amendment",
        "salary_date_amendment",
        "temporary_salary",
        "recurring_expense_amendment",
    }
)
_AMOUNT_AMENDMENTS = frozenset(
    {
        "salary_amount_amendment",
        "temporary_salary",
        "recurring_expense_amendment",
    }
)
_DATE_AMENDMENTS = frozenset({"salary_date_amendment"})
_SETTLEMENTS = frozenset({"event_settled", "refund_settled", "investment_sale_settled"})
_STATUS_FACTS = _SETTLEMENTS | {"event_cancelled"}
_PROVEN_DUPLICATE = ("duplicate representation", "duplicate record", "duplicate entry")
_POSSIBLE_DUPLICATE = (
    "possible duplicate",
    "potential duplicate",
    "suspected duplicate",
)


def resolve_event_lifecycles(
    events: Iterable[FinancialEvent],
    *,
    snapshot_date: date,
    home_currency: Currency,
    exchange_rates: Iterable[ExchangeRate] = (),
    message_evidence: Iterable[MessageEvidence] = (),
    image_resolutions: Iterable[ImageResolutionLike] = (),
    proven_duplicate_event_ids: Iterable[str] = (),
    allow_unresolved_amounts: bool = False,
) -> tuple[ResolvedEvent, ...]:
    """Apply explicit/newer/settled/safer precedence and return typed rows."""

    rows, rates = tuple(events), tuple(exchange_rates)
    by_id = _validate(rows)
    facts = _facts(message_evidence, by_id, snapshot_date)
    images = _images(image_resolutions, by_id)
    proven = frozenset(proven_duplicate_event_ids)
    if proven - by_id.keys():
        raise LifecycleResolutionError("unknown proven duplicate event id")
    output: list[ResolvedEvent] = []
    for row in rows:
        prov = [
            ResolutionProvenance(
                ProvenanceKind.EVENT, row.event_id, f"status={row.status.value}"
            )
        ]
        if row.linked_event_id:
            prov.append(
                ResolutionProvenance(
                    ProvenanceKind.LINK,
                    row.linked_event_id,
                    "context only; not duplication proof",
                )
            )
        amount, currency, when = row.amount, row.currency, row.settlement_date
        image = images.get(row.event_id)
        if image:
            amount, currency = image.linked_event_amount, image.currency
            when = when or image.relevant_date
            prov.append(
                ResolutionProvenance(
                    ProvenanceKind.IMAGE, image.image_id, "resolved blank amount"
                )
            )
        status, meta, extra = _status(row, facts.get(row.event_id, ()))
        prov.extend(extra)
        amount, currency, when, amend_meta, extra = _amend(
            row,
            status,
            amount,
            currency,
            when,
            facts.get(row.event_id, ()),
            home_currency,
            rates,
        )
        prov.extend(extra)
        meta = _merge(meta, amend_meta)
        output.append(
            _materialize(
                row,
                status,
                amount,
                currency,
                when,
                snapshot_date,
                home_currency,
                rates,
                facts.get(row.event_id, ()),
                meta,
                prov,
                allow_unresolved_amounts,
            )
        )
    output = _transfers(output, facts, by_id)
    return tuple(_duplicates(output, by_id, proven))


resolve_lifecycles = resolve_event_lifecycles
resolve_financial_events = resolve_event_lifecycles


def summarize_reason_codes(events: Iterable[ResolvedEvent]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in events:
        counts[row.reason_code.value] = counts.get(row.reason_code.value, 0) + 1
    return dict(sorted(counts.items()))


def _validate(rows: Sequence[FinancialEvent]) -> dict[str, FinancialEvent]:
    if len({r.user_id for r in rows}) > 1:
        raise LifecycleResolutionError("one lifecycle call may contain only one user")
    by_id: dict[str, FinancialEvent] = {}
    for row in rows:
        if row.event_id in by_id:
            raise LifecycleResolutionError(f"duplicate event_id: {row.event_id}")
        if row.amount is not None and (row.amount < 0 or not row.amount.is_finite()):
            raise LifecycleResolutionError(f"event {row.event_id}: invalid amount")
        by_id[row.event_id] = row
    for row in rows:
        if row.linked_event_id and row.linked_event_id not in by_id:
            raise LifecycleResolutionError(
                f"event {row.event_id}: unknown lifecycle link"
            )
        if row.linked_event_id and by_id[row.linked_event_id].user_id != row.user_id:
            raise LifecycleResolutionError(
                f"event {row.event_id}: cross-user lifecycle link"
            )
    return by_id


def _facts(
    evidence: Iterable[MessageEvidence],
    by_id: Mapping[str, FinancialEvent],
    cutoff: date,
) -> dict[str, tuple[_Fact, ...]]:
    grouped: dict[str, list[_Fact]] = {}
    for value in evidence:
        if isinstance(value, ExtractedMessageEvidence):
            item, at, source = value.item, value.sent_at, value.source_type.value
            at = (
                at.replace(tzinfo=timezone.utc)
                if at.tzinfo is None
                else at.astimezone(timezone.utc)
            )
            if at.date() > cutoff:
                continue
        elif isinstance(value, MessageEvidenceItem):
            item, at, source = (
                value,
                datetime.min.replace(tzinfo=timezone.utc),
                "unspecified",
            )
        else:
            raise LifecycleResolutionError("unsupported message evidence type")
        if item.related_event_id is None:
            continue
        if item.related_event_id not in by_id:
            raise LifecycleResolutionError(f"message {item.message_id}: unknown event")
        grouped.setdefault(item.related_event_id, []).append(_Fact(item, at, source))
    return {
        key: tuple(sorted(values, key=lambda f: (f.at, f.item.message_id)))
        for key, values in grouped.items()
    }


def _images(
    values: Iterable[ImageResolutionLike], by_id: Mapping[str, FinancialEvent]
) -> dict[str, ImageResolutionLike]:
    result: dict[str, ImageResolutionLike] = {}
    for value in values:
        event_id = value.event_id
        if event_id not in by_id or event_id in result:
            raise LifecycleResolutionError("invalid or duplicate image resolution")
        result[event_id] = value
    return result


def _newest(
    facts: Sequence[_Fact], kinds: frozenset[str]
) -> tuple[tuple[_Fact, ...], bool]:
    groups: dict[str, list[_Fact]] = {}
    for fact in facts:
        if fact.item.fact_type in kinds:
            groups.setdefault(fact.source, []).append(fact)
    selected: list[_Fact] = []
    changed = False
    for group in groups.values():
        newest = max(f.at for f in group)
        selected.extend(f for f in group if f.at == newest)
        changed |= any(f.at < newest for f in group)
    return tuple(sorted(selected, key=lambda f: (f.at, f.item.message_id))), changed


def _status(
    row: FinancialEvent, facts: Sequence[_Fact]
) -> tuple[EventStatus, _Meta, tuple[ResolutionProvenance, ...]]:
    chosen, newer = _newest(facts, _STATUS_FACTS)
    candidates: list[tuple[EventStatus, _Fact | None]] = []
    if row.status in {EventStatus.CANCELLED, EventStatus.SETTLED}:
        candidates.append((row.status, None))
    candidates += [
        (
            EventStatus.CANCELLED
            if f.item.fact_type == "event_cancelled"
            else EventStatus.SETTLED,
            f,
        )
        for f in chosen
    ]
    estimate = any(
        f.item.fact_type in {"refund_pending", "unconfirmed_income"} for f in facts
    )
    if not candidates:
        return row.status, _Meta(newer=newer), ()
    statuses = {s for s, _ in candidates}
    safer = len(statuses) > 1
    status = (
        (
            EventStatus.SETTLED
            if row.direction is Direction.DEBIT
            else EventStatus.CANCELLED
        )
        if safer
        else candidates[0][0]
    )
    prov = tuple(
        ResolutionProvenance(
            ProvenanceKind.MESSAGE,
            f.item.message_id,
            f"explicit status={s.value}",
            f.at,
        )
        for s, f in candidates
        if f
    )
    used_message = any(f for _, f in candidates)
    return (
        status,
        _Meta(
            newer=newer,
            settled_over_estimate=row.status is EventStatus.SETTLED and estimate,
            safer=safer,
            cancellation=status is EventStatus.CANCELLED and used_message,
            settlement=status is EventStatus.SETTLED and used_message,
        ),
        prov,
    )


def _amend(
    row: FinancialEvent,
    status: EventStatus,
    amount: Decimal | None,
    currency: Currency,
    when: date | None,
    facts: Sequence[_Fact],
    home: Currency,
    rates: Sequence[ExchangeRate],
) -> tuple[
    Decimal | None, Currency, date | None, _Meta, tuple[ResolutionProvenance, ...]
]:
    amount_facts, newer_amount = _newest(facts, _AMOUNT_AMENDMENTS)
    date_facts, newer_date = _newest(facts, _DATE_AMENDMENTS)
    selected = tuple(
        sorted({*amount_facts, *date_facts}, key=lambda f: (f.at, f.item.message_id))
    )
    newer = newer_amount or newer_date
    selected = tuple(
        f
        for f in selected
        if (
            (
                f.item.fact_type == "recurring_expense_amendment"
                and row.direction is Direction.DEBIT
            )
            or (
                f.item.fact_type != "recurring_expense_amendment"
                and row.event_type is EventType.INCOME
                and row.direction is Direction.CREDIT
            )
        )
        and not (
            f.item.fact_type != "salary_date_amendment"
            and f.item.effective_date
            and row.settlement_date
            and f.item.effective_date > row.settlement_date
        )
    )
    if not selected:
        return amount, currency, when, _Meta(newer=newer), ()
    if row.status is EventStatus.SETTLED:
        return amount, currency, when, _Meta(settled_over_estimate=True), ()
    dated: list[tuple[date, _Fact]] = []
    for fact in selected:
        candidate_date = fact.item.settlement_date or (
            fact.item.effective_date
            if fact.item.fact_type == "salary_date_amendment"
            else None
        )
        if candidate_date is not None:
            dated.append((candidate_date, fact))
    safer = len({d for d, _ in dated}) > 1
    if dated:
        when = (min if row.direction is Direction.DEBIT else max)(d for d, _ in dated)
    amounts = []
    for fact in selected:
        if fact.item.amount is None:
            continue
        if when is None:
            raise LifecycleResolutionError(
                f"event {row.event_id}: amendment needs settlement_date"
            )
        value, unit = (
            Decimal(fact.item.amount),
            Currency(fact.item.currency or currency.value),
        )
        try:
            converted = convert_to_home_currency(
                value, unit, home, when, rates, event_id=row.event_id
            )
        except MoneyError as exc:
            raise LifecycleResolutionError(str(exc)) from exc
        amounts.append((converted, value, unit, fact))
    if amounts:
        safer |= len({(v, c) for _, v, c, _ in amounts}) > 1
        picked = (max if row.direction is Direction.DEBIT else min)(
            amounts, key=lambda x: (x[0], x[3].item.message_id)
        )
        _, amount, currency, _ = picked
    changed = bool(dated or amounts)
    prov = tuple(
        ResolutionProvenance(
            ProvenanceKind.MESSAGE,
            f.item.message_id,
            f"considered {f.item.fact_type}",
            f.at,
        )
        for f in selected
    )
    return (
        amount,
        currency,
        when,
        _Meta(newer=newer, safer=safer, amendment=changed),
        prov,
    )


def _merge(a: _Meta, b: _Meta) -> _Meta:
    return _Meta(
        *(
            getattr(a, field) or getattr(b, field)
            for field in _Meta.__dataclass_fields__
        )
    )


def _materialize(
    row: FinancialEvent,
    status: EventStatus,
    amount: Decimal | None,
    currency: Currency,
    when: date | None,
    snapshot: date,
    home: Currency,
    rates: Sequence[ExchangeRate],
    facts: Sequence[_Fact],
    meta: _Meta,
    provenance: Sequence[ResolutionProvenance],
    allow_missing: bool,
) -> ResolvedEvent:
    cash = forecast = recurrence = False
    if (
        status is EventStatus.UNREALIZED
        or row.event_type is EventType.INVESTMENT_VALUATION
    ):
        base = ResolutionReason.UNREALIZED
    elif row.direction is Direction.NON_CASH:
        base = ResolutionReason.NON_CASH
    elif status is EventStatus.CANCELLED:
        base = ResolutionReason.CANCELLED
    elif status is EventStatus.FAILED:
        base = ResolutionReason.FAILED
    elif status is EventStatus.PENDING:
        base = (
            ResolutionReason.PENDING_CREDIT
            if row.direction is Direction.CREDIT
            else ResolutionReason.PENDING_DEBIT
        )
        cash = forecast = row.direction is Direction.DEBIT
    elif status is EventStatus.SCHEDULED:
        confirmed = (
            row.event_type is EventType.INCOME
            and row.direction is Direction.CREDIT
            and not any(f.item.fact_type == "unconfirmed_income" for f in facts)
            and (
                any(f.item.fact_type == "first_salary_confirmed" for f in facts)
                or (
                    "salary" in f"{row.category} {row.description}".casefold()
                    and "confirmed" in row.description.casefold()
                )
            )
        )
        base = (
            ResolutionReason.SCHEDULED_DEBIT
            if row.direction is Direction.DEBIT
            else ResolutionReason.CONFIRMED_SALARY
            if confirmed
            else ResolutionReason.UNCONFIRMED_CREDIT
        )
        cash = forecast = row.direction is Direction.DEBIT or confirmed
    else:
        cash = True
        historical = when is not None and when <= snapshot
        forecast = not historical
        recurrence = historical and row.event_type not in {
            EventType.REFUND,
            EventType.INVESTMENT_PURCHASE,
            EventType.INVESTMENT_SALE,
            EventType.INVESTMENT_VALUATION,
        }
        base = (
            ResolutionReason.SETTLED_INVESTMENT_SALE
            if row.event_type is EventType.INVESTMENT_SALE
            else ResolutionReason.HISTORICAL_SETTLED
            if historical
            else ResolutionReason.FUTURE_SETTLED
        )
    if cash and when is None:
        raise LifecycleResolutionError(
            f"event {row.event_id}: cash event requires settlement_date"
        )
    if cash and amount is None:
        if not allow_missing:
            raise LifecycleResolutionError(
                f"event {row.event_id}: amount requires prior evidence resolution"
            )
        return ResolvedEvent(
            row,
            status,
            None,
            currency,
            when,
            None,
            Decimal(0),
            Decimal(0),
            ResolutionDisposition.REQUIRES_AMOUNT,
            False,
            ResolutionReason.AMOUNT_UNAVAILABLE,
            tuple(provenance),
        )
    home_amount = signed = projected = Decimal(0)
    if cash:
        assert amount is not None and when is not None
        try:
            home_amount = convert_to_home_currency(
                amount, currency, home, when, rates, event_id=row.event_id
            )
        except MoneyError as exc:
            raise LifecycleResolutionError(str(exc)) from exc
        signed = home_amount if row.direction is Direction.CREDIT else -home_amount
        projected = signed if forecast else Decimal(0)
        if currency is not home:
            provenance = (
                *provenance,
                ResolutionProvenance(
                    ProvenanceKind.EXCHANGE_RATE,
                    f"{when}:{currency.value}->{home.value}",
                    "fixed dated conversion",
                ),
            )
    reason = (
        ResolutionReason.SAFER_INTERPRETATION
        if meta.safer
        else ResolutionReason.SETTLED_OVER_ESTIMATE
        if meta.settled_over_estimate
        else ResolutionReason.NEWER_SAME_SOURCE
        if meta.newer
        else ResolutionReason.EXPLICIT_CANCELLATION
        if meta.cancellation
        else ResolutionReason.EXPLICIT_SETTLEMENT
        if meta.settlement
        else ResolutionReason.EXPLICIT_AMENDMENT
        if meta.amendment
        else base
    )
    disposition = (
        ResolutionDisposition.PROJECTED_CASH_FLOW
        if forecast
        else ResolutionDisposition.HISTORICAL_CASH_EVIDENCE
        if cash and status is EventStatus.SETTLED
        else ResolutionDisposition.EXCLUDED
    )
    return ResolvedEvent(
        row,
        status,
        amount,
        currency,
        when,
        home_amount if cash else None,
        signed,
        projected,
        disposition,
        recurrence,
        reason,
        tuple(provenance),
    )


def _component(root: str, by_id: Mapping[str, FinancialEvent]) -> set[str]:
    graph: dict[str, set[str]] = {key: set() for key in by_id}
    for row in by_id.values():
        if row.linked_event_id:
            graph[row.event_id].add(row.linked_event_id)
            graph[row.linked_event_id].add(row.event_id)
    seen, pending = set(), [root]
    while pending:
        item = pending.pop()
        if item not in seen:
            seen.add(item)
            pending.extend(graph[item] - seen)
    return seen


def _transfers(
    rows: Sequence[ResolvedEvent],
    facts: Mapping[str, Sequence[_Fact]],
    by_id: Mapping[str, FinancialEvent],
) -> list[ResolvedEvent]:
    roots = {
        event_id
        for event_id, values in facts.items()
        if any(
            f.item.fact_type == "internal_transfer"
            and f.item.confidence == "high"
            and f.source in {SourceType.BANK.value, SourceType.FINANCIAL_SERVICE.value}
            for f in values
        )
    }
    ids = set().union(*(_component(root, by_id) for root in roots)) if roots else set()
    return [
        replace(
            row,
            signed_cash_effect=Decimal(0),
            projected_cash_effect=Decimal(0),
            disposition=ResolutionDisposition.EXCLUDED,
            supports_recurrence=False,
            reason_code=ResolutionReason.INTERNAL_TRANSFER,
            provenance=(
                *row.provenance,
                ResolutionProvenance(
                    ProvenanceKind.RULE,
                    row.event_id,
                    "trusted own-account transfer netted to zero",
                ),
            ),
        )
        if row.event_id in ids
        else row
        for row in rows
    ]


def _same(a: FinancialEvent, b: FinancialEvent) -> bool:
    return (a.direction, a.amount, a.currency, a.event_type) == (
        b.direction,
        b.amount,
        b.currency,
        b.event_type,
    )


def _duplicates(
    rows: Sequence[ResolvedEvent],
    by_id: Mapping[str, FinancialEvent],
    proven: frozenset[str],
) -> list[ResolvedEvent]:
    output = []
    for item in rows:
        row, text = item.source_event, item.source_event.description.casefold()
        linked = by_id.get(row.linked_event_id or "")
        comparable = linked is not None and _same(row, linked)
        is_proven = row.event_id in proven or (
            comparable and any(x in text for x in _PROVEN_DUPLICATE)
        )
        possible = comparable and any(x in text for x in _POSSIBLE_DUPLICATE)
        if is_proven:
            item = replace(
                item,
                signed_cash_effect=Decimal(0),
                projected_cash_effect=Decimal(0),
                disposition=ResolutionDisposition.EXCLUDED,
                supports_recurrence=False,
                reason_code=ResolutionReason.PROVEN_DUPLICATE,
                provenance=(
                    *item.provenance,
                    ResolutionProvenance(
                        ProvenanceKind.RULE,
                        row.event_id,
                        "proven duplicate suppressed once",
                    ),
                ),
            )
        elif possible:
            item = replace(
                item,
                projected_cash_effect=item.projected_cash_effect
                if row.direction is Direction.DEBIT
                else Decimal(0),
                disposition=item.disposition
                if row.direction is Direction.DEBIT
                else ResolutionDisposition.EXCLUDED,
                supports_recurrence=False,
                reason_code=ResolutionReason.UNRESOLVED_DUPLICATE,
                provenance=(
                    *item.provenance,
                    ResolutionProvenance(
                        ProvenanceKind.RULE,
                        row.event_id,
                        "possible duplicate unresolved; safer interpretation",
                    ),
                ),
            )
        output.append(item)
    return output


__all__ = [
    "LifecycleResolutionError",
    "ProvenanceKind",
    "ReasonCode",
    "ResolutionDisposition",
    "ResolutionProvenance",
    "ResolutionReason",
    "ResolvedEvent",
    "resolve_event_lifecycles",
    "resolve_financial_events",
    "resolve_lifecycles",
    "summarize_reason_codes",
]
