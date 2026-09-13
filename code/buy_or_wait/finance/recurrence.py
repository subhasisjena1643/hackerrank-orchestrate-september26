"""Conservative recurrence inference and projection (Phase 8)."""

from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import StrEnum
import re
from statistics import median
from typing import Iterable, Sequence, TypeAlias

from buy_or_wait.domain import Currency, Direction, EventStatus, EventType, Flexibility
from buy_or_wait.evidence.messages import ExtractedMessageEvidence
from buy_or_wait.evidence.schemas import MessageEvidenceItem
from buy_or_wait.finance.lifecycle import (
    ResolvedEvent,
    ResolutionDisposition,
    ResolutionReason,
)
from buy_or_wait.finance.recurrence_policies import (
    Aggregation,
    AmountRounding,
    CadenceAnchor,
    ForecastPolicy,
    estimate_amount,
    require_selected_policy,
    select_history,
)


class Cadence(StrEnum):
    WEEKLY = "weekly"
    BIWEEKLY = "biweekly"
    THREE_WEEK = "three_week"
    MONTHLY = "calendar_monthly"
    THREE_WEEKLY = THREE_WEEK
    CALENDAR_MONTHLY = MONTHLY

    @property
    def nominal_days(self) -> int:
        return {
            self.WEEKLY: 7,
            self.BIWEEKLY: 14,
            self.THREE_WEEK: 21,
            self.MONTHLY: 30,
        }[self]


@dataclass(frozen=True, slots=True)
class SeriesKey:
    user_id: str
    direction: Direction
    category: str
    event_type: EventType
    currency: Currency
    flexibility: Flexibility
    description_family: str


@dataclass(frozen=True, slots=True)
class RecurrenceObservation:
    event_id: str
    when: date
    amount: Decimal
    description: str
    provenance_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RecurrenceAmendment:
    message_id: str
    fact_type: str
    effective_date: date
    amount: Decimal | None
    settlement_date: date | None
    recurrence_scope: str | None
    source_type: str = "unspecified"
    observed_at: datetime = datetime.min.replace(tzinfo=timezone.utc)


@dataclass(frozen=True, slots=True)
class RecurringSeries:
    key: SeriesKey
    cadence: Cadence
    observations: tuple[RecurrenceObservation, ...]
    projected_amount: Decimal
    latest_date: date
    anchor_day: int | None
    month_end: bool
    supported_interval_days: int
    variable_amount: bool
    amendments: tuple[RecurrenceAmendment, ...] = ()

    @property
    def source_event_ids(self) -> tuple[str, ...]:
        return tuple(item.event_id for item in self.observations)


@dataclass(frozen=True, slots=True)
class InferredOccurrence:
    occurrence_id: str
    flow_date: date
    amount: Decimal
    series_key: SeriesKey
    source_event_ids: tuple[str, ...]
    source_message_ids: tuple[str, ...]
    reason_code: str = "inferred_supported_recurrence"


MessageEvidence: TypeAlias = ExtractedMessageEvidence | MessageEvidenceItem

_VARIABLE_CATEGORY_FAMILIES = frozenset({"groceries", "transport", "dining"})
_CATEGORY_FAMILIES = frozenset(
    {
        "rent",
        "utilities",
        "salary",
        "debt_repayment",
        "insurance",
        "family_support",
        "cloud_storage",
        "delivery_membership",
        "gym",
        "music_subscription",
        "streaming",
    }
)
_ONE_OFF_CATEGORIES = frozenset({"investment", "windfall"})
_ONE_OFF_WORDS = re.compile(
    r"\b(?:one[- ]?off|once|prorat(?:ed|a)|bonus|commission|arrears?|refund|"
    r"lottery|prize|sale proceeds|valuation|investment|transfer between|authorization|"
    r"reversal|duplicate|airline|flight|bulk|large|annual|setup)\b",
    re.I,
)
_NOISE = re.compile(
    r"\b(?:payment|purchase|transaction|transfer|monthly|weekly|regular|confirmed|plan|service)\b",
    re.I,
)
_NON_ALNUM = re.compile(r"[^a-z]+")
_EXCLUDED_REASONS = frozenset(
    {
        ResolutionReason.INTERNAL_TRANSFER,
        ResolutionReason.PROVEN_DUPLICATE,
        ResolutionReason.UNRESOLVED_DUPLICATE,
        ResolutionReason.CANCELLED,
        ResolutionReason.FAILED,
        ResolutionReason.UNREALIZED,
    }
)


def semantic_description_family(description: str, category: str) -> str:
    """Return a stable family without collapsing unrelated one-off categories."""
    category_key = category.strip().casefold()
    if category_key in _VARIABLE_CATEGORY_FAMILIES | _CATEGORY_FAMILIES:
        return f"category:{category_key}"
    clean = _NON_ALNUM.sub(" ", _NOISE.sub(" ", description.casefold()))
    words = tuple(word for word in clean.split() if len(word) > 2)
    return "description:" + " ".join(words[:5])


def infer_recurrence_series(
    resolved_events: Iterable[ResolvedEvent],
    *,
    request_date: date,
    policy: ForecastPolicy | None,
    message_evidence: Iterable[MessageEvidence] = (),
) -> tuple[RecurringSeries, ...]:
    """Infer only repeatedly supported, settled, pre-request series."""
    selected_policy = require_selected_policy(policy)
    evidence = tuple(message_evidence)
    facts_by_event = _facts_by_event(evidence)
    global_facts = tuple(
        item for item in evidence if _item(item).related_event_id is None
    )
    grouped: dict[SeriesKey, list[RecurrenceObservation]] = {}
    all_rows: dict[SeriesKey, list[ResolvedEvent]] = {}
    for row in resolved_events:
        event = row.source_event
        if not _eligible(row, request_date, facts_by_event.get(event.event_id, ())):
            continue
        key = SeriesKey(
            event.user_id,
            event.direction,
            event.category,
            event.event_type,
            event.currency,
            event.flexibility,
            semantic_description_family(event.description, event.category),
        )
        if row.amount_home_currency is None or row.cash_flow_date is None:
            continue
        provenance = tuple(
            dict.fromkeys(
                (*row.source_event_ids, *(p.source_id for p in row.provenance))
            )
        )
        grouped.setdefault(key, []).append(
            RecurrenceObservation(
                event.event_id,
                row.cash_flow_date,
                row.amount_home_currency,
                event.description,
                provenance,
            )
        )
        all_rows.setdefault(key, []).append(row)

    result: list[RecurringSeries] = []
    for key, raw in grouped.items():
        observations = _dedupe_dates(raw)
        cadence_info = detect_cadence(tuple(item.when for item in observations))
        if cadence_info is None:
            continue
        cadence, interval, anchor_day, month_end = cadence_info
        amendments = _amendments(
            (
                *global_facts,
                *(
                    fact
                    for row in all_rows[key]
                    for fact in facts_by_event.get(row.event_id, ())
                ),
            ),
            observations[-1].when,
            key,
        )
        if _series_ended(amendments, request_date):
            continue
        variable = key.category in _VARIABLE_CATEGORY_FAMILIES or (
            key.category not in _CATEGORY_FAMILIES
            and len({x.amount for x in observations}) > 1
        )
        amount_observations = (
            _aggregate(observations, cadence, interval, selected_policy)
            if variable
            else observations
        )
        history = select_history(amount_observations, selected_policy)
        if not history:
            continue
        amount = (
            estimate_amount((item.amount for item in history), selected_policy)
            if variable
            else _round(observations[-1].amount, selected_policy)
        )
        result.append(
            RecurringSeries(
                key,
                cadence,
                observations,
                amount,
                observations[-1].when,
                anchor_day,
                month_end,
                interval,
                variable,
                amendments,
            )
        )
    return tuple(
        sorted(
            result,
            key=lambda item: (
                item.key.user_id,
                item.key.direction.value,
                item.key.category,
                item.key.event_type.value,
                item.key.description_family,
            ),
        )
    )


infer_recurrences = infer_recurrence_series
infer_recurring_series = infer_recurrence_series


def detect_cadence(
    dates: Sequence[date],
) -> tuple[Cadence, int, int | None, bool] | None:
    """Recognize supported cadences with at least two agreeing gaps."""
    ordered = tuple(sorted(set(dates)))
    if len(ordered) < 3:
        return None
    month_pairs = [
        (a, b)
        for a, b in zip(ordered, ordered[1:])
        if (b.year * 12 + b.month) - (a.year * 12 + a.month) == 1
        and ((_is_month_end(a) and _is_month_end(b)) or abs(a.day - b.day) <= 2)
    ]
    if len(month_pairs) >= 2 and len(month_pairs) / (len(ordered) - 1) >= 0.6:
        days = [(b - a).days for a, b in month_pairs]
        is_end = sum(_is_month_end(item) for item in ordered) >= max(
            2, len(ordered) - 1
        )
        anchor = (
            monthrange(ordered[-1].year, ordered[-1].month)[1]
            if is_end
            else round(median(item.day for item in ordered))
        )
        return Cadence.MONTHLY, round(median(days)), anchor, is_end
    deltas = [(b - a).days for a, b in zip(ordered, ordered[1:])]
    candidates: list[tuple[int, int, Cadence]] = []
    for cadence, nominal in (
        (Cadence.WEEKLY, 7),
        (Cadence.BIWEEKLY, 14),
        (Cadence.THREE_WEEK, 21),
    ):
        support = [delta for delta in deltas if abs(delta - nominal) <= 1]
        if len(support) >= 2 and len(support) / len(deltas) >= 0.6:
            candidates.append(
                (len(support), -abs(round(median(support)) - nominal), cadence)
            )
    if not candidates:
        return None
    _, _, cadence = max(candidates)
    supported = [delta for delta in deltas if abs(delta - cadence.nominal_days) <= 1]
    return cadence, round(median(supported)), None, False


def project_recurrence_series(
    series: RecurringSeries,
    *,
    start_date: date,
    end_date: date,
    policy: ForecastPolicy | None,
) -> tuple[InferredOccurrence, ...]:
    selected_policy = require_selected_policy(policy)
    if end_date < start_date:
        return ()
    output: list[InferredOccurrence] = []
    one_cycle_used: set[str] = set()
    for ordinal, original_date in enumerate(
        _future_dates(series, end_date, selected_policy), 1
    ):
        when = _amended_date(series, original_date, one_cycle_used)
        if when < start_date or when > end_date or _ended_on(series.amendments, when):
            continue
        amount, message_ids = _amended_amount(
            series, when, one_cycle_used, selected_policy
        )
        output.append(
            InferredOccurrence(
                f"inferred:{series.observations[-1].event_id}:{ordinal}",
                when,
                amount,
                series.key,
                series.source_event_ids,
                message_ids,
            )
        )
    return tuple(sorted(output, key=lambda item: (item.flow_date, item.occurrence_id)))


def project_recurrences(
    series: Iterable[RecurringSeries],
    *,
    start_date: date,
    end_date: date,
    policy: ForecastPolicy | None,
) -> tuple[InferredOccurrence, ...]:
    require_selected_policy(policy)
    return tuple(
        sorted(
            (
                item
                for value in series
                for item in project_recurrence_series(
                    value,
                    start_date=start_date,
                    end_date=end_date,
                    policy=policy,
                )
            ),
            key=lambda item: (item.flow_date, item.occurrence_id),
        )
    )


project_occurrences = project_recurrences


def _eligible(
    row: ResolvedEvent, request_date: date, facts: Sequence[MessageEvidence]
) -> bool:
    event, text = (
        row.source_event,
        f"{row.source_event.description} {row.source_event.category}",
    )
    if (
        row.disposition is not ResolutionDisposition.HISTORICAL_CASH_EVIDENCE
        or row.effective_status is not EventStatus.SETTLED
        or not row.supports_recurrence
        or row.cash_flow_date is None
        or row.cash_flow_date >= request_date
        or row.reason_code in _EXCLUDED_REASONS
        or event.event_type
        in {
            EventType.REFUND,
            EventType.INVESTMENT_PURCHASE,
            EventType.INVESTMENT_SALE,
            EventType.INVESTMENT_VALUATION,
        }
        or event.category.casefold() in _ONE_OFF_CATEGORIES
        or _ONE_OFF_WORDS.search(text)
    ):
        return False
    items = tuple(_item(fact) for fact in facts)
    if any(item.fact_type == "unconfirmed_income" for item in items):
        return False
    first = [item for item in items if item.fact_type == "first_salary_confirmed"]
    return not first or any(item.recurrence_scope == "recurring" for item in first)


def _dedupe_dates(
    values: Sequence[RecurrenceObservation],
) -> tuple[RecurrenceObservation, ...]:
    # Lifecycle resolution already removes proven duplicates. Distinct same-day
    # variable purchases remain observations and may be summed by period policy.
    return tuple(sorted(values, key=lambda item: (item.when, item.event_id)))


def _facts_by_event(
    values: Sequence[MessageEvidence],
) -> dict[str, tuple[MessageEvidence, ...]]:
    grouped: dict[str, list[MessageEvidence]] = {}
    for value in values:
        event_id = _item(value).related_event_id
        if event_id:
            grouped.setdefault(event_id, []).append(value)
    return {key: tuple(items) for key, items in grouped.items()}


def _item(value: MessageEvidence) -> MessageEvidenceItem:
    return value.item if isinstance(value, ExtractedMessageEvidence) else value


def _amendments(
    values: Sequence[MessageEvidence], fallback: date, key: SeriesKey
) -> tuple[RecurrenceAmendment, ...]:
    result = []
    for value in values:
        item = _item(value)
        if (
            item.fact_type
            not in {
                "salary_amount_amendment",
                "salary_date_amendment",
                "temporary_salary",
                "recurring_expense_amendment",
            }
            and item.recurrence_scope != "ended"
        ):
            continue
        salary_fact = (
            item.fact_type
            in {
                "salary_amount_amendment",
                "salary_date_amendment",
                "temporary_salary",
            }
            or item.recurrence_scope == "ended"
        )
        if salary_fact and not (
            key.direction is Direction.CREDIT and key.event_type is EventType.INCOME
        ):
            continue
        if (
            item.fact_type == "recurring_expense_amendment"
            and key.direction is not Direction.DEBIT
        ):
            continue
        if (
            item.fact_type == "recurring_expense_amendment"
            and item.related_event_id is None
        ):
            grounding = " ".join(
                part for part in (item.evidence_quote, item.notes) if part
            ).casefold()
            if key.category.casefold() not in grounding:
                continue
        effective = item.effective_date or fallback
        observed_at = (
            value.sent_at.astimezone(timezone.utc)
            if isinstance(value, ExtractedMessageEvidence)
            else datetime.min.replace(tzinfo=timezone.utc)
        )
        source_type = (
            value.source_type.value
            if isinstance(value, ExtractedMessageEvidence)
            else "unspecified"
        )
        result.append(
            RecurrenceAmendment(
                item.message_id,
                item.fact_type,
                effective,
                Decimal(item.amount) if item.amount is not None else None,
                item.settlement_date,
                item.recurrence_scope,
                source_type,
                observed_at,
            )
        )
    return tuple(
        sorted(result, key=lambda item: (item.effective_date, item.message_id))
    )


def _series_ended(values: Sequence[RecurrenceAmendment], request_date: date) -> bool:
    return any(
        item.recurrence_scope == "ended" and item.effective_date <= request_date
        for item in values
    )


def _ended_on(values: Sequence[RecurrenceAmendment], when: date) -> bool:
    return any(
        item.recurrence_scope == "ended" and item.effective_date <= when
        for item in values
    )


def _aggregate(
    values: Sequence[RecurrenceObservation],
    cadence: Cadence,
    interval: int,
    policy: ForecastPolicy,
) -> tuple[RecurrenceObservation, ...]:
    if policy.aggregation is Aggregation.PER_OCCURRENCE:
        return tuple(values)
    groups: dict[object, list[RecurrenceObservation]] = {}
    latest = values[-1].when
    for value in values:
        key: object = (
            (value.when.year, value.when.month)
            if cadence is Cadence.MONTHLY
            else (latest - value.when).days // max(1, interval)
        )
        groups.setdefault(key, []).append(value)
    output = []
    for group in groups.values():
        last = max(group, key=lambda item: (item.when, item.event_id))
        output.append(
            RecurrenceObservation(
                last.event_id,
                last.when,
                sum((item.amount for item in group), Decimal(0)),
                last.description,
                tuple(source for item in group for source in item.provenance_ids),
            )
        )
    return tuple(sorted(output, key=lambda item: (item.when, item.event_id)))


def _round(value: Decimal, policy: ForecastPolicy) -> Decimal:
    return (
        value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        if policy.amount_rounding is AmountRounding.HALF_UP_2
        else value
    )


def _future_dates(
    series: RecurringSeries, end: date, policy: ForecastPolicy
) -> tuple[date, ...]:
    output, current = [], series.latest_date
    while True:
        if (
            series.cadence is Cadence.MONTHLY
            and policy.cadence_anchor is CadenceAnchor.CALENDAR_DAY_OF_MONTH
        ):
            current = _next_month(
                current, series.anchor_day or current.day, series.month_end
            )
        else:
            current += timedelta(days=series.supported_interval_days)
        if current > end:
            break
        output.append(current)
    return tuple(output)


def _amended_date(series: RecurringSeries, when: date, used: set[str]) -> date:
    eligible = tuple(
        amendment
        for amendment in series.amendments
        if amendment.fact_type == "salary_date_amendment"
        and amendment.settlement_date is not None
        and when >= amendment.effective_date
        and not (
            amendment.recurrence_scope == "one_cycle" and amendment.message_id in used
        )
    )
    selected = _newest_per_source(eligible)
    if not selected:
        return when
    pick = (min if series.key.direction is Direction.DEBIT else max)(
        selected,
        key=lambda item: (item.settlement_date, item.message_id),
    )
    assert pick.settlement_date is not None
    if pick.recurrence_scope == "one_cycle":
        used.add(pick.message_id)
        return pick.settlement_date
    return when.replace(
        day=min(pick.settlement_date.day, monthrange(when.year, when.month)[1])
    )


def _amended_amount(
    series: RecurringSeries, when: date, used: set[str], policy: ForecastPolicy
) -> tuple[Decimal, tuple[str, ...]]:
    eligible = tuple(
        amendment
        for amendment in series.amendments
        if amendment.amount is not None
        and amendment.fact_type != "salary_date_amendment"
        and when >= amendment.effective_date
        and not (
            amendment.recurrence_scope == "one_cycle" and amendment.message_id in used
        )
    )
    selected = _newest_per_source(eligible)
    if not selected:
        return series.projected_amount, ()
    pick = (max if series.key.direction is Direction.DEBIT else min)(
        selected,
        key=lambda item: (item.amount, item.message_id),
    )
    assert pick.amount is not None
    if pick.recurrence_scope == "one_cycle":
        used.add(pick.message_id)
    return _round(pick.amount, policy), (pick.message_id,)


def _newest_per_source(
    values: Sequence[RecurrenceAmendment],
) -> tuple[RecurrenceAmendment, ...]:
    """Apply recency only within one declared source, then retain conflicts."""

    grouped: dict[str, list[RecurrenceAmendment]] = {}
    for value in values:
        grouped.setdefault(value.source_type, []).append(value)
    return tuple(
        sorted(
            (
                max(
                    group,
                    key=lambda item: (
                        item.observed_at,
                        item.effective_date,
                        item.message_id,
                    ),
                )
                for group in grouped.values()
            ),
            key=lambda item: (item.source_type, item.message_id),
        )
    )


def _is_month_end(value: date) -> bool:
    return value.day == monthrange(value.year, value.month)[1]


def _next_month(value: date, anchor_day: int, month_end: bool) -> date:
    year, month = value.year + (value.month == 12), value.month % 12 + 1
    last = monthrange(year, month)[1]
    return date(year, month, last if month_end else min(anchor_day, last))


__all__ = [
    "Cadence",
    "InferredOccurrence",
    "RecurrenceAmendment",
    "RecurrenceObservation",
    "RecurringSeries",
    "SeriesKey",
    "detect_cadence",
    "infer_recurrence_series",
    "infer_recurrences",
    "infer_recurring_series",
    "project_occurrences",
    "project_recurrence_series",
    "project_recurrences",
    "semantic_description_family",
]
