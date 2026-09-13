"""Predeclared, deterministic Phase 8 forecast policies.

This module contains no sample-output reader. The complete hypothesis space is
constructed at import time and every policy has a stable JSON representation.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from enum import StrEnum
import hashlib
import json
from itertools import product
from typing import Iterable, Sequence, TypeVar, cast


class AmountEstimator(StrEnum):
    RECENT_ARITHMETIC_MEAN = "recent_arithmetic_mean"
    MEDIAN = "median"
    TRIMMED_MEAN_20 = "trimmed_mean_20_percent"
    LATEST = "latest_value"
    PERCENTILE_60 = "percentile_60"
    PERCENTILE_75 = "percentile_75"
    MEAN = RECENT_ARITHMETIC_MEAN
    TRIMMED_MEAN = TRIMMED_MEAN_20
    LATEST_VALUE = LATEST


class HistoryWindow(StrEnum):
    LAST_2 = "last_2_supported_cycles"
    LAST_3 = "last_3_supported_cycles"
    LAST_4 = "last_4_supported_cycles"
    RECENT_90_DAYS = "all_observations_recent_90_days"
    RECENT_90 = RECENT_90_DAYS


class Aggregation(StrEnum):
    PER_OCCURRENCE = "per_occurrence"
    CALENDAR_PERIOD_TOTAL = "calendar_period_total"


class CadenceAnchor(StrEnum):
    MEDIAN_SUPPORTED_INTERVAL = "median_supported_interval"
    CALENDAR_DAY_OF_MONTH = "calendar_day_of_month"


class AmountRounding(StrEnum):
    EXACT_DECIMAL = "exact_decimal"
    HALF_UP_2 = "round_half_up_2"
    ROUND_HALF_UP = HALF_UP_2


class SameDayOrdering(StrEnum):
    DEBITS_BEFORE_CREDITS = "debits_before_credits"
    CREDITS_BEFORE_DEBITS = "credits_before_debits"


SAME_DAY_POLICIES = {
    "conservative_debits_first": SameDayOrdering.DEBITS_BEFORE_CREDITS,
    "credits_first": SameDayOrdering.CREDITS_BEFORE_DEBITS,
}


@dataclass(frozen=True, slots=True)
class ForecastPolicy:
    amount_estimator: AmountEstimator
    history_window: HistoryWindow
    aggregation: Aggregation
    cadence_anchor: CadenceAnchor
    amount_rounding: AmountRounding
    same_day_ordering: SameDayOrdering
    schema_version: str = "forecast-policy-v1"

    def canonical_dict(self) -> dict[str, str]:
        return {
            "aggregation": self.aggregation.value,
            "amount_estimator": self.amount_estimator.value,
            "amount_rounding": self.amount_rounding.value,
            "cadence_anchor": self.cadence_anchor.value,
            "history_window": self.history_window.value,
            "same_day_ordering": self.same_day_ordering.value,
            "schema_version": self.schema_version,
        }

    def serialize_canonical(self) -> str:
        return json.dumps(
            self.canonical_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )

    canonical_json = serialize_canonical

    def to_canonical_json(self) -> str:
        return self.serialize_canonical()

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.serialize_canonical().encode("utf-8")).hexdigest()

    @property
    def policy_hash(self) -> str:
        return self.sha256

    def hash(self) -> str:
        return self.sha256


def _coerce_policy(policy: ForecastPolicy | None) -> ForecastPolicy:
    if policy is None:
        raise ValueError("an explicit selected ForecastPolicy is required")
    if not isinstance(policy, ForecastPolicy):
        raise TypeError("policy must be a ForecastPolicy")
    return policy


T = TypeVar("T")


def select_history(observations: Sequence[T], policy: ForecastPolicy) -> tuple[T, ...]:
    """Select chronological observations by configured cycles or recent days."""

    _coerce_policy(policy)
    values = tuple(observations)
    count = {
        HistoryWindow.LAST_2: 2,
        HistoryWindow.LAST_3: 3,
        HistoryWindow.LAST_4: 4,
    }.get(policy.history_window)
    if count is not None:
        return values[-count:]
    if not values:
        return ()

    def when(value: object):
        return getattr(value, "when", getattr(value, "date", None))

    latest = when(values[-1])
    if latest is None:
        return values
    return tuple(value for value in values if (latest - when(value)).days <= 90)


def estimate_amount(amounts: Iterable[Decimal], policy: ForecastPolicy) -> Decimal:
    """Apply exactly the estimator and rounding declared by ``policy``."""

    _coerce_policy(policy)
    values = tuple(Decimal(value) for value in amounts)
    if not values:
        raise ValueError("cannot estimate an empty amount sequence")
    estimator = policy.amount_estimator
    ordered = tuple(sorted(values))
    if estimator is AmountEstimator.LATEST:
        result = values[-1]
    elif estimator is AmountEstimator.RECENT_ARITHMETIC_MEAN:
        result = sum(values, Decimal(0)) / Decimal(len(values))
    elif estimator is AmountEstimator.MEDIAN:
        middle = len(ordered) // 2
        result = (
            ordered[middle]
            if len(ordered) % 2
            else (ordered[middle - 1] + ordered[middle]) / 2
        )
    elif estimator is AmountEstimator.TRIMMED_MEAN_20:
        trim = int(Decimal(len(ordered)) * Decimal("0.20"))
        retained = ordered[trim : len(ordered) - trim] if trim else ordered
        result = sum(retained, Decimal(0)) / Decimal(len(retained))
    elif estimator in {AmountEstimator.PERCENTILE_60, AmountEstimator.PERCENTILE_75}:
        percentile = (
            Decimal("0.60")
            if estimator is AmountEstimator.PERCENTILE_60
            else Decimal("0.75")
        )
        position = Decimal(len(ordered) - 1) * percentile
        lower = int(position)
        fraction = position - lower
        upper = min(lower + 1, len(ordered) - 1)
        result = ordered[lower] + (ordered[upper] - ordered[lower]) * fraction
    else:  # pragma: no cover - closed enum
        raise ValueError(f"unsupported estimator: {estimator}")
    if policy.amount_rounding is AmountRounding.HALF_UP_2:
        return result.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return result


# Declared before any evaluation code can read solved sample columns.
POLICY_GRID: tuple[ForecastPolicy, ...] = tuple(
    ForecastPolicy(
        amount_estimator=cast(AmountEstimator, values[0]),
        history_window=cast(HistoryWindow, values[1]),
        aggregation=cast(Aggregation, values[2]),
        cadence_anchor=cast(CadenceAnchor, values[3]),
        amount_rounding=cast(AmountRounding, values[4]),
        same_day_ordering=cast(SameDayOrdering, values[5]),
    )
    for values in product(
        tuple(AmountEstimator),
        tuple(HistoryWindow),
        tuple(Aggregation),
        tuple(CadenceAnchor),
        tuple(AmountRounding),
        tuple(SameDayOrdering),
    )
)
FORECAST_POLICY_GRID = POLICY_GRID

BASELINE_POLICY = ForecastPolicy(
    amount_estimator=AmountEstimator.RECENT_ARITHMETIC_MEAN,
    history_window=HistoryWindow.LAST_3,
    aggregation=Aggregation.PER_OCCURRENCE,
    cadence_anchor=CadenceAnchor.CALENDAR_DAY_OF_MONTH,
    amount_rounding=AmountRounding.HALF_UP_2,
    same_day_ordering=SameDayOrdering.DEBITS_BEFORE_CREDITS,
)


def require_selected_policy(policy: ForecastPolicy | None) -> ForecastPolicy:
    """Fail-closed production boundary: no implicit estimator fallback."""

    return _coerce_policy(policy)


__all__ = [
    "Aggregation",
    "AmountEstimator",
    "AmountRounding",
    "BASELINE_POLICY",
    "CadenceAnchor",
    "FORECAST_POLICY_GRID",
    "ForecastPolicy",
    "HistoryWindow",
    "POLICY_GRID",
    "SAME_DAY_POLICIES",
    "SameDayOrdering",
    "estimate_amount",
    "require_selected_policy",
    "select_history",
]
