"""Exact decimal parsing, formatting, and dated currency conversion."""

from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation, localcontext
import re
from typing import Iterable

from buy_or_wait.dates import add_months
from buy_or_wait.domain import Currency, ExchangeRate, FinancialEvent


_PLAIN_DECIMAL = re.compile(r"^-?(?:[0-9]+)(?:\.[0-9]+)?$")


class MoneyError(ValueError):
    """Raised when money cannot be parsed or converted under the data contract."""


def decimal_from_string(value: str) -> Decimal:
    """Parse plain base-10 notation exactly, without floats or exponent syntax."""

    if not isinstance(value, str):
        raise MoneyError("decimal input must be the original string")
    if "," in value:
        raise MoneyError("commas are not allowed in numeric fields")
    if not _PLAIN_DECIMAL.fullmatch(value):
        raise MoneyError("must use plain base-10 decimal notation")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise MoneyError("must be a valid base-10 decimal") from exc
    if not result.is_finite():
        raise MoneyError("must be a finite decimal")
    return result


def format_decimal(value: Decimal, *, preserve_precision: bool = False) -> str:
    """Format a finite Decimal without exponent notation or surplus zeros."""

    if not isinstance(value, Decimal) or not value.is_finite():
        raise MoneyError("only finite Decimal values can be formatted")
    rendered = format(value, "f")
    if preserve_precision:
        return rendered
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return "0" if Decimal(rendered) == 0 else rendered


def _currency_code(value: Currency | str) -> str:
    return value.value if isinstance(value, Currency) else value


def convert_to_home_currency(
    amount: Decimal,
    event_currency: Currency | str,
    home_currency: Currency | str,
    settlement_date: date,
    exchange_rates: Iterable[ExchangeRate],
    *,
    event_id: str,
) -> Decimal:
    """Convert using only the exact direct pair on the settlement date."""

    if not isinstance(amount, Decimal) or not amount.is_finite() or amount < 0:
        raise MoneyError(
            f"event {event_id}: amount must be a finite non-negative Decimal"
        )
    source = _currency_code(event_currency)
    target = _currency_code(home_currency)
    if source == target:
        return amount

    matches = [
        item
        for item in exchange_rates
        if item.rate_date == settlement_date
        and _currency_code(item.from_currency) == source
        and _currency_code(item.to_currency) == target
    ]
    pair = f"{source}->{target}"
    if len(matches) != 1:
        detail = "missing" if not matches else "ambiguous"
        raise MoneyError(
            f"event {event_id}: {detail} exchange rate for "
            f"{settlement_date.isoformat()} {pair}"
        )
    rate = matches[0].rate
    if not rate.is_finite() or rate <= 0:
        raise MoneyError(
            f"event {event_id}: invalid exchange rate for "
            f"{settlement_date.isoformat()} {pair}"
        )
    precision = max(28, len(amount.as_tuple().digits) + len(rate.as_tuple().digits))
    with localcontext() as context:
        context.prec = precision
        return amount * rate


def convert_event_amount(
    event: FinancialEvent,
    home_currency: Currency | str,
    exchange_rates: Iterable[ExchangeRate],
) -> Decimal:
    """Return an event amount in home currency under the fixed-rate contract."""

    if event.amount is None:
        raise MoneyError(f"event {event.event_id}: amount is unavailable")
    if _currency_code(event.currency) == _currency_code(home_currency):
        return event.amount
    if event.settlement_date is None:
        pair = f"{_currency_code(event.currency)}->{_currency_code(home_currency)}"
        raise MoneyError(
            f"event {event.event_id}: settlement date is required for exchange pair {pair}"
        )
    return convert_to_home_currency(
        event.amount,
        event.currency,
        home_currency,
        event.settlement_date,
        exchange_rates,
        event_id=event.event_id,
    )


__all__ = [
    "MoneyError",
    "add_months",
    "convert_event_amount",
    "convert_to_home_currency",
    "decimal_from_string",
    "format_decimal",
]
