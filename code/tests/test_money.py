"""Phase 3 exact money, exchange-rate, and calendar tests."""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from pathlib import Path
import sys

import pytest

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from buy_or_wait.dates import add_months
from buy_or_wait.domain import Currency, ExchangeRate
from buy_or_wait.money import (
    MoneyError,
    convert_to_home_currency,
    decimal_from_string,
    format_decimal,
)
from buy_or_wait.validation.inputs import InputValidationError, parse_decimal


def _rate(day: str, source: Currency, target: Currency, value: str) -> ExchangeRate:
    return ExchangeRate(date.fromisoformat(day), source, target, Decimal(value))


@pytest.mark.parametrize("currency", list(Currency))
def test_all_five_home_currencies_are_same_currency_no_ops(currency: Currency) -> None:
    amount = Decimal("123.450")
    assert (
        convert_to_home_currency(
            amount, currency, currency, date(2026, 1, 1), (), event_id="event_same"
        )
        is amount
    )


@pytest.mark.parametrize(
    ("source", "home", "rate", "expected"),
    [
        (Currency.USD, Currency.INR, "83.33", "166.66"),
        (Currency.EUR, Currency.ZAR, "20", "40"),
        (Currency.USD, Currency.IDR, "15833.33", "31666.66"),
        (Currency.USD, Currency.EUR, "0.92", "1.84"),
        (Currency.EUR, Currency.USD, "1.09", "2.18"),
    ],
)
def test_conversion_supports_every_home_currency(
    source: Currency, home: Currency, rate: str, expected: str
) -> None:
    rates = (_rate("2026-09-01", source, home, rate),)
    assert convert_to_home_currency(
        Decimal("2"), source, home, date(2026, 9, 1), rates, event_id="event_all"
    ) == Decimal(expected)


def test_exact_conversion_and_large_idr_value_with_cents() -> None:
    converted = convert_to_home_currency(
        Decimal("98765432101234567890.25"),
        Currency.USD,
        Currency.IDR,
        date(2026, 9, 1),
        (_rate("2026-09-01", Currency.USD, Currency.IDR, "15833.33"),),
        event_id="event_idr",
    )
    assert converted == Decimal("1563785679051440320813732.0325")


def test_rate_is_selected_by_exact_settlement_date() -> None:
    rates = (
        _rate("2026-08-31", Currency.USD, Currency.EUR, "0.91"),
        _rate("2026-09-01", Currency.USD, Currency.EUR, "0.92"),
    )
    assert convert_to_home_currency(
        Decimal("10.25"),
        Currency.USD,
        Currency.EUR,
        date(2026, 9, 1),
        rates,
        event_id="event_date",
    ) == Decimal("9.4300")


def test_missing_rate_does_not_use_nearest_date() -> None:
    rates = (_rate("2026-08-31", Currency.USD, Currency.EUR, "0.91"),)
    with pytest.raises(MoneyError, match=r"event_missing.*2026-09-01.*USD->EUR"):
        convert_to_home_currency(
            Decimal("10"),
            Currency.USD,
            Currency.EUR,
            date(2026, 9, 1),
            rates,
            event_id="event_missing",
        )


def test_wrong_direction_is_not_inverted() -> None:
    rates = (_rate("2026-09-01", Currency.EUR, Currency.USD, "1.09"),)
    with pytest.raises(MoneyError, match=r"event_direction.*USD->EUR"):
        convert_to_home_currency(
            Decimal("10"),
            Currency.USD,
            Currency.EUR,
            date(2026, 9, 1),
            rates,
            event_id="event_direction",
        )


@pytest.mark.parametrize(
    "bad", ["NaN", "Infinity", "-Infinity", "1,000", "1e3", "1E+3"]
)
def test_invalid_decimal_notation_is_rejected(bad: str) -> None:
    with pytest.raises(MoneyError):
        decimal_from_string(bad)


def test_schema_forbidden_negative_decimal_is_rejected() -> None:
    with pytest.raises(InputValidationError, match="must be at least 0"):
        parse_decimal(Path("amounts.csv"), 2, "amount", "-0.01", minimum=Decimal("0"))


def test_decimal_parsing_and_canonical_trailing_zero_formatting() -> None:
    value = decimal_from_string("100.2500")
    assert value == Decimal("100.2500")
    assert format_decimal(value) == "100.25"
    assert format_decimal(value, preserve_precision=True) == "100.2500"
    assert format_decimal(Decimal("1E+20")) == "100000000000000000000"
    assert format_decimal(Decimal("-0.00")) == "0"


@pytest.mark.parametrize(
    ("start", "months", "expected"),
    [
        (date(2025, 1, 31), 1, date(2025, 2, 28)),
        (date(2024, 1, 31), 1, date(2024, 2, 29)),
        (date(2025, 3, 31), -1, date(2025, 2, 28)),
        (date(2025, 12, 31), 2, date(2026, 2, 28)),
    ],
)
def test_end_of_month_date_movement(start: date, months: int, expected: date) -> None:
    assert add_months(start, months) == expected
