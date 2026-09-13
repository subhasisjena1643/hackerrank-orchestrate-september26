"""Strict parsing helpers with actionable input diagnostics."""

from __future__ import annotations

from datetime import date, datetime, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import NoReturn, TypeVar

from buy_or_wait.money import MoneyError, decimal_from_string


_MISSING_VALUES = frozenset({"", "na", "n/a", "null", "none"})


def _optional_value(value: str) -> str:
    normalized = value.strip()
    return "" if normalized.casefold() in _MISSING_VALUES else normalized


class InputValidationError(ValueError):
    """A dataset contract violation with file, row, and column context."""

    def __init__(
        self,
        path: Path,
        message: str,
        *,
        row_number: int | None = None,
        column: str | None = None,
        value: str | None = None,
    ) -> None:
        self.path = path
        self.row_number = row_number
        self.column = column
        self.value = value
        location = str(path)
        if row_number is not None:
            location += f":{row_number}"
        if column is not None:
            location += f" [{column}]"
        rendered = f"{location}: {message}"
        if value is not None:
            rendered += f" (received {value!r})"
        super().__init__(rendered)


def fail(
    path: Path,
    message: str,
    *,
    row_number: int | None = None,
    column: str | None = None,
    value: str | None = None,
) -> NoReturn:
    raise InputValidationError(
        path, message, row_number=row_number, column=column, value=value
    )


def validate_headers(path: Path, actual: list[str], expected: tuple[str, ...]) -> None:
    if tuple(actual) == expected:
        return
    missing = [column for column in expected if column not in actual]
    unexpected = [column for column in actual if column not in expected]
    details: list[str] = []
    if missing:
        details.append(f"missing columns: {', '.join(missing)}")
    if unexpected:
        details.append(f"unexpected columns: {', '.join(unexpected)}")
    if not missing and not unexpected:
        details.append("columns are not in the required order")
    fail(
        path,
        f"invalid CSV header ({'; '.join(details)}); expected: {', '.join(expected)}",
    )


def required_text(path: Path, row_number: int, column: str, value: str) -> str:
    normalized = value.strip()
    if not normalized:
        fail(
            path,
            "value is required and cannot be blank",
            row_number=row_number,
            column=column,
        )
    return normalized


def optional_text(value: str) -> str | None:
    normalized = _optional_value(value)
    return normalized or None


def parse_decimal(
    path: Path,
    row_number: int,
    column: str,
    value: str,
    *,
    optional: bool = False,
    minimum: Decimal | None = None,
    strictly_positive: bool = False,
) -> Decimal | None:
    normalized = _optional_value(value) if optional else value.strip()
    if not normalized:
        if optional:
            return None
        fail(path, "decimal value is required", row_number=row_number, column=column)
    try:
        parsed = decimal_from_string(normalized)
    except MoneyError as exc:
        fail(path, str(exc), row_number=row_number, column=column, value=value)
    if strictly_positive and parsed <= 0:
        fail(
            path,
            "must be greater than zero",
            row_number=row_number,
            column=column,
            value=value,
        )
    if minimum is not None and parsed < minimum:
        fail(
            path,
            f"must be at least {minimum}",
            row_number=row_number,
            column=column,
            value=value,
        )
    return parsed


def parse_int(
    path: Path,
    row_number: int,
    column: str,
    value: str,
    *,
    optional: bool = False,
    minimum: int | None = None,
) -> int | None:
    normalized = _optional_value(value) if optional else value.strip()
    if not normalized:
        if optional:
            return None
        fail(path, "integer value is required", row_number=row_number, column=column)
    try:
        parsed = int(normalized)
    except ValueError:
        fail(
            path,
            "must be an integer",
            row_number=row_number,
            column=column,
            value=value,
        )
    if str(parsed) != normalized:
        fail(
            path,
            "must use canonical integer syntax",
            row_number=row_number,
            column=column,
            value=value,
        )
    if minimum is not None and parsed < minimum:
        fail(
            path,
            f"must be at least {minimum}",
            row_number=row_number,
            column=column,
            value=value,
        )
    return parsed


def parse_date(
    path: Path, row_number: int, column: str, value: str, *, optional: bool = False
) -> date | None:
    normalized = _optional_value(value) if optional else value.strip()
    if not normalized:
        if optional:
            return None
        fail(path, "date is required", row_number=row_number, column=column)
    try:
        parsed = date.fromisoformat(normalized)
    except ValueError:
        fail(
            path,
            "must be a valid date in YYYY-MM-DD format",
            row_number=row_number,
            column=column,
            value=value,
        )
    if parsed.isoformat() != normalized:
        fail(
            path,
            "must use YYYY-MM-DD format",
            row_number=row_number,
            column=column,
            value=value,
        )
    return parsed


def parse_datetime(path: Path, row_number: int, column: str, value: str) -> datetime:
    normalized = required_text(path, row_number, column, value)
    try:
        parsed = datetime.fromisoformat(normalized.replace("Z", "+00:00"))
    except ValueError:
        fail(
            path,
            "must be a valid ISO-8601 timestamp",
            row_number=row_number,
            column=column,
            value=value,
        )
    if parsed.tzinfo is None:
        fail(
            path,
            "timestamp must include a timezone",
            row_number=row_number,
            column=column,
            value=value,
        )
    return parsed.astimezone(timezone.utc)


def parse_bool(path: Path, row_number: int, column: str, value: str) -> bool:
    normalized = value.strip()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    fail(
        path,
        "must be exactly 'true' or 'false'",
        row_number=row_number,
        column=column,
        value=value,
    )


EnumT = TypeVar("EnumT", bound=Enum)


def parse_enum(
    path: Path, row_number: int, column: str, value: str, enum_type: type[EnumT]
) -> EnumT:
    normalized = value.strip()
    try:
        return enum_type(normalized)
    except ValueError:
        allowed = ", ".join(repr(member.value) for member in enum_type)
        fail(
            path,
            f"must be one of: {allowed}",
            row_number=row_number,
            column=column,
            value=value,
        )


def parse_pipe_list(value: str) -> tuple[str, ...]:
    if not _optional_value(value):
        return ()
    return tuple(part.strip() for part in value.split("|") if part.strip())


def ensure_unique(
    path: Path,
    identifier_name: str,
    identifier: str,
    seen: dict[str, int],
    row_number: int,
) -> None:
    previous = seen.get(identifier)
    if previous is not None:
        fail(
            path,
            f"duplicate {identifier_name}; first seen on row {previous}",
            row_number=row_number,
            column=identifier_name,
            value=identifier,
        )
    seen[identifier] = row_number


def require_foreign_key(
    path: Path,
    row_number: int | None,
    column: str,
    value: str | None,
    valid_values: set[str],
    target: str,
) -> None:
    if value is not None and value not in valid_values:
        fail(
            path,
            f"broken foreign key: no matching {target}",
            row_number=row_number,
            column=column,
            value=value,
        )
