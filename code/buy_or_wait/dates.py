"""Small calendar-aware date helpers."""

from __future__ import annotations

import calendar
from datetime import date


def add_months(value: date, months: int) -> date:
    """Move by calendar months, clamping to the destination month's last day."""

    month_index = value.year * 12 + value.month - 1 + months
    year, zero_based_month = divmod(month_index, 12)
    month = zero_based_month + 1
    if not 1 <= year <= 9999:
        raise ValueError("calendar month movement is outside the supported date range")
    day = min(value.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


__all__ = ["add_months"]
