"""Strict output serialization, parsing, validation, and atomic writing."""

from __future__ import annotations

import csv
from dataclasses import replace
from datetime import date
import os
from pathlib import Path
import re
import tempfile
from typing import Iterable, Mapping, Sequence

from buy_or_wait.domain import (
    AffordabilityStatus,
    Decision,
    Payment,
    PaymentMethod,
    PaymentOption,
    SpendingChange,
    SpendingChangeType,
)
from buy_or_wait.explanation import ExplanationError, build_decision_explanation
from buy_or_wait.finance.plans import build_installment_schedule
from buy_or_wait.money import MoneyError, decimal_from_string, format_decimal
from buy_or_wait.validation.plan import (
    DecisionValidationContext,
    DecisionValidationError,
    validate_decision_plan,
)


OUTPUT_COLUMNS = (
    "request_id",
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
    "decision_explanation",
)


class OutputValidationError(ValueError):
    """A row or file violates the submission contract."""


def decision_to_output_row(decision: Decision) -> dict[str, str]:
    """Serialize a typed decision to the exact eight-column contract."""
    _validate_explanation(decision.decision_explanation)
    preserve = decision.recommended_payment_method is PaymentMethod.INSTALLMENTS
    payment_plan = (
        "|".join(
            f"{item.payment_date.isoformat()}:"
            f"{format_decimal(item.amount, preserve_precision=preserve)}"
            for item in decision.payment_plan
        )
        if decision.payment_plan
        else "none"
    )
    changes = (
        "|".join(_format_change(item) for item in decision.spending_changes_needed)
        if decision.spending_changes_needed
        else "none"
    )
    return {
        "request_id": decision.request_id,
        "amount_safe_to_pay": format_decimal(decision.amount_safe_to_pay),
        "affordability_status": decision.affordability_status.value,
        "recommended_payment_method": decision.recommended_payment_method.value,
        "payment_plan": payment_plan,
        "earliest_date_for_full_payment": (
            decision.earliest_date_for_full_payment.isoformat()
            if decision.earliest_date_for_full_payment
            else ""
        ),
        "spending_changes_needed": changes,
        "decision_explanation": decision.decision_explanation,
    }


def parse_output_row(row: Mapping[str, str]) -> Decision:
    """Parse one exact output row into immutable domain values."""
    if tuple(row.keys()) != OUTPUT_COLUMNS:
        raise OutputValidationError("output columns are not exact or ordered")
    if any(value is None for value in row.values()):
        raise OutputValidationError("output row has missing or extra fields")
    try:
        decision = Decision(
            row["request_id"],
            decimal_from_string(row["amount_safe_to_pay"]),
            AffordabilityStatus(row["affordability_status"]),
            PaymentMethod(row["recommended_payment_method"]),
            _parse_payments(row["payment_plan"]),
            _parse_optional_date(row["earliest_date_for_full_payment"]),
            _parse_changes(row["spending_changes_needed"]),
            row["decision_explanation"],
        )
    except (KeyError, MoneyError, ValueError) as error:
        raise OutputValidationError(f"invalid output row: {error}") from error
    _validate_explanation(decision.decision_explanation)
    return decision


def validate_output_rows(
    rows: Iterable[Mapping[str, str]],
    contexts: Sequence[DecisionValidationContext],
) -> tuple[Decision, ...]:
    """Validate IDs/order, field contracts, and each plan's independent replay."""
    materialized = tuple(rows)
    if len(materialized) != len(contexts):
        raise OutputValidationError("output row count does not match request count")
    decisions: list[Decision] = []
    seen: set[str] = set()
    for index, (row, context) in enumerate(zip(materialized, contexts, strict=True)):
        try:
            decision = parse_output_row(row)
            if decision.request_id in seen:
                raise OutputValidationError(
                    f"duplicate request_id: {decision.request_id}"
                )
            seen.add(decision.request_id)
            if decision.request_id != context.request.request_id:
                raise OutputValidationError(
                    f"row {index + 2}: request IDs are missing, extra, or out of order"
                )
            checked = validate_decision_plan(decision, context)
            try:
                expected_explanation = build_decision_explanation(
                    decision,
                    request=context.request,
                    profile=context.profile,
                    replay=checked.replay,
                    selected_option=checked.selected_option,
                    verified_changes=checked.verified_changes,
                )
            except ExplanationError as error:
                raise OutputValidationError(
                    f"decision_explanation is not grounded: {error}"
                ) from error
            if decision.decision_explanation != expected_explanation:
                raise OutputValidationError(
                    "decision_explanation does not match the deterministic template"
                )
            _validate_canonical_row(row, decision, checked.selected_option)
        except (DecisionValidationError, OutputValidationError) as error:
            raise OutputValidationError(
                f"request {context.request.request_id}: {error}"
            ) from error
        decisions.append(decision)
    return tuple(decisions)


def validate_output_csv(
    path: Path | str,
    contexts: Sequence[DecisionValidationContext],
) -> tuple[Decision, ...]:
    """Read and strictly validate a complete output CSV."""
    source = Path(path)
    try:
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, strict=True)
            if tuple(reader.fieldnames or ()) != OUTPUT_COLUMNS:
                raise OutputValidationError("output columns are not exact or ordered")
            rows = tuple(reader)
    except (OSError, csv.Error) as error:
        raise OutputValidationError(f"cannot read output CSV: {error}") from error
    if any(None in row for row in rows):
        raise OutputValidationError("output row contains an accidental extra comma")
    return validate_output_rows(rows, contexts)


def write_output_csv_atomic(
    decisions: Iterable[Decision],
    contexts: Sequence[DecisionValidationContext],
    output_path: Path | str,
) -> Path:
    """Validate, write a sibling temporary file, revalidate, then replace."""
    target = Path(output_path)
    rows = tuple(decision_to_output_row(value) for value in decisions)
    validate_output_rows(rows, contexts)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(
                handle,
                fieldnames=OUTPUT_COLUMNS,
                lineterminator="\n",
                extrasaction="raise",
            )
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        validate_output_csv(temporary, contexts)
        os.replace(temporary, target)
        temporary = None
        return target
    except (OSError, csv.Error) as error:
        raise OutputValidationError(f"atomic output write failed: {error}") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _parse_payments(value: str) -> tuple[Payment, ...]:
    if value == "none":
        return ()
    if not value or "," in value:
        raise OutputValidationError("invalid payment_plan delimiter")
    result: list[Payment] = []
    for entry in value.split("|"):
        parts = entry.split(":")
        if len(parts) != 2:
            raise OutputValidationError("payment_plan entry must be date:amount")
        result.append(Payment(_parse_date(parts[0]), decimal_from_string(parts[1])))
    return tuple(result)


def _parse_changes(value: str) -> tuple[SpendingChange, ...]:
    if value == "none":
        return ()
    if not value or "," in value:
        raise OutputValidationError("invalid spending_changes_needed delimiter")
    result: list[SpendingChange] = []
    for entry in value.split("|"):
        parts = entry.split(":")
        if len(parts) == 2 and parts[0] == SpendingChangeType.STOP.value:
            result.append(SpendingChange(SpendingChangeType.STOP, parts[1]))
        elif len(parts) == 3 and parts[0] == SpendingChangeType.REDUCE_TO.value:
            result.append(
                SpendingChange(
                    SpendingChangeType.REDUCE_TO,
                    parts[1],
                    decimal_from_string(parts[2]),
                )
            )
        else:
            raise OutputValidationError("invalid spending change syntax")
    return tuple(result)


def _parse_optional_date(value: str) -> date | None:
    return None if value == "" else _parse_date(value)


def _parse_date(value: str) -> date:
    try:
        result = date.fromisoformat(value)
    except ValueError as error:
        raise OutputValidationError(f"invalid ISO date: {value}") from error
    if result.isoformat() != value:
        raise OutputValidationError(f"date is not canonical: {value}")
    return result


def _format_change(change: SpendingChange) -> str:
    if change.change_type is SpendingChangeType.STOP:
        if change.new_amount is not None:
            raise OutputValidationError("stop change cannot contain an amount")
        return f"stop:{change.event_id}"
    if (
        change.change_type is SpendingChangeType.REDUCE_TO
        and change.new_amount is not None
    ):
        return f"reduce_to:{change.event_id}:{format_decimal(change.new_amount)}"
    raise OutputValidationError("reduce_to change requires an amount")


def _validate_explanation(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise OutputValidationError("decision_explanation must not be blank")
    lowered = value.strip().casefold()
    if re.search(r"(?<![a-z])(?:nan|[-+]?infinity)(?![a-z])", lowered):
        raise OutputValidationError("decision_explanation contains a non-finite value")
    if any(token in value for token in ("\r", "\n", "\x00", "|")):
        raise OutputValidationError("decision_explanation contains an unsafe delimiter")
    if lowered.startswith(("{", "[")):
        raise OutputValidationError("decision_explanation must not be JSON")
    if any(
        token in value for token in ("```", "**", "__", "![", "](")
    ) or value.startswith(("# ", "- ", "* ", "> ")):
        raise OutputValidationError("decision_explanation must not contain Markdown")


def _validate_canonical_row(
    row: Mapping[str, str],
    decision: Decision,
    selected_option: PaymentOption | None,
) -> None:
    canonical_decision = decision
    if selected_option is not None:
        canonical_decision = replace(
            decision,
            payment_plan=build_installment_schedule(selected_option),
        )
    canonical = decision_to_output_row(canonical_decision)
    for field in OUTPUT_COLUMNS:
        if row[field] != canonical[field]:
            raise OutputValidationError(f"{field} is not canonically formatted")


serialize_decision = decision_to_output_row
write_output_atomic = write_output_csv_atomic
validate_output_file = validate_output_csv

__all__ = [
    "OUTPUT_COLUMNS",
    "OutputValidationError",
    "decision_to_output_row",
    "parse_output_row",
    "serialize_decision",
    "validate_output_csv",
    "validate_output_file",
    "validate_output_rows",
    "write_output_atomic",
    "write_output_csv_atomic",
]
