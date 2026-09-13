"""Strict, versioned model-output contracts for untrusted evidence extraction."""

from __future__ import annotations

from datetime import date
from decimal import Decimal, InvalidOperation
import re
from typing import Annotated, Literal, TypeAlias

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
    model_validator,
)


EVIDENCE_SCHEMA_VERSION = "1.0.0"
MESSAGE_PROMPT_VERSION = "message-evidence-v1"
IMAGE_PROMPT_VERSION = "image-evidence-v1"
IMAGE_ADJUDICATION_PROMPT_VERSION = "image-adjudication-v1"

RecordId = Annotated[
    str,
    StringConstraints(
        strict=True,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$",
    ),
]
FactValue = Annotated[str, StringConstraints(strict=True, min_length=1, max_length=128)]


class EvidenceSchemaError(ValueError):
    """Raised when a provider response is not exactly one supported schema."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class EvidenceFactPayload(_StrictModel):
    """A closed fact shape with strict type/value pairing.

    A single object shape avoids ``oneOf``, which is unsupported by OpenAI strict
    structured outputs in array items. ``typed_value`` supplies typed parsing.
    """

    fact_type: Literal["amount", "currency", "event_date", "settlement_date", "status"]
    value: FactValue
    related_event_id: RecordId | None

    @model_validator(mode="after")
    def validate_typed_value(self) -> "EvidenceFactPayload":
        if self.fact_type == "amount":
            if re.fullmatch(r"(?:0|[1-9][0-9]*)(?:\.[0-9]+)?", self.value) is None:
                raise ValueError("amount must be a non-negative plain decimal string")
        elif self.fact_type == "currency":
            if self.value not in {"EUR", "IDR", "INR", "USD", "ZAR"}:
                raise ValueError("currency is not allowlisted")
        elif self.fact_type in {"event_date", "settlement_date"}:
            try:
                date.fromisoformat(self.value)
            except ValueError as exc:
                raise ValueError("date must be an ISO calendar date") from exc
        elif self.fact_type == "status" and self.value not in {
            "cancelled",
            "failed",
            "pending",
            "scheduled",
            "settled",
            "unrealized",
        }:
            raise ValueError("status is not allowlisted")
        return self

    @property
    def typed_value(self) -> Decimal | date | str:
        if self.fact_type == "amount":
            try:
                return Decimal(self.value)
            except InvalidOperation as exc:  # Defensive after model validation.
                raise EvidenceSchemaError("invalid validated amount") from exc
        if self.fact_type in {"event_date", "settlement_date"}:
            return date.fromisoformat(self.value)
        return self.value


ALLOWED_FACT_TYPES = frozenset(
    {"amount", "currency", "event_date", "settlement_date", "status"}
)


class MessageEvidenceResponse(_StrictModel):
    """Legacy single-source Phase 4 boundary retained for compatibility."""

    schema_version: Literal["1.0.0"]
    source_kind: Literal["message"]
    source_id: RecordId
    facts: tuple[EvidenceFactPayload, ...] = Field(max_length=20)


MessageFactType = Literal[
    "salary_amount_amendment",
    "salary_date_amendment",
    "temporary_salary",
    "first_salary_confirmed",
    "unconfirmed_income",
    "recurring_expense_amendment",
    "event_cancelled",
    "event_settled",
    "internal_transfer",
    "refund_pending",
    "refund_settled",
    "investment_valuation_non_cash",
    "investment_sale_settled",
    "suspicious_instruction",
    "no_relevant_fact",
]


class MessageEvidenceItem(_StrictModel):
    """One item in the exact ``message-evidence-v1`` runtime response."""

    message_id: RecordId
    related_event_id: RecordId | None
    fact_type: MessageFactType
    amount: (
        Annotated[
            str,
            StringConstraints(strict=True, pattern=r"^(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$"),
        ]
        | None
    )
    currency: Literal["INR", "ZAR", "IDR", "USD", "EUR"] | None
    effective_date: date | None
    settlement_date: date | None
    recurrence_scope: Literal["one_cycle", "recurring", "ended", "unknown"] | None
    cash_state: (
        Literal[
            "confirmed_credit",
            "confirmed_debit",
            "pending_credit",
            "pending_debit",
            "non_cash",
            "cancelled",
            "unknown",
        ]
        | None
    )
    confidence: Literal["high", "medium", "low"]
    evidence_quote: (
        Annotated[str, StringConstraints(strict=True, max_length=500)] | None
    )
    notes: Annotated[str, StringConstraints(strict=True, max_length=500)]

    @model_validator(mode="after")
    def validate_fact_contract(self) -> "MessageEvidenceItem":
        if self.evidence_quote is not None and len(self.evidence_quote.split()) > 20:
            raise ValueError("evidence_quote must contain at most 20 source words")
        if self.fact_type == "no_relevant_fact" and any(
            value is not None
            for value in (
                self.amount,
                self.currency,
                self.effective_date,
                self.settlement_date,
                self.recurrence_scope,
                self.cash_state,
                self.evidence_quote,
            )
        ):
            raise ValueError("no_relevant_fact requires null financial fields")
        if self.fact_type == "unconfirmed_income" and self.cash_state in {
            "confirmed_credit",
            "confirmed_debit",
        }:
            raise ValueError("unconfirmed income cannot be confirmed cash")
        if self.fact_type in {
            "salary_amount_amendment",
            "salary_date_amendment",
            "temporary_salary",
            "recurring_expense_amendment",
        } and self.cash_state not in {None, "unknown"}:
            raise ValueError("an amendment cannot itself confirm a cash flow")
        if (
            self.fact_type == "first_salary_confirmed"
            and self.cash_state != "confirmed_credit"
        ):
            raise ValueError("a confirmed first salary must be confirmed_credit")
        if self.fact_type == "internal_transfer" and self.cash_state != "non_cash":
            raise ValueError("an internal transfer must be non_cash")
        if self.fact_type == "refund_pending" and self.cash_state != "pending_credit":
            raise ValueError("a pending refund must be pending_credit")
        if self.fact_type == "refund_settled" and self.cash_state != "confirmed_credit":
            raise ValueError("a settled refund must be confirmed_credit")
        if (
            self.fact_type == "investment_valuation_non_cash"
            and self.cash_state != "non_cash"
        ):
            raise ValueError("an investment valuation must be non_cash")
        if (
            self.fact_type == "investment_sale_settled"
            and self.cash_state != "confirmed_credit"
        ):
            raise ValueError("a settled investment sale must be confirmed_credit")
        if self.fact_type == "event_cancelled" and self.cash_state != "cancelled":
            raise ValueError("a cancelled event must have cancelled cash_state")
        if self.fact_type == "temporary_salary" and self.recurrence_scope not in {
            "one_cycle",
            "recurring",
        }:
            raise ValueError("temporary salary must declare one_cycle or recurring")
        if self.fact_type == "suspicious_instruction" and self.cash_state not in {
            None,
            "unknown",
        }:
            raise ValueError("a suspicious instruction cannot create a cash flow")
        return self


class MessageEvidenceBatchResponse(_StrictModel):
    """Strict batch shape required by Runtime Prompt 1."""

    items: tuple[MessageEvidenceItem, ...] = Field(min_length=1, max_length=12)


class ImageEvidenceResponse(_StrictModel):
    """Exact strict shape required by Runtime Prompt 4."""

    image_id: RecordId
    event_id: RecordId
    document_type: Annotated[
        str, StringConstraints(strict=True, min_length=1, max_length=128)
    ]
    amount: (
        Annotated[
            str,
            StringConstraints(strict=True, pattern=r"^(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$"),
        ]
        | None
    )
    currency: Literal["INR", "ZAR", "IDR", "USD", "EUR"] | None
    document_date: date | None
    confidence: Literal["high", "medium", "low"]
    evidence_label: (
        Annotated[str, StringConstraints(strict=True, max_length=300)] | None
    )
    instruction_text_detected: bool
    notes: Annotated[str, StringConstraints(strict=True, max_length=500)]

    @property
    def decimal_amount(self) -> Decimal | None:
        return None if self.amount is None else Decimal(self.amount)


class ImageAdjudicationResponse(_StrictModel):
    """Exact strict shape required by Runtime Prompt 7."""

    image_id: RecordId
    event_id: RecordId
    amount: (
        Annotated[
            str,
            StringConstraints(strict=True, pattern=r"^(?:0|[1-9][0-9]*)(?:\.[0-9]+)?$"),
        ]
        | None
    )
    currency: Literal["INR", "ZAR", "IDR", "USD", "EUR"] | None
    confidence: Literal["high", "medium", "low"]
    selected_channel: Literal["ocr", "vision", "both", "unresolved"]
    evidence_label: (
        Annotated[str, StringConstraints(strict=True, max_length=300)] | None
    )
    reason: Annotated[str, StringConstraints(strict=True, min_length=1, max_length=500)]
    instruction_text_detected: bool

    @model_validator(mode="after")
    def validate_resolution(self) -> "ImageAdjudicationResponse":
        if self.selected_channel == "unresolved":
            if self.amount is not None or self.confidence != "low":
                raise ValueError(
                    "unresolved adjudication must have null amount and low confidence"
                )
        elif (
            self.amount is None or self.currency is None or self.evidence_label is None
        ):
            raise ValueError(
                "resolved adjudication requires amount, currency, and evidence_label"
            )
        return self

    @property
    def decimal_amount(self) -> Decimal | None:
        return None if self.amount is None else Decimal(self.amount)


EvidenceResponse: TypeAlias = MessageEvidenceResponse | ImageEvidenceResponse


def parse_message_response(raw_json: str | bytes) -> MessageEvidenceResponse:
    return _parse(raw_json, MessageEvidenceResponse, "message")


def parse_message_batch_response(
    raw_json: str | bytes,
) -> MessageEvidenceBatchResponse:
    return _parse(raw_json, MessageEvidenceBatchResponse, "message batch")


def parse_image_response(raw_json: str | bytes) -> ImageEvidenceResponse:
    return _parse(raw_json, ImageEvidenceResponse, "image")


def parse_image_adjudication_response(
    raw_json: str | bytes,
) -> ImageAdjudicationResponse:
    return _parse(raw_json, ImageAdjudicationResponse, "image adjudication")


def parse_evidence_response(
    raw_json: str | bytes, *, source_kind: Literal["message", "image"]
) -> EvidenceResponse:
    if source_kind == "message":
        return parse_message_response(raw_json)
    return parse_image_response(raw_json)


def _parse(raw_json: str | bytes, model: type[BaseModel], label: str):
    if not isinstance(raw_json, (str, bytes)):
        raise EvidenceSchemaError(f"{label} provider response must be JSON text")
    try:
        return model.model_validate_json(raw_json, strict=True)
    except (ValidationError, ValueError, TypeError) as exc:
        raise EvidenceSchemaError(f"invalid {label} evidence response") from exc


MESSAGE_EVIDENCE_JSON_SCHEMA = MessageEvidenceBatchResponse.model_json_schema()
LEGACY_MESSAGE_EVIDENCE_JSON_SCHEMA = MessageEvidenceResponse.model_json_schema()
IMAGE_EVIDENCE_JSON_SCHEMA = ImageEvidenceResponse.model_json_schema()
IMAGE_ADJUDICATION_JSON_SCHEMA = ImageAdjudicationResponse.model_json_schema()
