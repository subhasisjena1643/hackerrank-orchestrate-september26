"""Deterministic OCR/vision normalization and consensus for image evidence."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re
from typing import Literal

from buy_or_wait.domain import Currency, FinancialEvent

from .schemas import ImageEvidenceResponse


Confidence = Literal["high", "medium", "low"]
OCRStatus = Literal["success", "no_confident_candidate", "unavailable", "error"]
VisionStatus = Literal["success", "repaired", "invalid", "error"]
ResolutionStatus = Literal[
    "consensus", "adjudication_required", "adjudicated", "unresolved"
]


@dataclass(frozen=True, slots=True)
class OCRCandidate:
    amount: Decimal
    currency: Currency | None
    evidence_label: str
    confidence: Confidence

    def __post_init__(self) -> None:
        if self.amount <= 0:
            raise ValueError("OCR candidate amount must be positive")

    def as_prompt_data(self) -> dict[str, str | None]:
        return {
            "amount": decimal_text(self.amount),
            "currency": None if self.currency is None else self.currency.value,
            "evidence_label": self.evidence_label,
            "confidence": self.confidence,
        }


@dataclass(frozen=True, slots=True)
class OCRChannelResult:
    status: OCRStatus
    version: str
    candidates: tuple[OCRCandidate, ...] = ()
    instruction_text_detected: bool = False
    reason: str = ""


@dataclass(frozen=True, slots=True)
class VisionChannelResult:
    status: VisionStatus
    response: ImageEvidenceResponse | None
    reason: str = ""


@dataclass(frozen=True, slots=True)
class ConsensusResult:
    status: ResolutionStatus
    amount: Decimal | None
    currency: Currency | None
    confidence: Confidence
    evidence_label: str | None
    instruction_text_detected: bool
    reason: str


_STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "bill",
        "for",
        "invoice",
        "of",
        "order",
        "outstanding",
        "payment",
        "purchase",
        "the",
        "transaction",
    }
)
_CATEGORY_LABELS: dict[str, tuple[str, ...]] = {
    "salary": ("salary", "net pay", "take home", "gaji"),
    "rent": ("rent", "balance due", "amount due", "payable"),
    "groceries": (
        "grocery",
        "groceries",
        "pantry",
        "amount paid",
        "net payable",
        "grand total",
    ),
    "utilities": ("bill due", "amount due", "total due", "payable", "water", "telecom"),
    "dining": ("restaurant", "amount paid", "net payable", "grand total"),
    "housing": ("maintenance", "amount due", "net payable", "grand total"),
    "healthcare": ("hospital", "pharmacy", "amount due", "net payable", "payable"),
    "transport": ("fare", "ticket", "charging", "amount paid", "total paid"),
    "shopping": ("order total", "amount paid", "net payable", "grand total", "tote"),
}
_BAD_LABELS = re.compile(
    r"\b(?:account|reference|subtotal|tax|gst|vat|invoice no|order no)\b", re.I
)


def decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def normalize_decimal(value: str | Decimal) -> Decimal:
    try:
        result = value if isinstance(value, Decimal) else Decimal(value)
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("invalid decimal amount") from exc
    if not result.is_finite() or result <= 0:
        raise ValueError("amount must be positive and finite")
    return result.normalize()


def normalize_currency(value: str | Currency | None) -> Currency | None:
    if value is None:
        return None
    if isinstance(value, Currency):
        return value
    aliases = {
        "$": "USD",
        "US$": "USD",
        "€": "EUR",
        "₹": "INR",
        "RS": "INR",
        "RP": "IDR",
        "R": "ZAR",
    }
    token = aliases.get(value.strip().upper(), value.strip().upper())
    try:
        return Currency(token)
    except ValueError as exc:
        raise ValueError(f"unsupported currency: {value}") from exc


def label_supports_event(label: str | None, event: FinancialEvent) -> bool:
    """Require semantic support, not magnitude or visual prominence."""

    if not label or not label.strip():
        return False
    folded = " ".join(label.casefold().split())
    description_words = {
        word
        for word in re.findall(r"[a-z]+", event.description.casefold())
        if len(word) >= 3 and word not in _STOP_WORDS
    }
    if description_words & set(re.findall(r"[a-z]+", folded)):
        return True
    if any(
        term in folded for term in _CATEGORY_LABELS.get(event.category.casefold(), ())
    ):
        return True
    if _BAD_LABELS.search(folded):
        return False
    return any(
        term in folded
        for term in (
            "amount due",
            "amount paid",
            "net amount",
            "net payable",
            "total paid",
            "balance due",
        )
    )


def reach_consensus(
    ocr: OCRChannelResult,
    vision: VisionChannelResult,
    event: FinancialEvent,
) -> ConsensusResult:
    instruction = ocr.instruction_text_detected or bool(
        vision.response and vision.response.instruction_text_detected
    )
    if vision.status not in {"success", "repaired"} or vision.response is None:
        return _needs_adjudication(instruction, f"vision_{vision.status}")
    response = vision.response
    if response.amount is None or response.currency is None:
        return _needs_adjudication(instruction, "vision_has_no_complete_amount")
    try:
        vision_amount = normalize_decimal(response.amount)
        vision_currency = normalize_currency(response.currency)
    except ValueError:
        return _needs_adjudication(instruction, "vision_amount_or_currency_invalid")
    if vision_currency != event.currency:
        return _needs_adjudication(instruction, "vision_currency_conflicts_with_event")
    if not label_supports_event(response.evidence_label, event):
        return _needs_adjudication(instruction, "vision_label_does_not_support_event")
    confident = tuple(
        candidate
        for candidate in ocr.candidates
        if candidate.confidence in {"high", "medium"}
        and label_supports_event(candidate.evidence_label, event)
    )
    unique = {
        (normalize_decimal(candidate.amount), candidate.currency or event.currency)
        for candidate in confident
    }
    if ocr.status != "success" or len(unique) != 1:
        return _needs_adjudication(
            instruction, "ocr_has_no_single_defensible_labelled_candidate"
        )
    ocr_amount, ocr_currency = next(iter(unique))
    if (ocr_amount, ocr_currency) != (vision_amount, vision_currency):
        return _needs_adjudication(instruction, "normalized_ocr_and_vision_disagree")
    label = response.evidence_label or confident[0].evidence_label
    confidence: Confidence = "high" if response.confidence == "high" else "medium"
    return ConsensusResult(
        "consensus",
        vision_amount,
        vision_currency,
        confidence,
        label,
        instruction,
        "normalized OCR and vision agree on a semantically supported labelled amount",
    )


def _needs_adjudication(instruction: bool, reason: str) -> ConsensusResult:
    return ConsensusResult(
        "adjudication_required", None, None, "low", None, instruction, reason
    )
