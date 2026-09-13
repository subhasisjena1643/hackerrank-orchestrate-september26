"""Fail-closed validation for model-derived, untrusted evidence."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
import json
import re
from typing import Iterable, Mapping, Literal

from .schemas import (
    EvidenceResponse,
    ImageEvidenceResponse,
    MessageEvidenceBatchResponse,
    MessageEvidenceItem,
    parse_evidence_response,
    parse_message_batch_response,
)


class EvidenceGuardrailError(ValueError):
    """Raised when structured evidence is unsupported, ungrounded, or unsafe."""


@dataclass(frozen=True, slots=True)
class ValidatedEvidence:
    """Marker proving schema, source IDs, and literal grounding were checked."""

    response: EvidenceResponse

    @property
    def source_id(self) -> str:
        if isinstance(self.response, ImageEvidenceResponse):
            return self.response.image_id
        return self.response.source_id

    @property
    def facts(self):
        if isinstance(self.response, ImageEvidenceResponse):
            return ()
        return self.response.facts


@dataclass(frozen=True, slots=True)
class MessageGroundingSource:
    message_id: str
    related_event_id: str | None
    message_text: str


@dataclass(frozen=True, slots=True)
class ValidatedMessageBatch:
    """Marker proving every item in a runtime message batch was grounded."""

    response: MessageEvidenceBatchResponse

    @property
    def source_ids(self) -> tuple[str, ...]:
        return tuple(item.message_id for item in self.response.items)


FORBIDDEN_DECISION_FIELDS = frozenset(
    {
        "request_id",
        "amount_safe_to_pay",
        "affordability_status",
        "recommended_payment_method",
        "payment_plan",
        "earliest_date_for_full_payment",
        "spending_changes_needed",
        "decision_explanation",
    }
)
FORBIDDEN_INSTRUCTION_FIELDS = frozenset(
    {
        "instruction",
        "instructions",
        "system",
        "system_prompt",
        "prompt",
        "prompt_override",
        "tool",
        "tool_call",
        "function_call",
        "new_event",
        "new_events",
        "event_id_to_create",
    }
)

_NUMBER_RE = re.compile(
    r"(?<![\d.])[-+]?(?:\d{1,3}(?:[.,]\d{3})+|\d+)(?:[.,]\d+)?(?![\d.])"
)
_ISO_DATE_RE = re.compile(r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)")
_MONTH_DATE_RE = re.compile(
    r"\b(?:"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{1,2},?\s+\d{4}|"
    r"\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{4})\b",
    re.IGNORECASE,
)
_NUMERIC_DATE_RE = re.compile(r"(?<!\d)(\d{1,2})[/-](\d{1,2})[/-](\d{4})(?!\d)")
_INDONESIAN_DATE_RE = re.compile(
    r"(?<!\d)(\d{1,2})\s+"
    r"(Januari|Februari|Maret|April|Mei|Juni|Juli|Agustus|September|"
    r"Oktober|November|Desember)\s+(\d{4})(?!\d)",
    re.IGNORECASE,
)
_FRENCH_DATE_RE = re.compile(
    r"(?<!\d)(\d{1,2})\s+"
    r"(janvier|février|mars|avril|mai|juin|juillet|août|septembre|"
    r"octobre|novembre|décembre)\s+(\d{4})(?!\d)",
    re.IGNORECASE,
)
_FACT_ANCHORS: dict[str, tuple[str, ...]] = {
    "salary_amount_amendment": ("salary", "gaji", "upah", "salaire"),
    "salary_date_amendment": (
        "salary",
        "payday",
        "gaji",
        "tanggal gajian",
        "salaire",
        "paie",
    ),
    "temporary_salary": ("salary", "gaji", "upah", "salaire"),
    "first_salary_confirmed": ("salary", "gaji", "paycheck", "salaire", "paie"),
    "unconfirmed_income": (
        "bonus",
        "commission",
        "komisi",
        "gig",
        "payout",
        "prize",
        "hadiah",
        "lottery",
        "lotere",
        "refund",
        "pengembalian",
        "prime",
        "remboursement",
    ),
    "recurring_expense_amendment": (
        "rent",
        "sewa",
        "expense",
        "biaya",
        "subscription",
        "langganan",
        "loyer",
        "dépense",
        "abonnement",
    ),
    "event_cancelled": (
        "cancelled",
        "canceled",
        "dibatalkan",
        "batal",
        "annulé",
        "annulée",
    ),
    "event_settled": (
        "settled",
        "settlement",
        "completed",
        "credited",
        "selesai",
        "masuk",
        "réglé",
        "crédité",
        "reçu",
    ),
    "internal_transfer": (
        "own account",
        "own accounts",
        "rekening saya",
        "akun saya",
        "milik sendiri",
        "propres comptes",
        "compte à mon nom",
    ),
    "refund_pending": ("refund", "pengembalian", "dana kembali", "remboursement"),
    "refund_settled": ("refund", "pengembalian", "dana kembali", "remboursement"),
    "investment_valuation_non_cash": (
        "investment",
        "investasi",
        "market value",
        "nilai pasar",
        "unrealized",
        "investissement",
        "valeur de marché",
        "non réalisé",
    ),
    "investment_sale_settled": (
        "investment",
        "investasi",
        "sale",
        "sold",
        "penjualan",
        "dijual",
        "investissement",
        "vente",
        "vendu",
    ),
    "suspicious_instruction": (
        "release fee",
        "processing fee",
        "unlock fee",
        "biaya pelepasan",
        "biaya pemrosesan",
        "biaya proses",
        "biaya administrasi",
        "frais de traitement",
        "frais de libération",
        "frais de déblocage",
    ),
}


def validate_message_batch(
    raw_json: str | bytes,
    *,
    sources: Iterable[MessageGroundingSource],
    known_record_ids: Iterable[str] = (),
) -> ValidatedMessageBatch:
    """Strictly parse and ground one result per message, in supplied order."""

    _reject_forbidden_response_keys(raw_json)
    source_list = tuple(sources)
    try:
        response = parse_message_batch_response(raw_json)
    except ValueError as exc:
        raise EvidenceGuardrailError(
            "provider response failed strict message batch schema validation"
        ) from exc
    expected_ids = tuple(source.message_id for source in source_list)
    actual_ids = tuple(item.message_id for item in response.items)
    if actual_ids != expected_ids:
        raise EvidenceGuardrailError(
            "message response IDs/order do not exactly match supplied records"
        )
    known_ids = frozenset(known_record_ids)
    for item, source in zip(response.items, source_list, strict=True):
        _validate_message_item(item, source=source, known_record_ids=known_ids)
    return ValidatedMessageBatch(response)


def _validate_message_item(
    item: MessageEvidenceItem,
    *,
    source: MessageGroundingSource,
    known_record_ids: frozenset[str],
) -> None:
    if item.related_event_id != source.related_event_id:
        raise EvidenceGuardrailError(
            f"related_event_id was changed for {source.message_id}"
        )
    if (
        item.related_event_id is not None
        and item.related_event_id not in known_record_ids
    ):
        raise EvidenceGuardrailError(
            f"invented or unknown record ID: {item.related_event_id}"
        )
    text = source.message_text
    folded = " ".join(text.casefold().split())
    if item.fact_type != "no_relevant_fact":
        anchors = _FACT_ANCHORS[item.fact_type]
        if not any(anchor in folded for anchor in anchors):
            raise EvidenceGuardrailError(
                f"ungrounded fact type from {source.message_id}"
            )
    if item.amount is not None and not _amount_is_grounded(item.amount, text):
        raise EvidenceGuardrailError(f"ungrounded amount from {source.message_id}")
    if item.currency is not None and not _currency_is_grounded(item.currency, text):
        raise EvidenceGuardrailError(f"ungrounded currency from {source.message_id}")
    for field_name, value in (
        ("effective_date", item.effective_date),
        ("settlement_date", item.settlement_date),
    ):
        if value is not None and value not in _dates_in_text(text):
            raise EvidenceGuardrailError(
                f"ungrounded {field_name} from {source.message_id}"
            )
    if item.evidence_quote is not None:
        quote = " ".join(item.evidence_quote.casefold().split())
        if not quote or quote not in folded:
            raise EvidenceGuardrailError(
                f"evidence quote is not a source substring for {source.message_id}"
            )
    if item.fact_type in {
        "event_cancelled",
        "event_settled",
        "refund_settled",
        "investment_sale_settled",
    }:
        status_pattern = {
            "event_cancelled": r"cancelled|canceled|dibatalkan|batal",
            "event_settled": r"settled|completed|credited|selesai|masuk",
            "refund_settled": r"settled|completed|credited|received|selesai|masuk",
            "investment_sale_settled": r"settled|completed|credited|received|selesai|masuk",
        }[item.fact_type]
        if _status_is_negated_or_future(folded, status_pattern):
            raise EvidenceGuardrailError(
                f"negated or future status from {source.message_id}"
            )
    if item.fact_type == "internal_transfer" and re.search(
        r"\b(?:not|bukan|tidak)\b[^.!?]{0,24}\b(?:own|saya|sendiri)\b",
        folded,
    ):
        raise EvidenceGuardrailError(
            f"negated account ownership from {source.message_id}"
        )
    if re.search(
        r"\b(?:recommend(?:ed|ation)?|affordab(?:le|ility)|payment[_ ]plan|"
        r"spending[_ ]changes?|decision[_ ]explanation)\b",
        item.notes,
        re.IGNORECASE,
    ):
        raise EvidenceGuardrailError(
            f"recommendation language is forbidden for {source.message_id}"
        )


def validate_raw_evidence(
    raw_json: str | bytes,
    *,
    source_kind: Literal["message", "image"],
    source_text: str,
    known_source_ids: Iterable[str],
    known_record_ids: Iterable[str] = (),
    image_output_text: str | None = None,
) -> ValidatedEvidence:
    """Parse and validate a provider response without accepting partial output."""

    _reject_forbidden_response_keys(raw_json)
    try:
        response = parse_evidence_response(raw_json, source_kind=source_kind)
    except ValueError as exc:
        raise EvidenceGuardrailError(
            "provider response failed strict schema validation"
        ) from exc
    return validate_evidence(
        response,
        source_text=source_text,
        known_source_ids=known_source_ids,
        known_record_ids=known_record_ids,
        image_output_text=image_output_text,
    )


def validate_evidence(
    response: EvidenceResponse,
    *,
    source_text: str,
    known_source_ids: Iterable[str],
    known_record_ids: Iterable[str] = (),
    image_output_text: str | None = None,
) -> ValidatedEvidence:
    """Accept only allowlisted, source-linked, literally grounded facts.

    Source text is data only. Instruction-like phrases inside it are never parsed
    as commands and do not weaken any validation below.
    """

    source_ids = frozenset(known_source_ids)
    record_ids = frozenset(known_record_ids)
    if isinstance(response, ImageEvidenceResponse):
        if response.image_id not in source_ids:
            raise EvidenceGuardrailError(f"unknown source ID: {response.image_id}")
        if response.event_id not in record_ids:
            raise EvidenceGuardrailError(
                f"invented or unknown record ID: {response.event_id}"
            )
        grounding_text = (
            image_output_text if image_output_text is not None else source_text
        )
        if response.amount is not None and not _amount_is_grounded(
            response.amount, grounding_text
        ):
            raise EvidenceGuardrailError(f"ungrounded amount from {response.image_id}")
        if response.currency is not None and not _currency_is_grounded(
            response.currency, grounding_text
        ):
            raise EvidenceGuardrailError(
                f"ungrounded currency from {response.image_id}"
            )
        if (
            response.document_date is not None
            and response.document_date not in _dates_in_text(grounding_text)
        ):
            raise EvidenceGuardrailError(f"ungrounded date from {response.image_id}")
        if response.evidence_label is not None and " ".join(
            response.evidence_label.casefold().split()
        ) not in " ".join(grounding_text.casefold().split()):
            raise EvidenceGuardrailError(
                f"ungrounded evidence label from {response.image_id}"
            )
        _reject_recommendation_language(response.notes, response.image_id)
        return ValidatedEvidence(response)

    if response.source_id not in source_ids:
        raise EvidenceGuardrailError(f"unknown source ID: {response.source_id}")

    grounding_text = source_text

    for fact in response.facts:
        if (
            fact.related_event_id is not None
            and fact.related_event_id not in record_ids
        ):
            raise EvidenceGuardrailError(
                f"invented or unknown record ID: {fact.related_event_id}"
            )
        if fact.fact_type == "amount":
            if not _amount_is_grounded(fact.value, grounding_text):
                raise EvidenceGuardrailError(
                    f"ungrounded amount from {response.source_id}"
                )
        elif fact.fact_type == "currency":
            if (
                re.search(
                    rf"(?<![A-Za-z]){re.escape(fact.value)}(?![A-Za-z])",
                    grounding_text,
                    re.IGNORECASE,
                )
                is None
            ):
                raise EvidenceGuardrailError(
                    f"ungrounded currency from {response.source_id}"
                )
        elif fact.fact_type in {"event_date", "settlement_date"}:
            if date.fromisoformat(fact.value) not in _dates_in_text(grounding_text):
                raise EvidenceGuardrailError(
                    f"ungrounded date from {response.source_id}"
                )
        elif fact.fact_type == "status":
            if (
                re.search(
                    rf"\b{re.escape(fact.value)}\b",
                    grounding_text,
                    re.IGNORECASE,
                )
                is None
            ):
                raise EvidenceGuardrailError(
                    f"ungrounded status from {response.source_id}"
                )
        else:  # Defensive even if a caller circumvents typed parsing.
            raise EvidenceGuardrailError("unknown fact type")
    return ValidatedEvidence(response)


def _reject_recommendation_language(text: str, source_id: str) -> None:
    if re.search(
        r"\b(?:recommend(?:ed|ation)?|affordab(?:le|ility)|payment[_ ]plan|"
        r"spending[_ ]changes?|decision[_ ]explanation)\b",
        text,
        re.IGNORECASE,
    ):
        raise EvidenceGuardrailError(
            f"recommendation language is forbidden for {source_id}"
        )


def _status_is_negated_or_future(text: str, status_pattern: str) -> bool:
    return (
        re.search(
            rf"\b(?:not|never|belum|tidak|bukan|will|may|might|expected to|scheduled to|pas|jamais|sera|devrait)"
            rf"\b[^.!?]{{0,32}}\b(?:{status_pattern})\b",
            text,
            re.IGNORECASE,
        )
        is not None
    )


def _reject_forbidden_response_keys(raw_json: str | bytes) -> None:
    try:
        value = json.loads(raw_json)
    except (json.JSONDecodeError, TypeError, UnicodeDecodeError) as exc:
        raise EvidenceGuardrailError("provider response is not valid JSON") from exc

    def walk(item: object) -> None:
        if isinstance(item, Mapping):
            keys = {str(key).casefold() for key in item}
            forbidden = keys & (
                FORBIDDEN_DECISION_FIELDS | FORBIDDEN_INSTRUCTION_FIELDS
            )
            if forbidden:
                raise EvidenceGuardrailError(
                    f"forbidden provider field: {sorted(forbidden)[0]}"
                )
            for child in item.values():
                walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)

    walk(value)


def _amount_is_grounded(value: str, text: str) -> bool:
    try:
        expected = Decimal(value)
    except InvalidOperation:
        return False
    for token in _NUMBER_RE.findall(text):
        for candidate in _decimal_candidates(token):
            if candidate == expected:
                return True
    return False


def _decimal_candidates(token: str) -> frozenset[Decimal]:
    """Return defensible decimal/grouping interpretations for a source token."""

    raw = token.strip().lstrip("+")
    variants = {raw, raw.replace(",", ""), raw.replace(".", "")}
    if "," in raw and "." in raw:
        if raw.rfind(",") > raw.rfind("."):
            variants.add(raw.replace(".", "").replace(",", "."))
        else:
            variants.add(raw.replace(",", ""))
    elif "," in raw:
        variants.add(raw.replace(",", "."))
    candidates: set[Decimal] = set()
    for variant in variants:
        try:
            candidates.add(Decimal(variant))
        except InvalidOperation:
            pass
    return frozenset(candidates)


def _currency_is_grounded(currency: str, text: str) -> bool:
    patterns = {
        "USD": r"(?<![A-Za-z])(?:USD|US\$|\$)(?![A-Za-z])",
        "EUR": r"(?<![A-Za-z])(?:EUR|€)(?![A-Za-z])",
        "INR": r"(?<![A-Za-z])(?:INR|₹|Rs\.)(?![A-Za-z])",
        "IDR": r"(?<![A-Za-z])(?:IDR|Rp\.?)\s*\d",
        "ZAR": r"(?<![A-Za-z])(?:ZAR|R\s*\d)(?![A-Za-z])",
    }
    return re.search(patterns[currency], text, re.IGNORECASE) is not None


def _dates_in_text(text: str) -> frozenset[date]:
    dates: set[date] = set()
    for token in _ISO_DATE_RE.findall(text):
        try:
            dates.add(date.fromisoformat(token))
        except ValueError:
            pass
    formats = (
        "%B %d, %Y",
        "%B %d %Y",
        "%b %d, %Y",
        "%b %d %Y",
        "%d %B %Y",
        "%d %b %Y",
    )
    for match in _MONTH_DATE_RE.findall(text):
        for format_string in formats:
            try:
                dates.add(datetime.strptime(match, format_string).date())
                break
            except ValueError:
                continue
    for first, second, year in _NUMERIC_DATE_RE.findall(text):
        for month, day in ((int(second), int(first)), (int(first), int(second))):
            try:
                dates.add(date(int(year), month, day))
            except ValueError:
                pass
    indonesian_months = {
        "januari": 1,
        "februari": 2,
        "maret": 3,
        "april": 4,
        "mei": 5,
        "juni": 6,
        "juli": 7,
        "agustus": 8,
        "september": 9,
        "oktober": 10,
        "november": 11,
        "desember": 12,
    }
    for day, month_name, year in _INDONESIAN_DATE_RE.findall(text):
        try:
            dates.add(
                date(int(year), indonesian_months[month_name.casefold()], int(day))
            )
        except ValueError:
            pass
    french_months = {
        "janvier": 1,
        "février": 2,
        "mars": 3,
        "avril": 4,
        "mai": 5,
        "juin": 6,
        "juillet": 7,
        "août": 8,
        "septembre": 9,
        "octobre": 10,
        "novembre": 11,
        "décembre": 12,
    }
    for day, month_name, year in _FRENCH_DATE_RE.findall(text):
        try:
            dates.add(date(int(year), french_months[month_name.casefold()], int(day)))
        except ValueError:
            pass
    return frozenset(dates)
