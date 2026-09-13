"""Bounded multilingual message-evidence extraction; never mutates cash flow."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Iterable, Literal, Sequence

from buy_or_wait.domain import Currency, MessageRecord, SourceType
from buy_or_wait.telemetry import ModelPrice, TelemetryLedger, build_telemetry_record

from .cache import (
    CacheIdentity,
    EvidenceCache,
    EvidenceCacheError,
    compose_cache_source,
)
from .client import EvidenceClient, ProviderResponse
from .guardrails import (
    EvidenceGuardrailError,
    MessageGroundingSource,
    ValidatedMessageBatch,
    validate_message_batch,
)
from .schemas import (
    EVIDENCE_SCHEMA_VERSION,
    MESSAGE_EVIDENCE_JSON_SCHEMA,
    MESSAGE_PROMPT_VERSION,
    MessageEvidenceBatchResponse,
    MessageEvidenceItem,
    MessageFactType,
)

MAX_BATCH_SIZE = 8
MAX_BATCH_SOURCE_CHARS = 12_000
_PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"
_SYSTEM_PROMPT = (_PROMPT_DIR / "message_evidence.md").read_text(encoding="utf-8")
_USER_TEMPLATE = (_PROMPT_DIR / "message_evidence_user.md").read_text(encoding="utf-8")
_REPAIR_TEMPLATE = (_PROMPT_DIR / "message_evidence_repair.md").read_text(
    encoding="utf-8"
)


@dataclass(frozen=True, slots=True)
class ExtractedMessageEvidence:
    item: MessageEvidenceItem
    sent_at: datetime
    source_type: SourceType
    extraction_method: str
    cache_hit: bool = False


@dataclass(frozen=True, slots=True)
class MessageExtractionFailure:
    message_ids: tuple[str, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class MessageExtractionSummary:
    input_messages: int
    relevant_messages: int
    filtered_messages: int
    deterministic_results: int
    model_results: int
    failures: int
    cache_hits: int
    cache_misses: int
    model_calls: int
    repair_calls: int
    input_tokens: int
    cached_tokens: int
    output_tokens: int


@dataclass(frozen=True, slots=True)
class MessageExtractionRun:
    evidence: tuple[ExtractedMessageEvidence, ...]
    failures: tuple[MessageExtractionFailure, ...]
    summary: MessageExtractionSummary


def filter_relevant_messages(
    messages: Iterable[MessageRecord],
    *,
    user_id: str,
    request_id: str,
    known_event_ids: Iterable[str],
) -> tuple[MessageRecord, ...]:
    """Filter keys before any model call and return chronological evidence."""

    event_ids = frozenset(known_event_ids)
    return tuple(
        sorted(
            (
                message
                for message in messages
                if message.user_id == user_id
                and (
                    message.request_id is None
                    or message.request_id == request_id
                    or message.related_event_id in event_ids
                )
            ),
            key=lambda message: (message.sent_at, message.message_id),
        )
    )


def extract_message_evidence(
    messages: Sequence[MessageRecord],
    *,
    user_id: str,
    request_id: str,
    request_date: date,
    home_currency: Currency,
    known_event_ids: Iterable[str],
    client: EvidenceClient | None = None,
    cache: EvidenceCache | None = None,
    telemetry: TelemetryLedger | None = None,
    run_id: str = "message-extraction",
    price: ModelPrice | None = None,
    batch_size: int = MAX_BATCH_SIZE,
) -> MessageExtractionRun:
    if not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise ValueError(f"batch_size must be between 1 and {MAX_BATCH_SIZE}")
    event_ids = tuple(sorted(frozenset(known_event_ids)))
    relevant = filter_relevant_messages(
        messages, user_id=user_id, request_id=request_id, known_event_ids=event_ids
    )
    evidence: list[ExtractedMessageEvidence] = []
    pending: list[MessageRecord] = []
    failures: list[MessageExtractionFailure] = []
    counters = {
        "deterministic_results": 0,
        "model_results": 0,
        "cache_hits": 0,
        "cache_misses": 0,
        "model_calls": 0,
        "repair_calls": 0,
        "input_tokens": 0,
        "cached_tokens": 0,
        "output_tokens": 0,
    }
    for message in relevant:
        item = _obvious_fact(message)
        if item is None:
            pending.append(message)
        else:
            evidence.append(_wrap(item, message, "deterministic"))
            counters["deterministic_results"] += 1

    if client is None:
        failures.extend(
            MessageExtractionFailure(
                (message.message_id,), "model_required_but_unavailable"
            )
            for message in pending
        )
    else:
        batches, oversized = _bounded_batches(pending, batch_size=batch_size)
        failures.extend(
            MessageExtractionFailure(
                (message.message_id,), "message_exceeds_provider_batch_limit"
            )
            for message in oversized
        )
        for batch in batches:
            _extract_batch(
                batch,
                request_id=request_id,
                request_date=request_date,
                home_currency=home_currency,
                event_ids=event_ids,
                client=client,
                cache=cache,
                telemetry=telemetry,
                run_id=run_id,
                price=price,
                evidence=evidence,
                failures=failures,
                counters=counters,
            )

    evidence.sort(key=lambda result: (result.sent_at, result.item.message_id))
    return MessageExtractionRun(
        tuple(evidence),
        tuple(failures),
        MessageExtractionSummary(
            input_messages=len(messages),
            relevant_messages=len(relevant),
            filtered_messages=len(messages) - len(relevant),
            failures=sum(len(failure.message_ids) for failure in failures),
            **counters,
        ),
    )


def _bounded_batches(
    messages: Sequence[MessageRecord], *, batch_size: int
) -> tuple[tuple[tuple[MessageRecord, ...], ...], tuple[MessageRecord, ...]]:
    batches: list[tuple[MessageRecord, ...]] = []
    current: list[MessageRecord] = []
    current_chars = 0
    oversized: list[MessageRecord] = []
    for message in messages:
        size = len(message.message_text)
        if size > MAX_BATCH_SOURCE_CHARS:
            oversized.append(message)
            continue
        if current and (
            len(current) >= batch_size or current_chars + size > MAX_BATCH_SOURCE_CHARS
        ):
            batches.append(tuple(current))
            current = []
            current_chars = 0
        current.append(message)
        current_chars += size
    if current:
        batches.append(tuple(current))
    return tuple(batches), tuple(oversized)


def _extract_batch(
    batch: tuple[MessageRecord, ...],
    *,
    request_id: str,
    request_date: date,
    home_currency: Currency,
    event_ids: tuple[str, ...],
    client: EvidenceClient,
    cache: EvidenceCache | None,
    telemetry: TelemetryLedger | None,
    run_id: str,
    price: ModelPrice | None,
    evidence: list[ExtractedMessageEvidence],
    failures: list[MessageExtractionFailure],
    counters: dict[str, int],
) -> None:
    user_prompt = _render_user_prompt(
        batch,
        home_currency=home_currency,
        request_id=request_id,
        request_date=request_date,
        event_ids=event_ids,
    )
    batch_hash = hashlib.sha256(user_prompt.encode("utf-8")).hexdigest()[:16]
    source_id = f"message_batch_{batch_hash}"
    identity = CacheIdentity(
        compose_cache_source(
            _SYSTEM_PROMPT,
            user_prompt,
            _REPAIR_TEMPLATE,
            json.dumps(
                MESSAGE_EVIDENCE_JSON_SCHEMA, sort_keys=True, separators=(",", ":")
            ),
        ),
        MESSAGE_PROMPT_VERSION,
        EVIDENCE_SCHEMA_VERSION,
        client.provider,
        client.text_model,
    )
    try:
        cached = cache.get(identity) if cache is not None else None
    except EvidenceCacheError:
        cached = None
    if isinstance(cached, ValidatedMessageBatch):
        try:
            revalidated = validate_message_batch(
                cached.response.model_dump_json(),
                sources=(
                    MessageGroundingSource(
                        message.message_id,
                        message.related_event_id,
                        message.message_text,
                    )
                    for message in batch
                ),
                known_record_ids=event_ids,
            )
        except EvidenceGuardrailError:
            cached = None
        else:
            counters["cache_hits"] += 1
            _append_batch(
                evidence, batch, revalidated.response, "model", cache_hit=True
            )
            counters["model_results"] += len(batch)
            return
    counters["cache_misses"] += 1
    usage = [0, 0, 0]
    retry_count = 0
    try:
        counters["model_calls"] += 1
        first = client.extract_text(
            prompt=_SYSTEM_PROMPT,
            source_id=source_id,
            source_text=user_prompt,
            json_schema=MESSAGE_EVIDENCE_JSON_SCHEMA,
        )
        retry_count += first.retry_count
        counters["model_calls"] += first.retry_count
        _add_usage(first, usage, counters)
        try:
            validated = _validate(first, batch, event_ids)
        except EvidenceGuardrailError as error:
            retry_count = 1
            repair = _render_repair(error)
            repair_fn = getattr(client, "repair_text", None)
            counters["model_calls"] += 1
            counters["repair_calls"] += 1
            if repair_fn is None:
                second = client.extract_text(
                    prompt=_SYSTEM_PROMPT,
                    source_id=source_id,
                    source_text=f"{user_prompt}\n\n{repair}",
                    json_schema=MESSAGE_EVIDENCE_JSON_SCHEMA,
                )
            else:
                second = repair_fn(
                    system_prompt=_SYSTEM_PROMPT,
                    user_prompt=user_prompt,
                    previous_response=first.json_text,
                    repair_prompt=repair,
                    source_id=source_id,
                    json_schema=MESSAGE_EVIDENCE_JSON_SCHEMA,
                )
            _add_usage(second, usage, counters)
            retry_count += second.retry_count
            counters["model_calls"] += second.retry_count
            validated = _validate(second, batch, event_ids)
        if cache is not None:
            cache.put(identity, validated)
        _append_batch(evidence, batch, validated.response, "model")
        counters["model_results"] += len(batch)
        _record_telemetry(
            telemetry,
            run_id,
            request_id,
            source_id,
            client,
            usage,
            retry_count,
            True,
            None,
            price,
        )
    except Exception as error:
        failures.append(
            MessageExtractionFailure(
                tuple(message.message_id for message in batch),
                f"{type(error).__name__}: {str(error)[:200]}",
            )
        )
        _record_telemetry(
            telemetry,
            run_id,
            request_id,
            source_id,
            client,
            usage,
            retry_count,
            False,
            f"{type(error).__name__}: {str(error)[:200]}",
            price,
        )


def _validate(
    response: ProviderResponse,
    batch: tuple[MessageRecord, ...],
    event_ids: tuple[str, ...],
) -> ValidatedMessageBatch:
    return validate_message_batch(
        response.json_text,
        sources=(
            MessageGroundingSource(
                message.message_id, message.related_event_id, message.message_text
            )
            for message in batch
        ),
        known_record_ids=event_ids,
    )


def _add_usage(
    response: ProviderResponse, usage: list[int], counters: dict[str, int]
) -> None:
    values = (response.input_tokens, response.cached_tokens, response.output_tokens)
    for index, key in enumerate(("input_tokens", "cached_tokens", "output_tokens")):
        usage[index] += values[index]
        counters[key] += values[index]


def _record_telemetry(
    ledger: TelemetryLedger | None,
    run_id: str,
    request_id: str,
    source_id: str,
    client: EvidenceClient,
    usage: list[int],
    retry_count: int,
    success: bool,
    failure: str | None,
    price: ModelPrice | None,
) -> None:
    if ledger is None:
        return
    ledger.append(
        build_telemetry_record(
            run_id=run_id,
            request_id=request_id,
            stage="message_evidence",
            source_id=source_id,
            provider=client.provider,
            model=client.text_model,
            prompt_version=MESSAGE_PROMPT_VERSION,
            schema_version=EVIDENCE_SCHEMA_VERSION,
            input_tokens=usage[0],
            cached_tokens=usage[1],
            output_tokens=usage[2],
            retry_count=retry_count,
            success=success,
            failure=failure,
            price=price,
        )
    )


def _append_batch(
    target: list[ExtractedMessageEvidence],
    batch: tuple[MessageRecord, ...],
    response: MessageEvidenceBatchResponse,
    method: str,
    cache_hit: bool = False,
) -> None:
    for item, message in zip(response.items, batch, strict=True):
        target.append(_wrap(item, message, method, cache_hit))


def _wrap(
    item: MessageEvidenceItem,
    message: MessageRecord,
    method: str,
    cache_hit: bool = False,
) -> ExtractedMessageEvidence:
    return ExtractedMessageEvidence(
        item=item,
        sent_at=message.sent_at,
        source_type=message.source_type,
        extraction_method=method,
        cache_hit=cache_hit,
    )


def _obvious_fact(message: MessageRecord) -> MessageEvidenceItem | None:
    text = " ".join(message.message_text.casefold().split())
    fact_type: MessageFactType | None = None
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
    ) = None
    quote: str | None = None
    if _is_suspicious_fee_instruction(text):
        fact_type, cash_state = "suspicious_instruction", "unknown"
        quote = _matching_quote(
            message.message_text,
            r"(?:release|processing|unlock) fee|biaya (?:pelepasan|pemrosesan|proses)",
        )
    elif (
        message.related_event_id
        and not _looks_instruction_like(text)
        and re.search(
            r"\b(?:(?:was|is|has been|now) (?:cancelled|canceled)|"
            r"cancellation confirmed|telah dibatalkan|sudah dibatalkan|dibatalkan)\b",
            text,
        )
        and not _status_is_negated_or_future(
            text, r"cancelled|canceled|dibatalkan|batal"
        )
    ):
        fact_type, cash_state = "event_cancelled", "cancelled"
        quote = _matching_quote(
            message.message_text, r"cancelled|canceled|dibatalkan|batal"
        )
    elif (
        message.related_event_id
        and not re.search(
            r"\b(?:refund|investment|sale|prize|salary|gaji|pengembalian|investasi|hadiah)\b",
            text,
        )
        and not _looks_instruction_like(text)
        and re.search(
            r"\b(?:(?:was|is|has) settled|settlement (?:is )?completed|"
            r"(?:was|has been) credited|telah masuk|sudah masuk)\b",
            text,
        )
        and not _status_is_negated_or_future(
            text, r"settled|settlement completed|credited|telah masuk|sudah masuk"
        )
    ):
        fact_type, cash_state = "event_settled", "unknown"
        quote = _matching_quote(
            message.message_text,
            r"settled|settlement completed|credited|telah masuk|sudah masuk",
        )
    if fact_type is None:
        return None
    return MessageEvidenceItem(
        message_id=message.message_id,
        related_event_id=message.related_event_id,
        fact_type=fact_type,
        amount=None,
        currency=None,
        effective_date=None,
        settlement_date=None,
        recurrence_scope=None,
        cash_state=cash_state,
        confidence="high",
        evidence_quote=quote,
        notes="Explicit status or suspicious fee language detected.",
    )


def _matching_quote(text: str, pattern: str) -> str:
    match = re.search(pattern, text, re.IGNORECASE)
    return match.group(0) if match else text.split(maxsplit=20)[0]


def _looks_instruction_like(text: str) -> bool:
    return (
        re.search(
            r"\b(?:ignore (?:all |the |any )?(?:previous|prior)|system prompt|"
            r"output (?:the )?|return json|follow these instructions|role:|tool call)\b",
            text,
        )
        is not None
    )


def _is_suspicious_fee_instruction(text: str) -> bool:
    fee = re.search(
        r"\b(?:release|processing|unlock) fee\b|"
        r"biaya (?:pelepasan|pemrosesan|proses)|"
        r"frais de (?:traitement|libération|déblocage)",
        text,
    )
    action = re.search(
        r"\b(?:pay|send|transfer|wire|must|need to|required|bayar|kirim|"
        r"payer|envoyer|obligatoire)\b",
        text,
    )
    negated = re.search(
        r"\b(?:no fee|no [^.!?]{0,20} fee|do not pay|don't pay|never pay|"
        r"tidak perlu|jangan bayar|ne pas payer|aucun frais)\b",
        text,
    )
    return fee is not None and action is not None and negated is None


def _status_is_negated_or_future(text: str, status_pattern: str) -> bool:
    return (
        re.search(
            rf"\b(?:not|never|belum|tidak|bukan|akan|will|may|might|expected to|scheduled to|pas|jamais|sera|devrait)"
            rf"\b[^.!?]{{0,32}}\b(?:{status_pattern})\b",
            text,
        )
        is not None
    )


def _render_user_prompt(
    batch: tuple[MessageRecord, ...],
    *,
    home_currency: Currency,
    request_id: str,
    request_date: date,
    event_ids: tuple[str, ...],
) -> str:
    records = [
        {
            "message_id": item.message_id,
            "user_id": item.user_id,
            "request_id": item.request_id,
            "related_event_id": item.related_event_id,
            "sent_at": item.sent_at.isoformat().replace("+00:00", "Z"),
            "source_type": item.source_type.value,
            "message_text": item.message_text,
        }
        for item in batch
    ]
    values = {
        "{{HOME_CURRENCY}}": home_currency.value,
        "{{REQUEST_ID}}": request_id,
        "{{REQUEST_DATE}}": request_date.isoformat(),
        "{{KNOWN_EVENT_IDS_JSON}}": json.dumps(event_ids, ensure_ascii=False),
        "{{MESSAGE_RECORDS_JSON}}": json.dumps(records, ensure_ascii=False),
    }
    prompt = _USER_TEMPLATE
    for marker, value in values.items():
        prompt = prompt.replace(marker, value)
    return prompt


def _render_repair(error: Exception) -> str:
    safe_errors = json.dumps(
        [{"type": type(error).__name__, "message": str(error)[:300]}],
        ensure_ascii=False,
    )
    return _REPAIR_TEMPLATE.replace("{{VALIDATION_ERRORS_JSON}}", safe_errors)


def summarize_fact_types(run: MessageExtractionRun) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in run.evidence:
        key = result.item.fact_type
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


__all__ = [
    "ExtractedMessageEvidence",
    "MAX_BATCH_SIZE",
    "MAX_BATCH_SOURCE_CHARS",
    "MessageExtractionFailure",
    "MessageExtractionRun",
    "MessageExtractionSummary",
    "extract_message_evidence",
    "filter_relevant_messages",
    "summarize_fact_types",
]
