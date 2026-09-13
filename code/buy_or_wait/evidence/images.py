"""Phase 6 image OCR/vision extraction; never makes a financial decision."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
import re
import shutil
import subprocess
from typing import Protocol, cast

from buy_or_wait.domain import Currency, Dataset, FinancialEvent, ImageRecord, Profile
from buy_or_wait.telemetry import ModelPrice, TelemetryLedger, build_telemetry_record

from .cache import CacheIdentity, compose_cache_source, make_cache_key
from .client import EvidenceClient, ProviderResponse
from .image_consensus import (
    Confidence,
    ConsensusResult,
    OCRCandidate,
    OCRChannelResult,
    OCRStatus,
    VisionChannelResult,
    VisionStatus,
    decimal_text,
    label_supports_event,
    normalize_currency,
    normalize_decimal,
    reach_consensus,
)
from .schemas import (
    EVIDENCE_SCHEMA_VERSION,
    IMAGE_ADJUDICATION_JSON_SCHEMA,
    IMAGE_ADJUDICATION_PROMPT_VERSION,
    IMAGE_EVIDENCE_JSON_SCHEMA,
    IMAGE_PROMPT_VERSION,
    ImageAdjudicationResponse,
    ImageEvidenceResponse,
    parse_image_adjudication_response,
    parse_image_response,
)


PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_PROMPT_DIR = Path(__file__).resolve().parents[2] / "prompts"
_SYSTEM_PROMPT = (_PROMPT_DIR / "image_evidence.md").read_text(encoding="utf-8")
_USER_TEMPLATE = (_PROMPT_DIR / "image_evidence_user.md").read_text(encoding="utf-8")
_REPAIR_TEMPLATE = (_PROMPT_DIR / "image_evidence_repair.md").read_text(
    encoding="utf-8"
)
_ADJUDICATION_TEMPLATE = (_PROMPT_DIR / "image_adjudication.md").read_text(
    encoding="utf-8"
)

_AMOUNT_TOKEN = re.compile(
    r"(?<![A-Za-z0-9])(?:INR|IDR|USD|EUR|ZAR|US\$|Rp\.?|Rs\.?|[₹€$R])?\s*"
    r"([0-9]{1,3}(?:[ ,.][0-9]{2,3})+|[0-9]+(?:[.,][0-9]{1,2})?)"
    r"\s*(?:INR|IDR|USD|EUR|ZAR)?(?![A-Za-z0-9])",
    re.IGNORECASE,
)
_CURRENCY_PATTERNS = (
    (Currency.INR, re.compile(r"(?:\bINR\b|₹|\bRs\.?)", re.I)),
    (Currency.IDR, re.compile(r"(?:\bIDR\b|\bRp\.?)", re.I)),
    (Currency.USD, re.compile(r"(?:\bUSD\b|US\$|\$)", re.I)),
    (Currency.EUR, re.compile(r"(?:\bEUR\b|€)", re.I)),
    (Currency.ZAR, re.compile(r"(?:\bZAR\b|(?<![A-Za-z])R\s*\d)", re.I)),
)
_MONEY_LABEL = re.compile(
    r"\b(?:amount|balance|due|fare|grand total|net pay|net payable|paid|payable|salary|total)\b",
    re.IGNORECASE,
)
_INSTRUCTION = re.compile(
    r"\b(?:ignore (?:all |any |the )?(?:previous|prior)|system prompt|"
    r"do not follow|call (?:a |the )?tool|override (?:the )?rules|you are now)\b",
    re.IGNORECASE,
)
_FORBIDDEN_KEYS = frozenset(
    {
        "amount_safe_to_pay",
        "affordability_status",
        "recommended_payment_method",
        "payment_plan",
        "spending_changes_needed",
        "decision_explanation",
        "tool_call",
        "function_call",
        "new_event",
        "instructions",
        "prompt_override",
    }
)


class ImageEvidenceError(RuntimeError):
    """Base failure for Phase 6."""


class ImageValidationError(ImageEvidenceError):
    """Raised before extraction when file or relationships are unsafe."""


class UnresolvedImageAmountError(ImageEvidenceError):
    """Hard failure: a linked blank amount remained unresolved."""


class OCRExtractor(Protocol):
    version: str

    def extract(self, image_path: Path) -> OCRChannelResult:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class ImageResolution:
    image_id: str
    event_id: str
    document_type: str
    linked_event_amount: Decimal
    currency: Currency
    relevant_date: date | None
    confidence: str
    evidence_label: str
    instruction_text_detected: bool
    ocr_result: OCRChannelResult
    vision_result: VisionChannelResult
    resolution_status: str
    reason: str
    cache_hit: bool = False

    def __post_init__(self) -> None:
        if self.linked_event_amount <= 0 or not self.linked_event_amount.is_finite():
            raise ValueError("resolved image amount must be positive and finite")
        if self.confidence not in {"high", "medium"}:
            raise ValueError("resolved image confidence must be high or medium")
        if self.resolution_status not in {"consensus", "adjudicated"}:
            raise ValueError("resolved image must record consensus or adjudication")
        if not self.evidence_label.strip():
            raise ValueError("resolved image must retain a supporting evidence label")


class TesseractOCR:
    """Optional local OCR. Absence is explicit and triggers adjudication."""

    def __init__(self, executable: str | None = None) -> None:
        self.executable = executable or shutil.which("tesseract")
        self.version = self._version()

    def _version(self) -> str:
        if not self.executable:
            return "tesseract-unavailable"
        try:
            output = subprocess.run(
                [self.executable, "--version"],
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            ).stdout.splitlines()[0]
            return re.sub(r"\s+", "-", output.strip().casefold())[:80]
        except (OSError, subprocess.SubprocessError, IndexError):
            return "tesseract-version-error"

    def extract(self, image_path: Path) -> OCRChannelResult:
        if not self.executable:
            return OCRChannelResult(
                "unavailable", self.version, reason="local OCR executable not found"
            )
        try:
            completed = subprocess.run(
                [self.executable, str(image_path), "stdout", "--psm", "6"],
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            return candidates_from_ocr_text(completed.stdout, version=self.version)
        except (OSError, subprocess.SubprocessError) as exc:
            return OCRChannelResult("error", self.version, reason=type(exc).__name__)


def candidates_from_ocr_text(
    text: str, *, version: str = "test-ocr"
) -> OCRChannelResult:
    """Normalize every plausible labelled monetary candidate without ranking by size."""

    candidates: list[OCRCandidate] = []
    seen: set[tuple[Decimal, Currency | None, str]] = set()
    for raw_line in text.splitlines():
        line = " ".join(raw_line.strip().split())
        if not line:
            continue
        currency = _currency_in_text(line)
        labelled = _MONEY_LABEL.search(line) is not None
        if currency is None and not labelled:
            continue
        for match in _AMOUNT_TOKEN.finditer(line):
            token = match.group(1)
            if _looks_like_date_or_identifier(line, token, match.start(1)):
                continue
            try:
                amount = _parse_ocr_decimal(token)
            except ValueError:
                continue
            if amount <= 0:
                continue
            label = _candidate_label(line, match.start(), match.end())
            confidence: Confidence = (
                "high" if labelled and currency is not None else "medium"
            )
            key = (amount.normalize(), currency, label.casefold())
            if key not in seen:
                candidates.append(OCRCandidate(amount, currency, label, confidence))
                seen.add(key)
    status: OCRStatus = "success" if candidates else "no_confident_candidate"
    return OCRChannelResult(
        status,
        version,
        tuple(candidates),
        bool(_INSTRUCTION.search(text)),
        "labelled monetary candidates normalized"
        if candidates
        else "no defensible labelled monetary candidate",
    )


def resolve_image_amount(
    image: ImageRecord,
    event: FinancialEvent,
    profile: Profile,
    *,
    dataset_dir: Path | str,
    request_id: str,
    client: EvidenceClient,
    ocr: OCRExtractor | None = None,
    cache_dir: Path | None = None,
    telemetry: TelemetryLedger | None = None,
    run_id: str = "image-extraction",
    price: ModelPrice | None = None,
) -> ImageResolution:
    path, image_bytes = _validate_source(
        image, event, profile, Path(dataset_dir), request_id=request_id
    )
    extractor = ocr or TesseractOCR()
    user_prompt = _render_user_prompt(image, event, profile, request_id)
    ocr_result = extractor.extract(path)
    ocr_identity = json.dumps(
        {
            "status": ocr_result.status,
            "version": ocr_result.version,
            "candidates": [item.as_prompt_data() for item in ocr_result.candidates],
            "instruction_text_detected": ocr_result.instruction_text_detected,
            "reason": ocr_result.reason,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    identity = CacheIdentity(
        compose_cache_source(
            image_bytes,
            _SYSTEM_PROMPT,
            user_prompt,
            _REPAIR_TEMPLATE,
            _ADJUDICATION_TEMPLATE,
            ocr_identity,
            json.dumps(
                IMAGE_EVIDENCE_JSON_SCHEMA, sort_keys=True, separators=(",", ":")
            ),
            json.dumps(
                IMAGE_ADJUDICATION_JSON_SCHEMA, sort_keys=True, separators=(",", ":")
            ),
        ),
        f"{IMAGE_PROMPT_VERSION}+{IMAGE_ADJUDICATION_PROMPT_VERSION}+ocr:{extractor.version}",
        EVIDENCE_SCHEMA_VERSION,
        client.provider,
        client.vision_model,
    )
    if cache_dir is not None:
        cached = _cache_get(cache_dir, identity, image=image, event=event)
        if cached is not None:
            return _replace_cache_hit(cached)

    total_usage = [0, 0, 0]
    retry_count = 0
    try:
        vision_result, vision_usage, repair_count = _extract_vision(
            client, image_bytes, image, event, user_prompt
        )
        total_usage[:] = vision_usage
        retry_count += repair_count
        consensus = reach_consensus(ocr_result, vision_result, event)
        if consensus.status == "consensus":
            result = _from_consensus(image, event, ocr_result, vision_result, consensus)
        else:
            adjudicated, adjudication_usage, adjudication_retries = _adjudicate(
                client, image_bytes, image, event, ocr_result, vision_result
            )
            retry_count += 1 + adjudication_retries
            for index, value in enumerate(adjudication_usage):
                total_usage[index] += value
            result = _from_adjudication(
                image, event, ocr_result, vision_result, adjudicated
            )
    except Exception as error:
        _record_telemetry(
            telemetry,
            run_id,
            request_id,
            image.image_id,
            client,
            total_usage,
            retry_count,
            False,
            f"{type(error).__name__}: {str(error)[:200]}",
            price,
        )
        raise
    _record_telemetry(
        telemetry,
        run_id,
        request_id,
        image.image_id,
        client,
        total_usage,
        retry_count,
        True,
        None,
        price,
    )
    if cache_dir is not None:
        _cache_put(cache_dir, identity, result)
    return result


def resolve_all_linked_images(
    dataset: Dataset,
    *,
    dataset_dir: Path | str,
    client: EvidenceClient,
    ocr: OCRExtractor | None = None,
    cache_dir: Path | None = None,
    telemetry: TelemetryLedger | None = None,
    run_id: str = "image-extraction",
    price: ModelPrice | None = None,
) -> tuple[ImageResolution, ...]:
    events = {event.event_id: event for event in dataset.financial_events}
    profiles = {profile.user_id: profile for profile in dataset.profiles}
    results: list[ImageResolution] = []
    linked_blank_ids: set[str] = set()
    for image in dataset.images:
        if image.related_event_id is None:
            continue
        event = events[image.related_event_id]
        if event.amount is not None:
            continue
        if event.event_id in linked_blank_ids:
            raise ImageValidationError(
                f"blank event {event.event_id} has multiple linked images"
            )
        linked_blank_ids.add(event.event_id)
        if image.request_id is None:
            raise ImageValidationError(
                f"linked image {image.image_id} has no request_id"
            )
        results.append(
            resolve_image_amount(
                image,
                event,
                profiles[event.user_id],
                dataset_dir=dataset_dir,
                request_id=image.request_id,
                client=client,
                ocr=ocr,
                cache_dir=cache_dir,
                telemetry=telemetry,
                run_id=run_id,
                price=price,
            )
        )
    blank_ids = {
        event.event_id for event in dataset.financial_events if event.amount is None
    }
    missing = sorted(blank_ids - linked_blank_ids)
    if missing:
        raise UnresolvedImageAmountError(
            f"blank events without resolved linked image: {', '.join(missing)}"
        )
    return tuple(results)


def _validate_source(
    image: ImageRecord,
    event: FinancialEvent,
    profile: Profile,
    dataset_dir: Path,
    *,
    request_id: str,
) -> tuple[Path, bytes]:
    root = dataset_dir.resolve()
    expected = (root / "media" / "images" / f"{image.image_id}.png").resolve()
    if image.file_path.resolve() != expected:
        raise ImageValidationError(
            f"image {image.image_id} path is not the required dataset path"
        )
    if not expected.is_file():
        raise ImageValidationError(f"missing image file: {expected}")
    if expected.suffix.casefold() != ".png":
        raise ImageValidationError(f"image {image.image_id} is not a PNG path")
    image_bytes = expected.read_bytes()
    if (
        len(image_bytes) < 33
        or image_bytes[:8] != PNG_SIGNATURE
        or image_bytes[12:16] != b"IHDR"
    ):
        raise ImageValidationError(f"image {image.image_id} is not a valid PNG")
    if image.user_id != event.user_id or event.user_id != profile.user_id:
        raise ImageValidationError(f"user_id mismatch for image {image.image_id}")
    if image.request_id != request_id:
        raise ImageValidationError(f"request_id mismatch for image {image.image_id}")
    if image.related_event_id != event.event_id:
        raise ImageValidationError(
            f"related_event_id mismatch for image {image.image_id}"
        )
    if event.amount is not None:
        raise ImageValidationError(f"event {event.event_id} amount is not blank")
    return expected, image_bytes


def _extract_vision(
    client: EvidenceClient,
    image_bytes: bytes,
    image: ImageRecord,
    event: FinancialEvent,
    user_prompt: str,
) -> tuple[VisionChannelResult, tuple[int, int, int], int]:
    usage = [0, 0, 0]
    first_raw = ""
    provider_retries = 0
    try:
        first = client.extract_image(
            prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            source_id=image.image_id,
            image_bytes=image_bytes,
            mime_type="image/png",
            json_schema=IMAGE_EVIDENCE_JSON_SCHEMA,
        )
        first_raw = first.json_text
        provider_retries += first.retry_count
        _add_usage(first, usage)
        response = _validate_vision(first_raw, image, event)
        return (
            VisionChannelResult("success", response, "strict image response accepted"),
            (usage[0], usage[1], usage[2]),
            provider_retries,
        )
    except Exception as first_error:
        repair = _render_repair(first_error)
        repair_fn = getattr(client, "repair_image", None)
        if repair_fn is None:
            return (
                VisionChannelResult("invalid", None, type(first_error).__name__),
                (usage[0], usage[1], usage[2]),
                provider_retries,
            )
        try:
            second = repair_fn(
                system_prompt=_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                previous_response=first_raw,
                repair_prompt=repair,
                source_id=image.image_id,
                image_bytes=image_bytes,
                mime_type="image/png",
                json_schema=IMAGE_EVIDENCE_JSON_SCHEMA,
            )
            provider_retries += second.retry_count
            _add_usage(second, usage)
            response = _validate_vision(second.json_text, image, event)
            return (
                VisionChannelResult("repaired", response, "one schema repair accepted"),
                (usage[0], usage[1], usage[2]),
                1 + provider_retries,
            )
        except Exception as second_error:
            return (
                VisionChannelResult("invalid", None, type(second_error).__name__),
                (usage[0], usage[1], usage[2]),
                1 + provider_retries,
            )


def _validate_vision(
    raw: str, image: ImageRecord, event: FinancialEvent
) -> ImageEvidenceResponse:
    _reject_forbidden_keys(raw)
    response = parse_image_response(raw)
    if response.image_id != image.image_id or response.event_id != event.event_id:
        raise ImageValidationError("vision response changed trusted image/event IDs")
    if response.amount is not None:
        normalize_decimal(response.amount)
        if response.evidence_label is None:
            raise ImageValidationError("vision amount has no evidence label")
    return response


def _adjudicate(
    client: EvidenceClient,
    image_bytes: bytes,
    image: ImageRecord,
    event: FinancialEvent,
    ocr: OCRChannelResult,
    vision: VisionChannelResult,
) -> tuple[ImageAdjudicationResponse, tuple[int, int, int], int]:
    prompt = _render_adjudication(image, event, ocr, vision)
    adjudicate_fn = getattr(client, "adjudicate_image", None)
    if adjudicate_fn is None:
        raise UnresolvedImageAmountError(
            f"{image.image_id}/{event.event_id}: adjudication required but unavailable"
        )
    try:
        response = adjudicate_fn(
            prompt=prompt,
            source_id=image.image_id,
            image_bytes=image_bytes,
            mime_type="image/png",
            json_schema=IMAGE_ADJUDICATION_JSON_SCHEMA,
        )
        parsed = _validate_adjudication(response.json_text, image, event)
        return (
            parsed,
            (response.input_tokens, response.cached_tokens, response.output_tokens),
            response.retry_count,
        )
    except UnresolvedImageAmountError:
        raise
    except Exception as exc:
        raise UnresolvedImageAmountError(
            f"{image.image_id}/{event.event_id}: adjudication failed: {type(exc).__name__}"
        ) from exc


def _validate_adjudication(
    raw: str,
    image: ImageRecord,
    event: FinancialEvent,
) -> ImageAdjudicationResponse:
    _reject_forbidden_keys(raw)
    response = parse_image_adjudication_response(raw)
    if response.image_id != image.image_id or response.event_id != event.event_id:
        raise ImageValidationError("adjudicator changed trusted image/event IDs")
    if response.amount is None or response.selected_channel == "unresolved":
        raise UnresolvedImageAmountError(
            f"{image.image_id}/{event.event_id}: amount unresolved after adjudication"
        )
    amount = normalize_decimal(response.amount)
    currency = normalize_currency(response.currency)
    if currency != event.currency:
        raise ImageValidationError(
            f"adjudicated currency {response.currency} conflicts with event currency {event.currency.value}"
        )
    if response.confidence == "low" or not response.evidence_label:
        raise UnresolvedImageAmountError(
            f"{image.image_id}/{event.event_id}: adjudication evidence is not confident and labelled"
        )
    if amount <= 0:
        raise ImageValidationError("adjudicated amount is not positive")
    return response


def _from_consensus(
    image: ImageRecord,
    event: FinancialEvent,
    ocr: OCRChannelResult,
    vision: VisionChannelResult,
    consensus: ConsensusResult,
) -> ImageResolution:
    assert consensus.amount is not None and consensus.currency is not None
    assert consensus.evidence_label is not None and vision.response is not None
    return ImageResolution(
        image.image_id,
        event.event_id,
        vision.response.document_type,
        consensus.amount,
        consensus.currency,
        vision.response.document_date,
        consensus.confidence,
        consensus.evidence_label,
        consensus.instruction_text_detected,
        ocr,
        vision,
        "consensus",
        consensus.reason,
    )


def _from_adjudication(
    image: ImageRecord,
    event: FinancialEvent,
    ocr: OCRChannelResult,
    vision: VisionChannelResult,
    adjudicated: ImageAdjudicationResponse,
) -> ImageResolution:
    assert adjudicated.amount is not None and adjudicated.currency is not None
    assert adjudicated.evidence_label is not None
    vision_response = vision.response
    return ImageResolution(
        image.image_id,
        event.event_id,
        vision_response.document_type if vision_response else "unknown",
        normalize_decimal(adjudicated.amount),
        Currency(adjudicated.currency),
        vision_response.document_date if vision_response else None,
        adjudicated.confidence,
        adjudicated.evidence_label,
        ocr.instruction_text_detected
        or bool(vision_response and vision_response.instruction_text_detected)
        or adjudicated.instruction_text_detected,
        ocr,
        vision,
        "adjudicated",
        adjudicated.reason,
    )


def _render_user_prompt(
    image: ImageRecord,
    event: FinancialEvent,
    profile: Profile,
    request_id: str,
) -> str:
    values = {
        "{{IMAGE_ID}}": image.image_id,
        "{{EVENT_ID}}": event.event_id,
        "{{USER_ID}}": image.user_id,
        "{{REQUEST_ID}}": request_id,
        "{{EVENT_TYPE}}": event.event_type.value,
        "{{EVENT_DESCRIPTION}}": event.description,
        "{{EVENT_CATEGORY}}": event.category,
        "{{EVENT_DIRECTION}}": event.direction.value,
        "{{EVENT_CURRENCY}}": event.currency.value,
        "{{EVENT_DATE}}": event.event_date.isoformat(),
        "{{SETTLEMENT_DATE}}": event.settlement_date.isoformat()
        if event.settlement_date
        else "",
        "{{HOME_CURRENCY}}": profile.home_currency.value,
    }
    return _replace_template(_USER_TEMPLATE, values)


def _render_repair(error: Exception) -> str:
    safe = json.dumps(
        {"error": type(error).__name__, "message": str(error)[:300]},
        separators=(",", ":"),
    )
    return _REPAIR_TEMPLATE.replace("{{VALIDATION_ERRORS_JSON}}", safe)


def _render_adjudication(
    image: ImageRecord,
    event: FinancialEvent,
    ocr: OCRChannelResult,
    vision: VisionChannelResult,
) -> str:
    candidates = json.dumps(
        [candidate.as_prompt_data() for candidate in ocr.candidates],
        sort_keys=True,
        separators=(",", ":"),
    )
    vision_json = (
        vision.response.model_dump_json()
        if vision.response is not None
        else json.dumps(
            {"status": vision.status, "reason": vision.reason}, separators=(",", ":")
        )
    )
    values = {
        "{{IMAGE_ID}}": image.image_id,
        "{{EVENT_ID}}": event.event_id,
        "{{EVENT_DESCRIPTION}}": event.description,
        "{{EVENT_CATEGORY}}": event.category,
        "{{EVENT_DIRECTION}}": event.direction.value,
        "{{EVENT_CURRENCY}}": event.currency.value,
        "{{EVENT_DATE}}": event.event_date.isoformat(),
        "{{SETTLEMENT_DATE}}": event.settlement_date.isoformat()
        if event.settlement_date
        else "",
        "{{OCR_CANDIDATES_JSON}}": candidates,
        "{{VISION_RESULT_JSON}}": vision_json,
    }
    return _replace_template(_ADJUDICATION_TEMPLATE, values)


def _replace_template(template: str, values: dict[str, str]) -> str:
    rendered = template
    for marker, value in values.items():
        rendered = rendered.replace(marker, value)
    if "{{" in rendered or "}}" in rendered:
        raise ImageValidationError(
            "runtime image prompt contains an unresolved placeholder"
        )
    return rendered


def _currency_in_text(text: str) -> Currency | None:
    found = [
        currency for currency, pattern in _CURRENCY_PATTERNS if pattern.search(text)
    ]
    return found[0] if len(found) == 1 else None


def _parse_ocr_decimal(token: str) -> Decimal:
    raw = token.replace(" ", "")
    if "," in raw and "." in raw:
        decimal_sep = "," if raw.rfind(",") > raw.rfind(".") else "."
        grouping_sep = "." if decimal_sep == "," else ","
        raw = raw.replace(grouping_sep, "").replace(decimal_sep, ".")
    elif "," in raw:
        parts = raw.split(",")
        raw = (
            ".".join(parts)
            if len(parts[-1]) in {1, 2} and len(parts) == 2
            else "".join(parts)
        )
    elif "." in raw:
        parts = raw.split(".")
        raw = (
            ".".join(parts)
            if len(parts[-1]) in {1, 2} and len(parts) == 2
            else "".join(parts)
        )
    try:
        value = Decimal(raw)
    except InvalidOperation as exc:
        raise ValueError("invalid OCR decimal") from exc
    if not value.is_finite():
        raise ValueError("non-finite OCR decimal")
    return value


def _candidate_label(line: str, start: int, end: int) -> str:
    before = line[:start].strip(" :-|")
    after = line[end:].strip(" :-|")
    label = before or after or line
    return label[:160]


def _looks_like_date_or_identifier(line: str, token: str, position: int) -> bool:
    compact = re.sub(r"[ ,.]+", "", token)
    context = line[max(0, position - 24) : position].casefold()
    if any(
        word in context
        for word in ("account", "invoice no", "order no", "reference", "phone")
    ):
        return True
    if len(compact) >= 9 and _MONEY_LABEL.search(line) is None:
        return True
    return bool(
        re.fullmatch(r"(?:19|20)\d{2}", compact) and _MONEY_LABEL.search(line) is None
    )


def _reject_forbidden_keys(raw: str) -> None:
    try:
        value = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ImageValidationError("image provider returned invalid JSON") from exc

    def walk(item: object) -> None:
        if isinstance(item, dict):
            forbidden = {str(key).casefold() for key in item} & _FORBIDDEN_KEYS
            if forbidden:
                raise ImageValidationError(
                    f"forbidden image response field: {sorted(forbidden)[0]}"
                )
            for child in item.values():
                walk(child)
        elif isinstance(item, list):
            for child in item:
                walk(child)

    walk(value)


def _add_usage(response: ProviderResponse, usage: list[int]) -> None:
    for index, value in enumerate(
        (response.input_tokens, response.cached_tokens, response.output_tokens)
    ):
        usage[index] += value


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
            stage="image_evidence",
            source_id=source_id,
            provider=client.provider,
            model=client.vision_model,
            prompt_version=f"{IMAGE_PROMPT_VERSION}+{IMAGE_ADJUDICATION_PROMPT_VERSION}",
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


def _cache_path(directory: Path, identity: CacheIdentity) -> Path:
    return directory / f"image-resolution-{make_cache_key(identity)}.json"


def _cache_put(
    directory: Path, identity: CacheIdentity, result: ImageResolution
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    path = _cache_path(directory, identity)
    temporary = path.with_suffix(".tmp")
    payload = {"key": make_cache_key(identity), "result": _serialize_result(result)}
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _cache_get(
    directory: Path,
    identity: CacheIdentity,
    *,
    image: ImageRecord,
    event: FinancialEvent,
) -> ImageResolution | None:
    path = _cache_path(directory, identity)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("key") != make_cache_key(identity):
            return None
        result = _deserialize_result(payload["result"])
        if result.image_id != image.image_id or result.event_id != event.event_id:
            return None
        if result.currency != event.currency or result.linked_event_amount <= 0:
            return None
        if result.resolution_status == "consensus" and not label_supports_event(
            result.evidence_label, event
        ):
            return None
        return result
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _serialize_result(result: ImageResolution) -> dict[str, object]:
    return {
        "image_id": result.image_id,
        "event_id": result.event_id,
        "document_type": result.document_type,
        "linked_event_amount": decimal_text(result.linked_event_amount),
        "currency": result.currency.value,
        "relevant_date": result.relevant_date.isoformat()
        if result.relevant_date
        else None,
        "confidence": result.confidence,
        "evidence_label": result.evidence_label,
        "instruction_text_detected": result.instruction_text_detected,
        "ocr_result": {
            "status": result.ocr_result.status,
            "version": result.ocr_result.version,
            "instruction_text_detected": result.ocr_result.instruction_text_detected,
            "reason": result.ocr_result.reason,
            "candidates": [
                candidate.as_prompt_data() for candidate in result.ocr_result.candidates
            ],
        },
        "vision_result": {
            "status": result.vision_result.status,
            "reason": result.vision_result.reason,
            "response": result.vision_result.response.model_dump(mode="json")
            if result.vision_result.response
            else None,
        },
        "resolution_status": result.resolution_status,
        "reason": result.reason,
    }


def _deserialize_result(value: dict[str, object]) -> ImageResolution:
    ocr_value = value["ocr_result"]
    vision_value = value["vision_result"]
    if not isinstance(ocr_value, dict) or not isinstance(vision_value, dict):
        raise ValueError("invalid cached channel result")
    candidates = tuple(
        OCRCandidate(
            normalize_decimal(item["amount"]),
            normalize_currency(item.get("currency")),
            str(item["evidence_label"]),
            str(item["confidence"]),  # type: ignore[arg-type]
        )
        for item in ocr_value["candidates"]  # type: ignore[union-attr]
    )
    ocr = OCRChannelResult(
        cast(OCRStatus, str(ocr_value["status"])),
        str(ocr_value["version"]),
        candidates,  # type: ignore[arg-type]
        bool(ocr_value["instruction_text_detected"]),
        str(ocr_value["reason"]),
    )
    raw_vision = vision_value.get("response")
    vision_response = (
        None
        if raw_vision is None
        else ImageEvidenceResponse.model_validate_json(
            json.dumps(raw_vision, separators=(",", ":")), strict=True
        )
    )
    vision = VisionChannelResult(
        cast(VisionStatus, str(vision_value["status"])),
        vision_response,
        str(vision_value["reason"]),  # type: ignore[arg-type]
    )
    return ImageResolution(
        str(value["image_id"]),
        str(value["event_id"]),
        str(value["document_type"]),
        normalize_decimal(str(value["linked_event_amount"])),
        Currency(str(value["currency"])),
        date.fromisoformat(str(value["relevant_date"]))
        if value.get("relevant_date")
        else None,
        str(value["confidence"]),
        str(value["evidence_label"]),
        bool(value["instruction_text_detected"]),
        ocr,
        vision,
        str(value["resolution_status"]),
        str(value["reason"]),
    )


def _replace_cache_hit(result: ImageResolution) -> ImageResolution:
    return ImageResolution(
        result.image_id,
        result.event_id,
        result.document_type,
        result.linked_event_amount,
        result.currency,
        result.relevant_date,
        result.confidence,
        result.evidence_label,
        result.instruction_text_detected,
        result.ocr_result,
        result.vision_result,
        result.resolution_status,
        result.reason,
        True,
    )
