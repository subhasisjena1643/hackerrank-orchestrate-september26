"""Connected Phase 0-9 pipeline ending at baseline safe capacity."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from buy_or_wait.data_loader import load_request_contexts
from buy_or_wait.domain import RequestContext
from buy_or_wait.evidence.cache import EvidenceCache
from buy_or_wait.evidence.client import EvidenceClient
from buy_or_wait.evidence.images import (
    ImageResolution,
    ImageValidationError,
    OCRExtractor,
    resolve_image_amount,
)
from buy_or_wait.evidence.messages import MessageExtractionRun, extract_message_evidence
from buy_or_wait.finance.cashflows import (
    CashFlowProjection,
    construct_future_cash_flows,
)
from buy_or_wait.finance.forecast import BaselineForecast, calculate_baseline_forecast
from buy_or_wait.finance.lifecycle import ResolvedEvent, resolve_event_lifecycles
from buy_or_wait.finance.recurrence_policies import (
    ForecastPolicy,
    require_selected_policy,
)
from buy_or_wait.telemetry import ModelPrice, TelemetryLedger


class Phase9IntegrationError(RuntimeError):
    """A Phase 0-9 boundary failed closed with source context."""


@dataclass(frozen=True, slots=True)
class Phase9PipelineResult:
    context: RequestContext
    message_extraction: MessageExtractionRun
    image_resolutions: tuple[ImageResolution, ...]
    resolved_events: tuple[ResolvedEvent, ...]
    cash_flow_projection: CashFlowProjection
    baseline_forecast: BaselineForecast


def run_phase9_context(
    context: RequestContext,
    *,
    dataset_dir: Path | str,
    policy: ForecastPolicy | None,
    client: EvidenceClient,
    ocr: OCRExtractor | None = None,
    message_cache: EvidenceCache | None = None,
    image_cache_dir: Path | None = None,
    telemetry: TelemetryLedger | None = None,
    run_id: str = "phase9-pipeline",
    text_price: ModelPrice | None = None,
    vision_price: ModelPrice | None = None,
) -> Phase9PipelineResult:
    """Run one validated request through evidence, finance, and forecast.

    This function deliberately stops at Phase 9: it creates no spending changes,
    candidate plans, recommendations, explanations, or output rows.
    """

    selected_policy = require_selected_policy(policy)
    request = context.request
    known_event_ids = tuple(event.event_id for event in context.financial_events)
    message_run = extract_message_evidence(
        context.messages,
        user_id=request.user_id,
        request_id=request.request_id,
        request_date=request.request_date,
        home_currency=context.profile.home_currency,
        known_event_ids=known_event_ids,
        client=client,
        cache=message_cache,
        telemetry=telemetry,
        run_id=run_id,
        price=text_price,
    )
    if message_run.failures:
        details = "; ".join(
            f"{','.join(item.message_ids)}={item.reason}"
            for item in message_run.failures
        )
        raise Phase9IntegrationError(
            f"request {request.request_id}: message evidence failed: {details}"
        )

    images_by_event: dict[str, list[object]] = {}
    for image in context.images:
        if image.related_event_id is not None:
            images_by_event.setdefault(image.related_event_id, []).append(image)
    image_results: list[ImageResolution] = []
    for event in context.financial_events:
        if event.amount is not None:
            continue
        matches = images_by_event.get(event.event_id, [])
        if len(matches) != 1:
            raise ImageValidationError(
                f"event {event.event_id}: expected one linked image; found {len(matches)}"
            )
        image_results.append(
            resolve_image_amount(
                matches[0],  # type: ignore[arg-type]
                event,
                context.profile,
                dataset_dir=dataset_dir,
                request_id=request.request_id,
                client=client,
                ocr=ocr,
                cache_dir=image_cache_dir,
                telemetry=telemetry,
                run_id=run_id,
                price=vision_price,
            )
        )

    resolved = resolve_event_lifecycles(
        context.financial_events,
        snapshot_date=request.request_date,
        home_currency=context.profile.home_currency,
        exchange_rates=context.exchange_rates,
        message_evidence=message_run.evidence,
        image_resolutions=image_results,
    )
    projection = construct_future_cash_flows(
        resolved,
        request_date=request.request_date,
        policy=selected_policy,
        message_evidence=message_run.evidence,
    )
    forecast = calculate_baseline_forecast(
        current_available_balance=context.profile.current_available_balance,
        minimum_balance_to_keep=context.profile.minimum_balance_to_keep,
        request_date=request.request_date,
        requested_amount=request.requested_amount,
        cash_flows=projection.cash_flows,
        policy=selected_policy,
    )
    return Phase9PipelineResult(
        context,
        message_run,
        tuple(image_results),
        resolved,
        projection,
        forecast,
    )


def run_phase9_dataset_request(
    dataset_dir: Path | str,
    request_id: str,
    **kwargs: object,
) -> Phase9PipelineResult:
    """Load normalized evaluation inputs and run one request through Phase 9."""

    contexts = load_request_contexts(dataset_dir)
    matches = tuple(
        context for context in contexts if context.request.request_id == request_id
    )
    if len(matches) != 1:
        raise Phase9IntegrationError(
            f"request_id {request_id!r} must identify exactly one evaluation request"
        )
    return run_phase9_context(matches[0], dataset_dir=dataset_dir, **kwargs)  # type: ignore[arg-type]


__all__ = [
    "Phase9IntegrationError",
    "Phase9PipelineResult",
    "run_phase9_context",
    "run_phase9_dataset_request",
]
