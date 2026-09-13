"""Privacy-minimized model usage telemetry and configured cost arithmetic."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
import re
from typing import Annotated, Iterable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


PRICING_SCHEMA_VERSION = "1.0.0"
TELEMETRY_SCHEMA_VERSION = "1.0.0"
MILLION = Decimal("1000000")


class TelemetryError(ValueError):
    pass


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class ModelPrice(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    input_usd_per_million_tokens: Decimal = Field(ge=0)
    cached_input_usd_per_million_tokens: Decimal = Field(ge=0)
    output_usd_per_million_tokens: Decimal = Field(ge=0)


class PricingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: str
    currency: str
    prices: tuple[ModelPrice, ...]

    @model_validator(mode="after")
    def validate_contract(self) -> "PricingConfig":
        if self.schema_version != PRICING_SCHEMA_VERSION:
            raise ValueError("unsupported pricing schema version")
        if self.currency != "USD":
            raise ValueError("pricing currency must be USD")
        keys = [(price.provider, price.model) for price in self.prices]
        if len(keys) != len(set(keys)):
            raise ValueError("duplicate provider/model pricing entry")
        return self

    def price_for(self, provider: str, model: str) -> ModelPrice:
        for price in self.prices:
            if price.provider == provider and price.model == model:
                return price
        raise TelemetryError(f"pricing is not configured for {provider}/{model}")


def load_pricing(path: Path) -> PricingConfig:
    try:
        return PricingConfig.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError, ValueError) as exc:
        raise TelemetryError(f"invalid pricing configuration: {path}") from exc


def estimate_cost(
    *, input_tokens: int, cached_tokens: int, output_tokens: int, price: ModelPrice
) -> Decimal:
    if min(input_tokens, cached_tokens, output_tokens) < 0:
        raise TelemetryError("token counts cannot be negative")
    if cached_tokens > input_tokens:
        raise TelemetryError("cached tokens cannot exceed input tokens")
    uncached_tokens = input_tokens - cached_tokens
    return (
        Decimal(uncached_tokens) * price.input_usd_per_million_tokens
        + Decimal(cached_tokens) * price.cached_input_usd_per_million_tokens
        + Decimal(output_tokens) * price.output_usd_per_million_tokens
    ) / MILLION


class TelemetryRecord(_StrictModel):
    telemetry_schema_version: str = TELEMETRY_SCHEMA_VERSION
    run_id: Annotated[str, Field(min_length=1, max_length=128)]
    request_id: Annotated[str, Field(min_length=1, max_length=128)]
    stage: Annotated[str, Field(min_length=1, max_length=128)]
    source_id: Annotated[str, Field(min_length=1, max_length=128)]
    provider: Annotated[str, Field(min_length=1, max_length=128)]
    model: Annotated[str, Field(min_length=1, max_length=256)]
    prompt_version: Annotated[str, Field(min_length=1, max_length=128)]
    schema_version: Annotated[str, Field(min_length=1, max_length=128)]
    input_tokens: int = Field(ge=0)
    cached_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    retry_count: int = Field(ge=0)
    estimated_cost: Decimal | None = Field(default=None, ge=0)
    timestamp: datetime
    success: bool
    failure: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def validate_consistency(self) -> "TelemetryRecord":
        if self.telemetry_schema_version != TELEMETRY_SCHEMA_VERSION:
            raise ValueError("unsupported telemetry schema version")
        if self.cached_tokens > self.input_tokens:
            raise ValueError("cached tokens cannot exceed input tokens")
        if self.success and self.failure is not None:
            raise ValueError("successful telemetry cannot include a failure")
        if not self.success and not self.failure:
            raise ValueError("failed telemetry requires a failure summary")
        return self


_SECRET_ASSIGNMENT_RE = re.compile(
    r"(?i)\b("
    r"[A-Z0-9_]*(?:API_KEY|ACCESS_TOKEN|AUTH_TOKEN|SECRET|PASSWORD)"
    r"|api[_-]?key|authorization|bearer|token|secret|password"
    r")\b(\s*[:=]\s*|\s+)[^\s,;]+"
)
_OPENAI_KEY_RE = re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b")


def redact_secrets(text: str, *, secrets: Iterable[str] = ()) -> str:
    redacted = _SECRET_ASSIGNMENT_RE.sub(
        lambda match: f"{match.group(1)}=[REDACTED]", text
    )
    redacted = _OPENAI_KEY_RE.sub("[REDACTED]", redacted)
    for secret in secrets:
        if secret:
            redacted = redacted.replace(secret, "[REDACTED]")
    return redacted


def build_telemetry_record(
    *,
    run_id: str,
    request_id: str,
    stage: str,
    source_id: str,
    provider: str,
    model: str,
    prompt_version: str,
    schema_version: str,
    input_tokens: int,
    cached_tokens: int,
    output_tokens: int,
    retry_count: int,
    success: bool,
    failure: str | None = None,
    price: ModelPrice | None = None,
    timestamp: datetime | None = None,
    secrets: Iterable[str] = (),
) -> TelemetryRecord:
    cost = (
        None
        if price is None
        else estimate_cost(
            input_tokens=input_tokens,
            cached_tokens=cached_tokens,
            output_tokens=output_tokens,
            price=price,
        )
    )
    clean_failure = (
        None if failure is None else redact_secrets(failure, secrets=secrets)
    )
    return TelemetryRecord(
        run_id=run_id,
        request_id=request_id,
        stage=stage,
        source_id=source_id,
        provider=provider,
        model=model,
        prompt_version=prompt_version,
        schema_version=schema_version,
        input_tokens=input_tokens,
        cached_tokens=cached_tokens,
        output_tokens=output_tokens,
        retry_count=retry_count,
        estimated_cost=cost,
        timestamp=timestamp or datetime.now(timezone.utc),
        success=success,
        failure=clean_failure,
    )


class TelemetryLedger:
    """Append-only JSONL ledger containing no prompts or credentials."""

    def __init__(self, path: Path, *, secrets: Iterable[str] = ()) -> None:
        self.path = path
        self._secrets = tuple(secret for secret in secrets if secret)

    def append(self, record: TelemetryRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        safe_record = record
        if record.failure is not None:
            safe_record = record.model_copy(
                update={
                    "failure": redact_secrets(
                        record.failure,
                        secrets=self._secrets,
                    )
                }
            )
        line = safe_record.model_dump_json(exclude_none=False)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)
            handle.write("\n")
