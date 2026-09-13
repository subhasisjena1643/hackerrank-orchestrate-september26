"""Minimal provider boundary for evidence extraction.

Only the OpenAI provider has a live adapter. The offline implementation is a
deterministic test double and never fabricates a model call.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping, Protocol

from dotenv import dotenv_values


_CODE_DIR = Path(__file__).resolve().parents[2]
_REPO_ROOT = _CODE_DIR.parent


class EvidenceClientError(RuntimeError):
    pass


class EvidenceConfigurationError(EvidenceClientError):
    pass


class EvidenceProviderError(EvidenceClientError):
    pass


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    json_text: str
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    retry_count: int = 0

    def __post_init__(self) -> None:
        if (
            min(
                self.input_tokens,
                self.cached_tokens,
                self.output_tokens,
                self.retry_count,
            )
            < 0
        ):
            raise ValueError("token counts cannot be negative")
        if self.cached_tokens > self.input_tokens:
            raise ValueError("cached tokens cannot exceed input tokens")


class EvidenceClient(Protocol):
    provider: str
    text_model: str
    vision_model: str

    def extract_text(
        self,
        *,
        prompt: str,
        source_id: str,
        source_text: str,
        json_schema: Mapping[str, object],
    ) -> ProviderResponse:
        raise NotImplementedError

    def repair_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        previous_response: str,
        repair_prompt: str,
        source_id: str,
        json_schema: Mapping[str, object],
    ) -> ProviderResponse:
        raise NotImplementedError

    def extract_image(
        self,
        *,
        prompt: str,
        user_prompt: str = "",
        source_id: str,
        image_bytes: bytes,
        mime_type: str,
        json_schema: Mapping[str, object],
    ) -> ProviderResponse:
        raise NotImplementedError

    def repair_image(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        previous_response: str,
        repair_prompt: str,
        source_id: str,
        image_bytes: bytes,
        mime_type: str,
        json_schema: Mapping[str, object],
    ) -> ProviderResponse:
        raise NotImplementedError

    def adjudicate_image(
        self,
        *,
        prompt: str,
        source_id: str,
        image_bytes: bytes,
        mime_type: str,
        json_schema: Mapping[str, object],
    ) -> ProviderResponse:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class EvidenceEnvironmentStatus:
    provider_configured: bool
    provider_supported: bool
    text_model_configured: bool
    vision_model_configured: bool
    credential_configured: bool
    live_ready: bool


def inspect_evidence_environment(
    environ: Mapping[str, str] | None = None,
) -> EvidenceEnvironmentStatus:
    env = _default_environment() if environ is None else environ
    provider = env.get("EVIDENCE_PROVIDER", "").strip().casefold()
    text = bool(env.get("EVIDENCE_TEXT_MODEL", "").strip())
    vision = bool(env.get("EVIDENCE_VISION_MODEL", "").strip())
    credential = bool(env.get("OPENAI_API_KEY", "").strip())
    supported = provider == "openai"
    return EvidenceEnvironmentStatus(
        bool(provider),
        supported,
        text,
        vision,
        credential,
        supported and text and vision and credential,
    )


class OpenAIEvidenceClient:
    """The single live provider adapter. The SDK is imported lazily."""

    provider = "openai"

    def __init__(self, *, api_key: str, text_model: str, vision_model: str) -> None:
        if not api_key.strip():
            raise EvidenceConfigurationError(
                "OPENAI_API_KEY is required for live extraction"
            )
        if not text_model.strip() or not vision_model.strip():
            raise EvidenceConfigurationError(
                "EVIDENCE_TEXT_MODEL and EVIDENCE_VISION_MODEL are required"
            )
        self.text_model = text_model
        self.vision_model = vision_model
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency is pinned
            raise EvidenceConfigurationError(
                "the openai package is not installed"
            ) from exc
        self._client = OpenAI(api_key=api_key, max_retries=0, timeout=60.0)

    def extract_text(
        self,
        *,
        prompt: str,
        source_id: str,
        source_text: str,
        json_schema: Mapping[str, object],
    ) -> ProviderResponse:
        return self._complete(
            self.text_model,
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": source_text},
            ],
            json_schema,
            "message_evidence",
        )

    def repair_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        previous_response: str,
        repair_prompt: str,
        source_id: str,
        json_schema: Mapping[str, object],
    ) -> ProviderResponse:
        return self._complete(
            self.text_model,
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": previous_response},
                {"role": "user", "content": repair_prompt},
            ],
            json_schema,
            "message_evidence",
        )

    def extract_image(
        self,
        *,
        prompt: str,
        user_prompt: str = "",
        source_id: str,
        image_bytes: bytes,
        mime_type: str,
        json_schema: Mapping[str, object],
    ) -> ProviderResponse:
        if mime_type not in {"image/png", "image/jpeg", "image/webp"}:
            raise EvidenceConfigurationError(
                f"unsupported image MIME type: {mime_type}"
            )
        content = self._image_content(user_prompt, image_bytes, mime_type)
        return self._complete(
            self.vision_model,
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": content},
            ],
            json_schema,
            "image_evidence",
        )

    def repair_image(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        previous_response: str,
        repair_prompt: str,
        source_id: str,
        image_bytes: bytes,
        mime_type: str,
        json_schema: Mapping[str, object],
    ) -> ProviderResponse:
        return self._complete(
            self.vision_model,
            [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": self._image_content(user_prompt, image_bytes, mime_type),
                },
                {"role": "assistant", "content": previous_response},
                {
                    "role": "user",
                    "content": self._image_content(
                        repair_prompt, image_bytes, mime_type
                    ),
                },
            ],
            json_schema,
            "image_evidence",
        )

    def adjudicate_image(
        self,
        *,
        prompt: str,
        source_id: str,
        image_bytes: bytes,
        mime_type: str,
        json_schema: Mapping[str, object],
    ) -> ProviderResponse:
        return self._complete(
            self.vision_model,
            [
                {
                    "role": "user",
                    "content": self._image_content(prompt, image_bytes, mime_type),
                }
            ],
            json_schema,
            "image_adjudication",
        )

    @staticmethod
    def _image_content(
        text: str, image_bytes: bytes, mime_type: str
    ) -> list[dict[str, object]]:
        if mime_type != "image/png":
            raise EvidenceConfigurationError("Phase 6 accepts PNG images only")
        encoded = base64.b64encode(image_bytes).decode("ascii")
        content: list[dict[str, object]] = []
        if text:
            content.append({"type": "text", "text": text})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
            }
        )
        return content

    def _complete(
        self,
        model: str,
        messages: list[dict[str, object]],
        schema: Mapping[str, object],
        name: str,
    ) -> ProviderResponse:
        request: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": name,
                    "strict": True,
                    "schema": dict(schema),
                },
            },
        }
        # GPT-5 reasoning models accept only their fixed default temperature.
        # Other configured models retain temperature zero for repeatability.
        if not model.casefold().startswith("gpt-5"):
            request["temperature"] = 0
        for attempt in range(2):
            try:
                response = self._client.chat.completions.create(**request)
                if not response.choices:
                    raise EvidenceProviderError("provider returned no choices")
                message = response.choices[0].message
                refusal = getattr(message, "refusal", None)
                if refusal:
                    raise EvidenceProviderError("provider refused evidence extraction")
                content = message.content
                usage = response.usage
                if not isinstance(content, str) or usage is None:
                    raise EvidenceProviderError(
                        "provider returned no structured content or usage"
                    )
                details = getattr(usage, "prompt_tokens_details", None)
                cached = getattr(details, "cached_tokens", 0) or 0
                return ProviderResponse(
                    content,
                    usage.prompt_tokens,
                    cached,
                    usage.completion_tokens,
                    attempt,
                )
            except EvidenceProviderError:
                raise
            except Exception as exc:
                if attempt == 0 and _is_retryable_provider_error(exc):
                    continue
                raise EvidenceProviderError(
                    f"OpenAI evidence request failed: {type(exc).__name__}"
                ) from exc
        raise AssertionError("unreachable provider retry state")


class DeterministicOfflineEvidenceClient:
    """Explicit-response test double; missing fixtures fail instead of inventing facts."""

    provider = "offline-test-double"

    def __init__(
        self,
        *,
        text_model: str,
        vision_model: str,
        responses: Mapping[tuple[str, str], str],
    ) -> None:
        if not text_model or not vision_model:
            raise EvidenceConfigurationError("offline model labels must be explicit")
        self.text_model = text_model
        self.vision_model = vision_model
        self._responses = dict(responses)

    def _response(self, kind: str, source_id: str) -> ProviderResponse:
        try:
            value = self._responses[(kind, source_id)]
        except KeyError as exc:
            raise EvidenceProviderError(
                f"no offline fixture for {kind} source {source_id}"
            ) from exc
        return ProviderResponse(value, 0, 0, 0)

    def extract_text(
        self,
        *,
        prompt: str,
        source_id: str,
        source_text: str,
        json_schema: Mapping[str, object],
    ) -> ProviderResponse:
        return self._response("message", source_id)

    def repair_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        previous_response: str,
        repair_prompt: str,
        source_id: str,
        json_schema: Mapping[str, object],
    ) -> ProviderResponse:
        return self._response("message_repair", source_id)

    def extract_image(
        self,
        *,
        prompt: str,
        user_prompt: str = "",
        source_id: str,
        image_bytes: bytes,
        mime_type: str,
        json_schema: Mapping[str, object],
    ) -> ProviderResponse:
        return self._response("image", source_id)

    def repair_image(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        previous_response: str,
        repair_prompt: str,
        source_id: str,
        image_bytes: bytes,
        mime_type: str,
        json_schema: Mapping[str, object],
    ) -> ProviderResponse:
        return self._response("image_repair", source_id)

    def adjudicate_image(
        self,
        *,
        prompt: str,
        source_id: str,
        image_bytes: bytes,
        mime_type: str,
        json_schema: Mapping[str, object],
    ) -> ProviderResponse:
        return self._response("image_adjudication", source_id)


def create_live_evidence_client(
    environ: Mapping[str, str] | None = None,
) -> EvidenceClient:
    """Create the env-selected live client, failing closed when incomplete."""

    env = _default_environment() if environ is None else environ
    provider = env.get("EVIDENCE_PROVIDER", "").strip().casefold()
    text_model = env.get("EVIDENCE_TEXT_MODEL", "").strip()
    vision_model = env.get("EVIDENCE_VISION_MODEL", "").strip()
    if not provider:
        raise EvidenceConfigurationError("EVIDENCE_PROVIDER is required")
    if provider != "openai":
        raise EvidenceConfigurationError(
            "unsupported EVIDENCE_PROVIDER; only openai is implemented"
        )
    if not text_model or not vision_model:
        raise EvidenceConfigurationError(
            "EVIDENCE_TEXT_MODEL and EVIDENCE_VISION_MODEL are required"
        )
    credential = env.get("OPENAI_API_KEY", "")
    if not credential.strip():
        raise EvidenceConfigurationError(
            "live evidence extraction is blocked: OPENAI_API_KEY is not configured"
        )
    return OpenAIEvidenceClient(
        api_key=credential,
        text_model=text_model,
        vision_model=vision_model,
    )


def _default_environment() -> dict[str, str]:
    """Load ignored dotenv configuration without mutating the process environment."""

    merged: dict[str, str] = {}
    for path in (_REPO_ROOT / ".env", _CODE_DIR / ".env"):
        if path.is_file():
            merged.update(
                {
                    key: value
                    for key, value in dotenv_values(path).items()
                    if value is not None
                }
            )
    merged.update(os.environ)
    return merged


def _is_retryable_provider_error(error: Exception) -> bool:
    """Retry one transient provider failure; never retry refusals/schema errors."""

    return isinstance(error, (TimeoutError, ConnectionError)) or type(
        error
    ).__name__ in {
        "APITimeoutError",
        "APIConnectionError",
        "RateLimitError",
        "InternalServerError",
    }
