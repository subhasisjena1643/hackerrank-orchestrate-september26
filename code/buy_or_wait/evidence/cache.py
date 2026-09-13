"""Content-addressed cache for validated structured evidence only."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from pydantic import ValidationError

from .guardrails import ValidatedEvidence, ValidatedMessageBatch
from .schemas import (
    ImageEvidenceResponse,
    MessageEvidenceBatchResponse,
    MessageEvidenceResponse,
)


CACHE_SCHEMA_VERSION = "1.0.0"


class EvidenceCacheError(RuntimeError):
    pass


def compose_cache_source(*parts: str | bytes) -> bytes:
    """Length-frame every extraction input so cache identities cannot alias."""

    payload = bytearray()
    for part in parts:
        value = part.encode("utf-8") if isinstance(part, str) else part
        payload.extend(len(value).to_bytes(8, "big"))
        payload.extend(value)
    return bytes(payload)


@dataclass(frozen=True, slots=True)
class CacheIdentity:
    source: str | bytes
    prompt_version: str
    schema_version: str
    provider: str
    model: str

    def __post_init__(self) -> None:
        for name in ("prompt_version", "schema_version", "provider", "model"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must not be blank")


def make_cache_key(identity: CacheIdentity) -> str:
    """Hash source bytes plus every model-boundary version dimension."""

    source_bytes = (
        identity.source.encode("utf-8")
        if isinstance(identity.source, str)
        else identity.source
    )
    digest = hashlib.sha256()
    for label, value in (
        (b"source", source_bytes),
        (b"prompt_version", identity.prompt_version.encode("utf-8")),
        (b"schema_version", identity.schema_version.encode("utf-8")),
        (b"provider", identity.provider.encode("utf-8")),
        (b"model", identity.model.encode("utf-8")),
    ):
        digest.update(len(label).to_bytes(2, "big"))
        digest.update(label)
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()


class EvidenceCache:
    def __init__(self, directory: Path) -> None:
        self.directory = directory

    def get(
        self, identity: CacheIdentity
    ) -> ValidatedEvidence | ValidatedMessageBatch | None:
        key = make_cache_key(identity)
        path = self.directory / f"{key}.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or set(payload) != {
                "cache_schema_version",
                "key",
                "response",
            }:
                raise EvidenceCacheError("cached evidence envelope is invalid")
            if payload["cache_schema_version"] != CACHE_SCHEMA_VERSION:
                raise EvidenceCacheError(
                    "cached evidence schema version is unsupported"
                )
            if payload["key"] != key:
                raise EvidenceCacheError(
                    "cached evidence key does not match content identity"
                )
            response = payload["response"]
            if not isinstance(response, dict):
                raise EvidenceCacheError("cached evidence response is invalid")
            if "items" in response:
                return ValidatedMessageBatch(
                    MessageEvidenceBatchResponse.model_validate_json(
                        json.dumps(response, separators=(",", ":")), strict=True
                    )
                )
            kind = response.get("source_kind")
            encoded_response = json.dumps(response, separators=(",", ":"))
            if kind == "message":
                return ValidatedEvidence(
                    MessageEvidenceResponse.model_validate_json(
                        encoded_response, strict=True
                    )
                )
            if "image_id" in response and "event_id" in response:
                return ValidatedEvidence(
                    ImageEvidenceResponse.model_validate_json(
                        encoded_response, strict=True
                    )
                )
            raise EvidenceCacheError("cached evidence source kind is invalid")
        except (OSError, json.JSONDecodeError, ValidationError) as exc:
            raise EvidenceCacheError("cached evidence failed validation") from exc

    def put(
        self,
        identity: CacheIdentity,
        evidence: ValidatedEvidence | ValidatedMessageBatch,
    ) -> Path:
        """Persist typed output; raw or merely JSON-shaped values are rejected."""

        if not isinstance(
            evidence, (ValidatedEvidence, ValidatedMessageBatch)
        ) or not isinstance(
            evidence.response,
            (
                MessageEvidenceResponse,
                MessageEvidenceBatchResponse,
                ImageEvidenceResponse,
            ),
        ):
            raise TypeError("cache accepts only validated typed evidence responses")
        key = make_cache_key(identity)
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{key}.json"
        temporary = self.directory / f".{key}.tmp"
        payload = {
            "cache_schema_version": CACHE_SCHEMA_VERSION,
            "key": key,
            "response": evidence.response.model_dump(mode="json"),
        }
        temporary.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
            newline="\n",
        )
        temporary.replace(path)
        return path
