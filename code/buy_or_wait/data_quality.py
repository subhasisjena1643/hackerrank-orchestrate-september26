"""Read-only, deterministic data-quality auditing for participant inputs."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import timezone
from hashlib import sha256
from pathlib import Path
from typing import Callable

from buy_or_wait.data_loader import (
    CSV_SCHEMAS,
    InputValidationError,
    _parse_event,
    _parse_image,
    _parse_message,
    _parse_option,
    _parse_profile,
    _parse_rate,
    _parse_request,
    load_dataset,
)


OPTIONAL_FIELDS = frozenset(
    {
        "max_installment_months",
        "amount",
        "settlement_date",
        "linked_event_id",
        "minimum_allowed_amount",
        "earliest_date_for_full_payment",
        "payment_frequency_days",
        "request_id",
        "related_event_id",
    }
)
MISSING_MARKERS = frozenset({"", "na", "n/a", "null", "none"})
IDENTIFIERS = {
    "financial_profiles.csv": "user_id",
    "financial_events.csv": "event_id",
    "requests.csv": "request_id",
    "sample_requests.csv": "request_id",
    "request_payment_options.csv": "payment_option_id",
    "messages.csv": "message_id",
    "images.csv": "image_id",
}


@dataclass(slots=True)
class FileQuality:
    filename: str
    sha256: str = ""
    raw_count: int = 0
    valid_count: int = 0
    rejected_count: int = 0
    missing_fields: dict[str, int] = field(default_factory=dict)
    duplicate_ids: list[str] = field(default_factory=list)
    exact_duplicates: list[str] = field(default_factory=list)
    broken_media_references: list[str] = field(default_factory=list)
    transformations: dict[str, int] = field(default_factory=dict)
    rejected_records: list[str] = field(default_factory=list)


@dataclass(slots=True)
class DataQualityReport:
    dataset_dir: Path
    files: tuple[FileQuality, ...]
    dataset_errors: tuple[str, ...]
    unresolved_risks: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.dataset_errors and all(
            item.rejected_count == 0 and not item.broken_media_references
            for item in self.files
        )


def _normalized(column: str, value: str, quality: FileQuality) -> str:
    result = value.strip()
    if result != value:
        quality.transformations["trimmed_outer_whitespace"] = (
            quality.transformations.get("trimmed_outer_whitespace", 0) + 1
        )
    if column in OPTIONAL_FIELDS and result.casefold() in MISSING_MARKERS:
        if result:
            quality.transformations["missing_marker_to_blank"] = (
                quality.transformations.get("missing_marker_to_blank", 0) + 1
            )
        return ""
    return result


def _parser(
    filename: str, root: Path
) -> Callable[[Path, int, dict[str, str]], object] | None:
    return {
        "financial_profiles.csv": _parse_profile,
        "financial_events.csv": _parse_event,
        "exchange_rates.csv": _parse_rate,
        "requests.csv": _parse_request,
        "sample_requests.csv": _parse_request,
        "request_payment_options.csv": _parse_option,
        "messages.csv": _parse_message,
        "images.csv": _parse_image(root),
    }.get(filename)


def audit_dataset(dataset_dir: Path | str) -> DataQualityReport:
    """Audit without writing to or removing any raw record."""

    root = Path(dataset_dir).resolve()
    results: list[FileQuality] = []
    for filename in sorted(CSV_SCHEMAS):
        path = root / filename
        quality = FileQuality(filename)
        results.append(quality)
        try:
            payload = path.read_bytes()
            quality.sha256 = sha256(payload).hexdigest()
            text = payload.decode("utf-8-sig")
        except (OSError, UnicodeDecodeError) as exc:
            quality.rejected_count = 1
            quality.rejected_records.append(f"file: {type(exc).__name__}: {exc}")
            continue

        reader = csv.reader(text.splitlines(), strict=True)
        try:
            headers = next(reader)
        except (StopIteration, csv.Error) as exc:
            quality.rejected_count = 1
            quality.rejected_records.append(f"header: malformed or missing: {exc}")
            continue
        expected = CSV_SCHEMAS[filename]
        if tuple(headers) != expected:
            quality.rejected_count = 1
            quality.rejected_records.append(
                f"header: expected {','.join(expected)}; received {','.join(headers)}"
            )
            continue

        seen_ids: dict[str, int] = {}
        seen_rows: dict[tuple[str, ...], int] = {}
        parser = _parser(filename, root)
        for row_number, values in enumerate(reader, start=2):
            quality.raw_count += 1
            reasons: list[str] = []
            if len(values) != len(headers):
                reasons.append(f"row has {len(values)} fields; expected {len(headers)}")
                row = None
            else:
                row = {
                    column: _normalized(column, value, quality)
                    for column, value in zip(headers, values)
                }
                for column, value in row.items():
                    if value == "":
                        quality.missing_fields[column] = (
                            quality.missing_fields.get(column, 0) + 1
                        )
                identity_column = IDENTIFIERS.get(filename)
                if identity_column:
                    identity = row[identity_column]
                    if identity in seen_ids:
                        quality.duplicate_ids.append(
                            f"{identity!r} rows {seen_ids[identity]} and {row_number}"
                        )
                        reasons.append(
                            f"duplicate {identity_column}; first seen on row {seen_ids[identity]}"
                        )
                    elif identity:
                        seen_ids[identity] = row_number
                fingerprint = tuple(row[column] for column in headers)
                if fingerprint in seen_rows:
                    quality.exact_duplicates.append(
                        f"rows {seen_rows[fingerprint]} and {row_number}"
                    )
                else:
                    seen_rows[fingerprint] = row_number
                if filename == "messages.csv" and row["sent_at"]:
                    try:
                        parsed = _parse_message(path, row_number, row)
                        canonical = parsed.sent_at.astimezone(timezone.utc).isoformat()
                        if canonical != row["sent_at"]:
                            quality.transformations["timestamp_to_utc"] = (
                                quality.transformations.get("timestamp_to_utc", 0) + 1
                            )
                    except InputValidationError:
                        pass
                if filename == "images.csv" and row["image_id"]:
                    media = root / "media" / "images" / f"{row['image_id']}.png"
                    if not media.is_file():
                        reference = f"row {row_number}: {media}"
                        quality.broken_media_references.append(reference)
                if parser is not None:
                    try:
                        parser(path, row_number, row)
                    except InputValidationError as exc:
                        reasons.append(str(exc))
            if reasons:
                quality.rejected_count += 1
                quality.rejected_records.append(
                    f"row {row_number}: {'; '.join(dict.fromkeys(reasons))}"
                )
            else:
                quality.valid_count += 1

    dataset_errors: list[str] = []
    if not any(item.rejected_count for item in results):
        try:
            load_dataset(root, include_sample_outputs=True)
        except InputValidationError as exc:
            dataset_errors.append(str(exc))
    unresolved = tuple(
        f"{item.filename}: exact duplicates require human classification and were not removed."
        for item in results
        if item.exact_duplicates
    )
    return DataQualityReport(root, tuple(results), tuple(dataset_errors), unresolved)


def render_markdown(report: DataQualityReport) -> str:
    def portable(value: str) -> str:
        return value.replace(str(report.dataset_dir), report.dataset_dir.name)

    total_raw = sum(item.raw_count for item in report.files)
    total_valid = sum(item.valid_count for item in report.files)
    total_rejected = sum(item.rejected_count for item in report.files)
    lines = [
        "# Data-quality report",
        "",
        f"Dataset: `{report.dataset_dir.name}/`",
        "",
        f"Status: **{'PASS' if report.passed else 'FAIL'}**",
        "",
        f"Totals: raw={total_raw}, valid={total_valid}, rejected={total_rejected}",
        "",
        "Ordering: stable source-row order is the canonical in-memory order; this preserves linked-event precedence and legitimate recurring records.",
        "",
        "| File | Raw | Valid | Rejected | SHA-256 |",
        "|---|---:|---:|---:|---|",
    ]
    for item in report.files:
        lines.append(
            f"| {item.filename} | {item.raw_count} | {item.valid_count} | {item.rejected_count} | `{item.sha256}` |"
        )
    lines.extend(["", "## Findings", ""])
    for item in report.files:
        missing = (
            ", ".join(f"{k}={v}" for k, v in sorted(item.missing_fields.items()))
            or "none"
        )
        transforms = (
            ", ".join(f"{k}={v}" for k, v in sorted(item.transformations.items()))
            or "none required"
        )
        lines.extend(
            [
                f"### {item.filename}",
                "",
                f"- Missing fields: {missing}",
                f"- Duplicate IDs: {', '.join(item.duplicate_ids) or 'none'}",
                f"- Exact duplicates: {', '.join(item.exact_duplicates) or 'none'}",
                f"- Broken media references: {', '.join(portable(x) for x in item.broken_media_references) or 'none'}",
                f"- Transformations applied in memory: {transforms}",
                f"- Rejected records: {' | '.join(portable(x) for x in item.rejected_records) or 'none'}",
                "",
            ]
        )
    lines.extend(
        [
            "## Dataset relationship errors",
            "",
            *(f"- {portable(x)}" for x in report.dataset_errors or ("none",)),
            "",
            "## Unresolved risks",
            "",
            *(f"- {portable(x)}" for x in report.unresolved_risks or ("none",)),
            "",
        ]
    )
    return "\n".join(lines)


__all__ = ["DataQualityReport", "FileQuality", "audit_dataset", "render_markdown"]
