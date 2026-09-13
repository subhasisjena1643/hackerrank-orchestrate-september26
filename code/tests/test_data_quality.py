"""Data-engineering backfill regressions."""

from __future__ import annotations

from datetime import timezone
from pathlib import Path
import sys

import pytest

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from buy_or_wait.data_loader import CSV_SCHEMAS, load_dataset
from buy_or_wait.data_quality import audit_dataset, render_markdown
from buy_or_wait.validation.inputs import InputValidationError
from test_data_loader import _valid_rows, _write


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    rows = _valid_rows()
    for filename, headers in CSV_SCHEMAS.items():
        _write(tmp_path / filename, headers, rows[filename])
    return tmp_path


def test_timezone_offsets_normalize_to_same_utc_instant(dataset_dir: Path) -> None:
    rows = _valid_rows()["messages.csv"]
    rows[0]["sent_at"] = "2026-01-10T15:30:00+05:30"
    _write(dataset_dir / "messages.csv", CSV_SCHEMAS["messages.csv"], rows)
    messages = load_dataset(dataset_dir).messages
    assert messages[0].sent_at.tzinfo is timezone.utc
    assert messages[0].sent_at == messages[1].sent_at.replace(day=10)


def test_optional_missing_markers_and_whitespace_are_normalized(
    dataset_dir: Path,
) -> None:
    profiles = _valid_rows()["financial_profiles.csv"]
    profiles[0]["max_installment_months"] = " N/A "
    _write(
        dataset_dir / "financial_profiles.csv",
        CSV_SCHEMAS["financial_profiles.csv"],
        profiles,
    )
    messages = _valid_rows()["messages.csv"]
    messages[0]["request_id"] = " null "
    _write(dataset_dir / "messages.csv", CSV_SCHEMAS["messages.csv"], messages)
    loaded = load_dataset(dataset_dir)
    assert loaded.profiles[0].max_installment_months is None
    assert loaded.messages[0].request_id is None


def test_broken_media_reference_is_rejected_and_reported(dataset_dir: Path) -> None:
    images = [
        {
            "image_id": "missing",
            "user_id": "user_a",
            "request_id": "request_a",
            "related_event_id": "event_a",
        }
    ]
    _write(dataset_dir / "images.csv", CSV_SCHEMAS["images.csv"], images)
    with pytest.raises(
        InputValidationError, match="referenced image file does not exist"
    ):
        load_dataset(dataset_dir)
    report = audit_dataset(dataset_dir)
    item = next(x for x in report.files if x.filename == "images.csv")
    assert item.rejected_count == 1
    assert item.broken_media_references


def test_malformed_and_duplicate_records_have_explicit_reasons(
    dataset_dir: Path,
) -> None:
    rows = _valid_rows()["financial_profiles.csv"]
    rows.append(dict(rows[0]))
    rows.append({**rows[0], "user_id": "user_c", "current_available_balance": "bad"})
    _write(
        dataset_dir / "financial_profiles.csv",
        CSV_SCHEMAS["financial_profiles.csv"],
        rows,
    )
    report = audit_dataset(dataset_dir)
    item = next(x for x in report.files if x.filename == "financial_profiles.csv")
    assert item.raw_count == 4 and item.valid_count == 2 and item.rejected_count == 2
    assert item.duplicate_ids and item.exact_duplicates
    assert any("plain base-10 decimal" in reason for reason in item.rejected_records)
    rendered = render_markdown(report)
    assert "Rejected records:" in rendered
    assert str(dataset_dir.resolve()) not in rendered
    assert "Dataset: `" + dataset_dir.name + "/`" in rendered
