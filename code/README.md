# Buy or Wait? implementation

This directory contains the submission-owned Python package. Phases 0–9 provide the
contract and repository scaffold; typed input contracts and strict CSV loading;
exact money, date, and currency conversion; provider/cache/telemetry boundaries;
bounded multilingual message evidence; strict image OCR/vision consensus with one
targeted adjudication pass; event-lifecycle resolution; recurrence and future
cash-flow construction; and the inclusive 90-day baseline safe-capacity forecast.

The connected Phase 9 library boundary is
`buy_or_wait.pipeline.run_phase9_dataset_request`. It deliberately stops after the
baseline forecast: spending-change search, payment-plan generation, recommendation
selection, explanations, output generation, and full-dataset execution belong to
later phases and are not wired into `main.py` yet.

## Setup

Create and activate a Python 3.11+ virtual environment using the command appropriate
for your operating system, then install the pinned dependencies:

```text
python -m pip install -r code/requirements.txt
```

Copy `code/.env.example` to the repository-root `.env` (or `code/.env`) and supply
the selected evidence models and provider credential. The evidence client discovers
these ignored files without overriding exported process variables. Never commit a
local `.env` file.

Message extraction uses the exact versioned runtime prompts under `code/prompts/`,
strict structured output, deterministic safe status handling, one schema-repair
attempt, source grounding, content-addressed caching, and token telemetry. It fails
closed when live configuration is unavailable.

Run the 16 linked-image extractions with `python code/scripts/precompute_evidence.py`.
It prints only IDs, channel/status fields, currency, and confidence. By default every
image is reprocessed; pass `--cache evidence_cache/images` only when an explicitly
cached run is desired. Local Tesseract OCR is optional; when its executable is not
available, the declared consensus policy records that channel as unavailable and
requires the configured vision adjudication path rather than inventing an OCR value.

## Run

The canonical command, run from the repository root, is:

```text
python code/main.py --dataset dataset --output output.csv --strict
```

CLI paths are resolved with `pathlib` relative to the code's repository location,
so execution does not depend on the shell's current directory.

The entire `dataset/` directory is read-only. Runtime code must never create, edit,
replace, or delete files inside it. Final predictions will be written to the
repository-root `output.csv` in a later phase.

## Phase 0–9 verification

Run the implemented test scope, including the mocked end-to-end Phase 9 pipeline,
without making a live provider call:

```text
pytest -q
pytest -q code/tests/test_phase9_integration.py
```

The end-to-end fixture asserts typed evidence, lifecycle outcomes, recurrence at the
inclusive horizon boundary, baseline forecast values, safe-capacity replay, rejected
input isolation, and deterministic repeatability. Live extraction is intentionally a
separate explicit command so normal unit and CI runs cannot incur provider charges.

## Inspect inputs

Run schema and distribution diagnostics without reading solved sample output values:

```text
python code/scripts/inspect_dataset.py --dataset dataset
```

Add `--include-sample-summary` only when sample-label distributions are explicitly
needed. The profiler never writes to `dataset/`.

Generate the full read-only data-quality report with:

```text
python code/scripts/audit_data_quality.py --dataset dataset
```

The report is written to `code/evaluation/data_quality_report.md`; raw files remain
untouched. Invalid records are reported with explicit file/row reasons and cause a
non-zero exit status.
