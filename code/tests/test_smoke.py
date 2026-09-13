"""Phase 1 smoke tests."""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path


CODE_DIR = Path(__file__).resolve().parents[1]


def test_package_modules_import() -> None:
    sys.path.insert(0, str(CODE_DIR))
    modules = (
        "buy_or_wait",
        "buy_or_wait.config",
        "buy_or_wait.domain",
        "buy_or_wait.data_loader",
        "buy_or_wait.money",
        "buy_or_wait.pipeline",
        "buy_or_wait.explanation",
        "buy_or_wait.telemetry",
        "buy_or_wait.evidence.schemas",
        "buy_or_wait.evidence.messages",
        "buy_or_wait.evidence.images",
        "buy_or_wait.evidence.image_consensus",
        "buy_or_wait.evidence.guardrails",
        "buy_or_wait.evidence.cache",
        "buy_or_wait.finance.lifecycle",
        "buy_or_wait.finance.recurrence",
        "buy_or_wait.finance.recurrence_policies",
        "buy_or_wait.finance.cashflows",
        "buy_or_wait.finance.forecast",
        "buy_or_wait.finance.spending_changes",
        "buy_or_wait.finance.plans",
        "buy_or_wait.finance.decision",
        "buy_or_wait.validation.inputs",
        "buy_or_wait.validation.plan",
        "buy_or_wait.validation.output",
    )
    for module in modules:
        importlib.import_module(module)


def test_main_help_succeeds(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, str(CODE_DIR / "main.py"), "--help"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_readme_documents_the_current_phase9_boundary() -> None:
    readme = (CODE_DIR / "README.md").read_text(encoding="utf-8")
    assert "buy_or_wait.pipeline.run_phase9_dataset_request" in readme
    assert "deliberately stops after the\nbaseline forecast" in readme
    assert "not wired into `main.py` yet" in readme
