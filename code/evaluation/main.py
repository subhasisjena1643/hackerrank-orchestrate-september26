"""Phase 13 public-sample policy tournament.

Solved columns are loaded only after the immutable policy manifest is written.
Prediction code receives request contexts and never receives sample labels.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import date, datetime, timezone
from decimal import Decimal
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from typing import Any, Iterable, Mapping, Sequence

CODE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = CODE_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from buy_or_wait.data_loader import build_request_contexts, load_dataset
from buy_or_wait.domain import (
    Currency,
    Direction,
    EventType,
    Flexibility,
    RequestContext,
    SampleOutput,
)
from buy_or_wait.evidence.cache import EvidenceCache
from buy_or_wait.evidence.client import create_live_evidence_client
from buy_or_wait.finance.recurrence import (
    Cadence,
    RecurrenceObservation,
    RecurringSeries,
    SeriesKey,
    project_recurrence_series,
)
from buy_or_wait.finance.recurrence_policies import (
    BASELINE_POLICY,
    POLICY_GRID,
    AmountRounding,
    CadenceAnchor,
    ForecastPolicy,
    estimate_amount,
)
from buy_or_wait.pipeline import (
    Phase9PipelineResult,
    complete_production_pipeline,
    run_phase9_context,
)
from buy_or_wait.telemetry import TelemetryLedger
from buy_or_wait.validation.output import decision_to_output_row

DEFAULT_DATASET_DIR = REPO_ROOT / "dataset"
ARTIFACT_DIR = CODE_DIR / "evaluation"
SAMPLE_METRICS_PATH = ARTIFACT_DIR / "sample_metrics.json"
HYPOTHESIS_RESULTS_PATH = ARTIFACT_DIR / "hypothesis_results.json"
SELECTED_POLICY_PATH = ARTIFACT_DIR / "selected_policy.json"
MANIFEST_PATH = ARTIFACT_DIR / "final_run_manifest.json"
TOLERANCE = Decimal("0.01")
COMPOSITE_VERSION = "phase13-composite-v1"
COMPOSITE_WEIGHTS = {
    "safe_amount_normalized_accuracy": Decimal("0.35"),
    "affordability_status_accuracy": Decimal("0.15"),
    "recommended_payment_method_accuracy": Decimal("0.15"),
    "payment_plan_accuracy": Decimal("0.15"),
    "earliest_date_accuracy": Decimal("0.10"),
    "spending_changes_accuracy": Decimal("0.10"),
}


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def materialize_manifest() -> tuple[dict[str, Any], str]:
    """Freeze the full policy grid before any solved sample field is loaded."""
    policies = [policy.canonical_dict() for policy in POLICY_GRID]
    grid_hash = sha256_json(policies)
    manifest = {
        "schema_version": "phase13-policy-grid-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "candidate_count": len(policies),
        "grid_hash_sha256": grid_hash,
        "policies": policies,
        "composite": {
            "version": COMPOSITE_VERSION,
            "formula": "weighted sum; safe accuracy = 1 - min(1, mean normalized absolute error)",
            "weights": {key: float(value) for key, value in COMPOSITE_WEIGHTS.items()},
        },
        "candidate_addition_rule": (
            "A new general hypothesis requires a fresh manifest and complete rerun."
        ),
        "general_evidence_hypotheses": [
            {
                "id": "evidence-message-single-record-v1",
                "rationale": (
                    "Validate ambiguous provider evidence one message at a time to "
                    "reduce cross-record schema and grounding failures."
                ),
            },
            {
                "id": "evidence-explicit-temporary-payroll-v1",
                "rationale": (
                    "Parse explicit currency/amount temporary payroll and next-salary "
                    "reductions deterministically as one-cycle amendments when the "
                    "source explicitly names the affected cycle."
                ),
            },
            {
                "id": "evidence-confirmed-base-payroll-v1",
                "rationale": (
                    "Parse an explicitly confirmed base salary deterministically as "
                    "a recurring amendment while never counting an unapproved or "
                    "unearned commission mentioned in the same message."
                ),
            },
            {
                "id": "evidence-explicit-first-salary-v1",
                "rationale": (
                    "Parse an explicit first-salary amount and confirmed credit date "
                    "deterministically without creating any unsupported event row."
                ),
            },
            {
                "id": "evidence-explicit-internal-transfer-v1",
                "rationale": (
                    "Classify an explicitly matched debit and credit between two "
                    "same-owner accounts as a non-cash internal transfer."
                ),
            },
            {
                "id": "evidence-explicit-linked-proceeds-settlement-v1",
                "rationale": (
                    "Treat linked proceeds as settled only when the source explicitly "
                    "states they reached the account and the claim is closed."
                ),
            },
        ],
    }
    write_json(MANIFEST_PATH, manifest)
    return manifest, grid_hash


def load_sample_inputs(dataset_dir: Path) -> tuple[RequestContext, ...]:
    """Load sample inputs with solved columns deliberately unavailable."""
    inputs = load_dataset(dataset_dir, include_sample_outputs=False)
    sample_only = replace(
        inputs,
        evaluation_requests=inputs.sample_requests,
        sample_requests=(),
        sample_outputs=(),
    )
    return build_request_contexts(sample_only)


def load_expected_outputs(dataset_dir: Path) -> tuple[SampleOutput, ...]:
    return load_dataset(dataset_dir, include_sample_outputs=True).sample_outputs


def prepare_evidence(
    contexts: Sequence[RequestContext], dataset_dir: Path
) -> tuple[Phase9PipelineResult, ...]:
    """Run policy-independent production evidence/lifecycle processing once."""
    client = create_live_evidence_client()
    cache_root = ARTIFACT_DIR / ".evidence_cache"
    cache = EvidenceCache(cache_root / "messages")
    telemetry = TelemetryLedger(ARTIFACT_DIR / ".telemetry" / "phase13.jsonl")
    return tuple(
        run_phase9_context(
            context,
            dataset_dir=dataset_dir,
            policy=BASELINE_POLICY,
            client=client,
            message_cache=cache,
            image_cache_dir=cache_root / "images",
            telemetry=telemetry,
            run_id="phase13-sample-evidence",
            message_batch_size=1,
        )
        for context in contexts
    )


def _expected_row(output: SampleOutput) -> dict[str, str]:
    return {
        "amount_safe_to_pay": format(output.amount_safe_to_pay, "f"),
        "affordability_status": output.affordability_status.value,
        "recommended_payment_method": output.recommended_payment_method.value,
        "payment_plan": output.payment_plan,
        "earliest_date_for_full_payment": (
            output.earliest_date_for_full_payment.isoformat()
            if output.earliest_date_for_full_payment
            else ""
        ),
        "spending_changes_needed": output.spending_changes_needed,
    }


def _provenance(phase9: Phase9PipelineResult) -> dict[str, object]:
    context = phase9.context
    return {
        "event_ids": sorted(event.event_id for event in context.financial_events),
        "message_ids": sorted(message.message_id for message in context.messages),
        "image_ids": sorted(image.image_id for image in context.images),
        "resolved_event_count": len(phase9.resolved_events),
        "message_fact_count": len(phase9.message_extraction.evidence),
        "image_resolution_count": len(phase9.image_resolutions),
        "content_omitted": True,
    }


def evaluate_policy(
    policy: ForecastPolicy,
    prepared: Sequence[Phase9PipelineResult],
    expected: Mapping[str, SampleOutput],
) -> tuple[list[dict[str, Any]], list[str], float]:
    started = perf_counter()
    diffs: list[dict[str, Any]] = []
    failures: list[str] = []
    scored_fields = (
        "affordability_status",
        "recommended_payment_method",
        "payment_plan",
        "earliest_date_for_full_payment",
        "spending_changes_needed",
    )
    for phase9 in prepared:
        request = phase9.context.request
        label = expected[request.request_id]
        wanted = _expected_row(label)
        try:
            result = complete_production_pipeline(phase9, policy=policy)
            actual = decision_to_output_row(result.decision)
            safe_error = abs(
                Decimal(actual["amount_safe_to_pay"]) - label.amount_safe_to_pay
            )
            fields = {name: actual[name] == wanted[name] for name in scored_fields}
            diffs.append(
                {
                    "request_id": request.request_id,
                    "requested_amount": format(request.requested_amount, "f"),
                    "expected": wanted,
                    "actual": {key: actual[key] for key in wanted},
                    "safe_amount_absolute_error": format(safe_error, "f"),
                    "safe_amount_normalized_error": float(
                        safe_error / request.requested_amount
                    ),
                    "safe_amount_exact": safe_error == 0,
                    "safe_amount_tolerance": safe_error <= TOLERANCE,
                    "field_exact": fields,
                    "invariant_pass": True,
                    "provenance": _provenance(phase9),
                }
            )
        except Exception as error:  # fail-closed tournament boundary
            reason = f"{request.request_id}: {type(error).__name__}: {error}"
            failures.append(reason)
            diffs.append(
                {
                    "request_id": request.request_id,
                    "requested_amount": format(request.requested_amount, "f"),
                    "expected": wanted,
                    "actual": None,
                    "safe_amount_absolute_error": format(request.requested_amount, "f"),
                    "safe_amount_normalized_error": 1.0,
                    "safe_amount_exact": False,
                    "safe_amount_tolerance": False,
                    "field_exact": {name: False for name in scored_fields},
                    "invariant_pass": False,
                    "failure": reason,
                    "provenance": _provenance(phase9),
                }
            )
    return diffs, failures, perf_counter() - started


def summarize(diffs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    count = len(diffs)
    safe_errors = [Decimal(str(item["safe_amount_absolute_error"])) for item in diffs]
    normalized = [Decimal(str(item["safe_amount_normalized_error"])) for item in diffs]

    def accuracy(field: str) -> float:
        return sum(bool(item["field_exact"][field]) for item in diffs) / count

    mean_normalized = sum(normalized, Decimal(0)) / Decimal(count)
    components = {
        "safe_amount_normalized_accuracy": float(
            max(Decimal(0), Decimal(1) - mean_normalized)
        ),
        "affordability_status_accuracy": accuracy("affordability_status"),
        "recommended_payment_method_accuracy": accuracy("recommended_payment_method"),
        "payment_plan_accuracy": accuracy("payment_plan"),
        "earliest_date_accuracy": accuracy("earliest_date_for_full_payment"),
        "spending_changes_accuracy": accuracy("spending_changes_needed"),
    }
    score = sum(
        Decimal(str(components[key])) * weight
        for key, weight in COMPOSITE_WEIGHTS.items()
    )
    return {
        "request_count": count,
        "safe_amount": {
            "exact_matches": sum(bool(item["safe_amount_exact"]) for item in diffs),
            "exact_accuracy": sum(bool(item["safe_amount_exact"]) for item in diffs)
            / count,
            "tolerance": format(TOLERANCE, "f"),
            "tolerance_matches": sum(
                bool(item["safe_amount_tolerance"]) for item in diffs
            ),
            "tolerance_accuracy": sum(
                bool(item["safe_amount_tolerance"]) for item in diffs
            )
            / count,
            "mae": float(sum(safe_errors, Decimal(0)) / Decimal(count)),
            "mean_normalized_error": float(mean_normalized),
        },
        "affordability_status": {
            "exact_accuracy": components["affordability_status_accuracy"]
        },
        "recommended_payment_method": {
            "exact_accuracy": components["recommended_payment_method_accuracy"]
        },
        "payment_plan": {"exact_accuracy": components["payment_plan_accuracy"]},
        "earliest_date_for_full_payment": {
            "exact_accuracy": components["earliest_date_accuracy"]
        },
        "spending_changes_needed": {
            "exact_accuracy": components["spending_changes_accuracy"]
        },
        "invariant_pass_rate": sum(bool(item["invariant_pass"]) for item in diffs)
        / count,
        "composite": {
            "version": COMPOSITE_VERSION,
            "score": float(score),
            "components": components,
            "weights": {key: float(value) for key, value in COMPOSITE_WEIGHTS.items()},
        },
    }


def _family(policy: ForecastPolicy) -> str:
    config = policy.canonical_dict()
    return "/".join(
        config[key]
        for key in (
            "amount_estimator",
            "history_window",
            "aggregation",
            "cadence_anchor",
        )
    )


def _robustness(policy: ForecastPolicy) -> dict[str, Any]:
    """Deterministic amount/date boundary checks independent of sample labels."""
    cases: dict[str, bool] = {}
    estimates = [
        estimate_amount(
            (Decimal("99.99") + delta, Decimal("100.00"), Decimal("100.01") - delta),
            policy,
        )
        for delta in (Decimal("-0.01"), Decimal("0"), Decimal("0.01"))
    ]
    cases["amount_plus_minus_0_01_finite"] = all(
        value.is_finite() for value in estimates
    )
    cases[
        "rounding_stable"
    ] = policy.amount_rounding is not AmountRounding.HALF_UP_2 or all(
        value == value.quantize(Decimal("0.01")) for value in estimates
    )
    key = SeriesKey(
        "synthetic-user",
        Direction.DEBIT,
        "utilities",
        EventType.EXPENSE,
        Currency.USD,
        Flexibility.FIXED,
        "category:utilities",
    )
    observations = (
        RecurrenceObservation(
            "synthetic-1", date(2026, 1, 31), Decimal("100"), "synthetic", ()
        ),
        RecurrenceObservation(
            "synthetic-2", date(2026, 2, 28), Decimal("100"), "synthetic", ()
        ),
        RecurrenceObservation(
            "synthetic-3", date(2026, 3, 31), Decimal("100"), "synthetic", ()
        ),
    )
    series = RecurringSeries(
        key,
        Cadence.MONTHLY,
        observations,
        Decimal("100"),
        date(2026, 3, 31),
        31,
        True,
        30,
        False,
    )
    start = date(2026, 4, 1)
    boundaries = [date(2026, 6, 29), date(2026, 6, 30), date(2026, 7, 1)]
    projections = [
        project_recurrence_series(series, start_date=start, end_date=end, policy=policy)
        for end in boundaries
    ]
    cases["plus_minus_1_day_boundary_monotonic"] = (
        len(projections[0]) <= len(projections[1]) <= len(projections[2])
    )
    cases["day_90_inclusive"] = all(
        item.flow_date <= boundaries[1] for item in projections[1]
    )
    cases["month_end_valid"] = all(
        1 <= item.flow_date.day <= 31 for item in projections[1]
    )
    if policy.cadence_anchor is CadenceAnchor.CALENDAR_DAY_OF_MONTH:
        cases["month_end_anchor"] = [item.flow_date for item in projections[1]] == [
            date(2026, 4, 30),
            date(2026, 5, 31),
            date(2026, 6, 30),
        ]
    else:
        cases["month_end_anchor"] = True
    return {"passed": all(cases.values()), "cases": cases}


def _loo(
    survivors: Sequence[dict[str, Any]], selected: dict[str, Any]
) -> dict[str, Any]:
    request_ids = [item["request_id"] for item in selected["diffs"]]
    folds, exact_wins, family_wins = [], 0, 0
    for request_id in request_ids:
        ranked = []
        for candidate in survivors:
            training = [
                item for item in candidate["diffs"] if item["request_id"] != request_id
            ]
            score = summarize(training)["composite"]["score"]
            ranked.append((score, candidate["canonical_config"], candidate))
        winner = sorted(
            ranked, key=lambda item: (-item[0], len(item[1]), item[2]["policy_hash"])
        )[0][2]
        held_metrics = summarize(
            [item for item in winner["diffs"] if item["request_id"] == request_id]
        )
        exact_wins += winner["policy_hash"] == selected["policy_hash"]
        family_wins += winner["policy_family"] == selected["policy_family"]
        folds.append(
            {
                "withheld_request_id": request_id,
                "winner_policy_hash": winner["policy_hash"],
                "winner_policy_family": winner["policy_family"],
                "held_out_metrics": held_metrics,
            }
        )
    count = len(folds)
    return {
        "fold_count": count,
        "same_exact_policy_wins": exact_wins,
        "same_exact_policy_rate": exact_wins / count,
        "same_policy_family_wins": family_wins,
        "same_policy_family_rate": family_wins / count,
        "folds": folds,
    }


def _root_causes(diffs: Iterable[Mapping[str, Any]]) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {}
    for item in diffs:
        fields = item["field_exact"]
        causes = []
        if not item["safe_amount_tolerance"]:
            causes.append("forecast_safe_amount")
        if (
            not fields["affordability_status"]
            or not fields["recommended_payment_method"]
        ):
            causes.append("decision_classification")
        if not fields["payment_plan"] or not fields["earliest_date_for_full_payment"]:
            causes.append("payment_timing_or_plan")
        if not fields["spending_changes_needed"]:
            causes.append("spending_change_selection")
        if causes:
            groups.setdefault("+".join(causes), []).append(str(item["request_id"]))
    return groups


def run_tournament(
    dataset_dir: Path, *, test_count: int | None = None
) -> dict[str, Any]:
    manifest, grid_hash = materialize_manifest()
    contexts = load_sample_inputs(dataset_dir)
    prepared = prepare_evidence(contexts, dataset_dir)
    labels = load_expected_outputs(dataset_dir)  # first solved-column read
    expected = {item.request_id: item for item in labels}
    if set(expected) != {item.request.request_id for item in contexts}:
        raise RuntimeError("sample input/output request IDs do not match")

    results: list[dict[str, Any]] = []
    for policy in POLICY_GRID:
        diffs, failures, runtime = evaluate_policy(policy, prepared, expected)
        metrics, robustness = summarize(diffs), _robustness(policy)
        rejection = None
        if failures:
            rejection = (
                "invariant/unsafe replay/invalid option or change/evidence failure: "
                + failures[0]
            )
        elif not robustness["passed"]:
            rejection = "boundary perturbation failure"
        results.append(
            {
                "policy": policy.canonical_dict(),
                "canonical_config": policy.canonical_json(),
                "policy_hash": policy.sha256,
                "policy_family": _family(policy),
                "metrics": metrics,
                "runtime_seconds": runtime,
                "robustness": robustness,
                "rejection_reason": rejection,
                "diffs": diffs,
            }
        )
    survivors = [item for item in results if item["rejection_reason"] is None]
    if not survivors:
        raise RuntimeError("all predeclared policies were rejected")
    survivors.sort(
        key=lambda item: (
            -item["metrics"]["composite"]["score"],
            len(item["canonical_config"]),
            item["policy_hash"],
        )
    )
    selected = survivors[0]
    loo = _loo(survivors, selected)
    for item in results:
        item["leave_one_out_sensitivity"] = {
            "winner_fold_count": sum(
                fold["winner_policy_hash"] == item["policy_hash"]
                for fold in loo["folds"]
            )
        }
    baseline = next(
        item for item in results if item["policy_hash"] == BASELINE_POLICY.sha256
    )
    root_causes = _root_causes(selected["diffs"])
    sample_metrics = {
        "schema_version": "phase13-sample-metrics-v1",
        "grid_hash_sha256": grid_hash,
        "selected_policy_hash": selected["policy_hash"],
        "metrics": selected["metrics"],
        "per_request_diffs": selected["diffs"],
        "remaining_mismatches_by_root_cause": root_causes,
    }
    hypothesis_results = {
        "schema_version": "phase13-hypothesis-results-v1",
        "grid_manifest_schema": manifest["schema_version"],
        "grid_hash_sha256": grid_hash,
        "candidate_count": len(results),
        "composite": manifest["composite"],
        "test_suite": {"passed": test_count is not None, "test_count": test_count},
        "baseline_policy_hash": baseline["policy_hash"],
        "selected_policy_hash": selected["policy_hash"],
        "leave_one_out": loo,
        "policies": results,
        "top_five_policy_hashes": [item["policy_hash"] for item in survivors[:5]],
    }
    stability = {
        "leave_one_out_same_policy_rate": loo["same_exact_policy_rate"],
        "leave_one_out_same_family_rate": loo["same_policy_family_rate"],
        "boundary_perturbations_passed": selected["robustness"]["passed"],
        "boundary_cases": selected["robustness"]["cases"],
    }
    selected_policy = {
        "schema_version": "selected-global-policy-v1",
        "policy": selected["policy"],
        "composite_score": selected["metrics"]["composite"],
        "rationale": (
            "Highest non-rejected predeclared composite; deterministic simplicity "
            "tie-break applied within exact score ties. No request-specific or "
            "label-neighbour rule is used."
        ),
        "stability_summary": stability,
        "sha256": selected["policy_hash"],
    }
    write_json(SAMPLE_METRICS_PATH, sample_metrics)
    write_json(HYPOTHESIS_RESULTS_PATH, hypothesis_results)
    write_json(SELECTED_POLICY_PATH, selected_policy)
    return {
        "test_count": test_count,
        "grid_hash": grid_hash,
        "baseline": baseline,
        "selected": selected,
        "top_five": survivors[:5],
        "leave_one_out": loo,
        "root_causes": root_causes,
    }


def _repo_relative_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run frozen Phase 13 sample tournament."
    )
    parser.add_argument(
        "--dataset", type=_repo_relative_path, default=DEFAULT_DATASET_DIR
    )
    parser.add_argument("--tests-passed", type=int, default=None)
    parser.add_argument("--manifest-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.manifest_only:
        manifest, _ = materialize_manifest()
        print(
            json.dumps(
                {
                    "candidate_count": manifest["candidate_count"],
                    "grid_hash": manifest["grid_hash_sha256"],
                }
            )
        )
        return 0
    summary = run_tournament(args.dataset, test_count=args.tests_passed)
    print(
        json.dumps(
            {
                "grid_hash": summary["grid_hash"],
                "selected_policy_hash": summary["selected"]["policy_hash"],
                "baseline_score": summary["baseline"]["metrics"]["composite"]["score"],
                "selected_score": summary["selected"]["metrics"]["composite"]["score"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
