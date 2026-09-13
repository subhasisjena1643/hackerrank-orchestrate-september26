"""Phase 12 deterministic explanation and strict output-contract coverage."""

from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path
import sys

import pytest

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from buy_or_wait.domain import (
    AffordabilityStatus,
    Currency,
    Decision,
    Direction,
    EventType,
    Flexibility,
    Payment,
    PaymentMethod,
    PaymentOption,
    Profile,
    Request,
    RequestType,
    SpendingChange,
    SpendingChangeType,
)
from buy_or_wait.explanation import (
    ExplanationError,
    build_decision_explanation,
)
from buy_or_wait.finance.cashflows import ProjectedCashFlow
from buy_or_wait.finance.forecast import calculate_baseline_forecast
from buy_or_wait.finance.recurrence import SeriesKey
from buy_or_wait.finance.recurrence_policies import BASELINE_POLICY
from buy_or_wait.finance.spending_changes import (
    SpendingChangeSource,
    VerifiedSpendingChange,
)
from buy_or_wait.validation.output import (
    OUTPUT_COLUMNS,
    OutputValidationError,
    decision_to_output_row,
    parse_output_row,
    validate_output_csv,
    validate_output_rows,
    write_output_csv_atomic,
)
from buy_or_wait.validation.plan import (
    DecisionValidationContext,
    DecisionValidationError,
    validate_decision_plan,
)


TODAY = date(2026, 9, 1)


def request(*, request_type=RequestType.OTHER, partial=True, deadline=None):
    return Request(
        "request_1",
        "user_1",
        TODAY,
        request_type,
        Decimal("100"),
        deadline or TODAY + timedelta(days=90),
        partial,
        "Can I afford this?",
    )


def profile(*, balance="250", methods=None, months=4):
    return Profile(
        "user_1",
        Currency.USD,
        Decimal(balance),
        Decimal("20"),
        (),
        (),
        ("fun",),
        ("fun",),
        tuple(
            methods
            or (
                PaymentMethod.FULL_PAYMENT,
                PaymentMethod.PARTIAL_PAYMENT,
                PaymentMethod.INSTALLMENTS,
            )
        ),
        months,
    )


def context(req=None, prof=None, *, options=(), flows=()):
    req, prof = req or request(), prof or profile()
    baseline = calculate_baseline_forecast(
        current_available_balance=prof.current_available_balance,
        minimum_balance_to_keep=prof.minimum_balance_to_keep,
        request_date=req.request_date,
        requested_amount=req.requested_amount,
        cash_flows=flows,
        policy=BASELINE_POLICY,
    )
    return DecisionValidationContext(
        req,
        prof,
        baseline,
        tuple(options),
        tuple(flows),
        BASELINE_POLICY,
    )


def salary_credit(when, amount="50"):
    key = SeriesKey(
        "user_1",
        Direction.CREDIT,
        "salary",
        EventType.INCOME,
        Currency.USD,
        Flexibility.FIXED,
        "salary",
    )
    return ProjectedCashFlow(
        "salary",
        when,
        Decimal(amount),
        Direction.CREDIT,
        "salary",
        "Confirmed salary",
        (),
        (),
        (),
        ("test",),
        False,
        key,
    )


def full_decision(ctx=None):
    ctx = ctx or context()
    return Decision(
        ctx.request.request_id,
        ctx.baseline.amount_safe_to_pay,
        AffordabilityStatus.AFFORDABLE_NOW,
        PaymentMethod.FULL_PAYMENT,
        (Payment(TODAY, Decimal("100")),),
        ctx.baseline.earliest_date_for_full_payment,
        (),
        (
            "Pay USD 100 today. This keeps the USD 20 minimum protected over the "
            "next 90 days."
        ),
    )


def installment_option():
    return PaymentOption(
        "payment_option_2",
        "request_1",
        PaymentMethod.INSTALLMENTS,
        Decimal("55.00"),
        2,
        TODAY,
        30,
        Decimal("10"),
        Decimal("110.00"),
    )


def test_full_now_explanation_and_exact_serialization_columns():
    ctx = context()
    decision = replace(full_decision(ctx), decision_explanation="")
    explanation = build_decision_explanation(
        decision, request=ctx.request, profile=ctx.profile
    )
    decision = replace(decision, decision_explanation=explanation)
    row = decision_to_output_row(decision)
    assert tuple(row) == OUTPUT_COLUMNS
    assert row["payment_plan"] == "2026-09-01:100"
    assert "90 days" in explanation and "USD 20" in explanation
    assert parse_output_row(row) == decision
    assert validate_decision_plan(decision, ctx).replay is not None


def test_partial_wait_installment_and_fallback_truth_table():
    payday = TODAY + timedelta(days=10)
    partial_ctx = context(
        prof=profile(balance="70", methods=(PaymentMethod.PARTIAL_PAYMENT,)),
        flows=(salary_credit(payday),),
    )
    partial = Decision(
        "request_1",
        Decimal("50"),
        AffordabilityStatus.AFFORDABLE_WITH_PLAN,
        PaymentMethod.PARTIAL_PAYMENT,
        (Payment(TODAY, Decimal("50")), Payment(payday, Decimal("50"))),
        payday,
        (),
        "Pay USD 50 today and USD 50 today while protecting USD 20.",
    )
    validate_decision_plan(partial, partial_ctx)
    partial_text = build_decision_explanation(
        partial, request=partial_ctx.request, profile=partial_ctx.profile
    )
    assert "USD 50 today" in partial_text and payday.isoformat() in partial_text

    wait_ctx = context(
        prof=profile(balance="20", methods=(PaymentMethod.FULL_PAYMENT,)),
        flows=(salary_credit(payday, "100"),),
    )
    wait = Decision(
        "request_1",
        Decimal("0"),
        AffordabilityStatus.AFFORDABLE_LATER,
        PaymentMethod.WAIT,
        (Payment(payday, Decimal("100")),),
        payday,
        (),
        "Wait for the confirmed salary while protecting USD 20.",
    )
    validate_decision_plan(wait, wait_ctx)
    wait_text = build_decision_explanation(
        wait, request=wait_ctx.request, profile=wait_ctx.profile
    )
    assert payday.isoformat() in wait_text and "Paying earlier" in wait_text

    option = installment_option()
    installment_ctx = context(
        prof=profile(methods=(PaymentMethod.INSTALLMENTS,)), options=(option,)
    )
    installment = Decision(
        "request_1",
        Decimal("100"),
        AffordabilityStatus.AFFORDABLE_WITH_PLAN,
        PaymentMethod.INSTALLMENTS,
        (
            Payment(TODAY, Decimal("55.00")),
            Payment(TODAY + timedelta(days=30), Decimal("55.00")),
        ),
        TODAY,
        (),
        "Use two installments while protecting USD 20.",
    )
    checked = validate_decision_plan(installment, installment_ctx)
    assert checked.selected_option == option
    assert decision_to_output_row(installment)["payment_plan"].endswith(":55.00")

    fallback_ctx = context(
        prof=profile(balance="20", methods=(PaymentMethod.INSTALLMENTS,))
    )
    fallback = Decision(
        "request_1",
        Decimal("0"),
        AffordabilityStatus.NOT_AFFORDABLE,
        PaymentMethod.NOT_RECOMMENDED,
        (),
        None,
        (),
        "No eligible option protects the minimum by the deadline.",
    )
    assert validate_decision_plan(fallback, fallback_ctx).replay is None


VALID_STATUS_METHODS = {
    (AffordabilityStatus.AFFORDABLE_NOW, PaymentMethod.FULL_PAYMENT),
    (AffordabilityStatus.AFFORDABLE_WITH_PLAN, PaymentMethod.FULL_PAYMENT),
    (AffordabilityStatus.AFFORDABLE_WITH_PLAN, PaymentMethod.PARTIAL_PAYMENT),
    (AffordabilityStatus.AFFORDABLE_WITH_PLAN, PaymentMethod.INSTALLMENTS),
    (AffordabilityStatus.AFFORDABLE_LATER, PaymentMethod.WAIT),
    (AffordabilityStatus.NOT_AFFORDABLE, PaymentMethod.NOT_RECOMMENDED),
}


@pytest.mark.parametrize(
    ("status", "method"),
    [
        (status, method)
        for status in AffordabilityStatus
        for method in PaymentMethod
        if (status, method) not in VALID_STATUS_METHODS
    ],
)
def test_invalid_status_method_truth_table(status, method):
    ctx = context()
    with pytest.raises(DecisionValidationError, match="status/payment-method"):
        validate_decision_plan(
            replace(
                full_decision(ctx),
                affordability_status=status,
                recommended_payment_method=method,
            ),
            ctx,
        )


def test_baseline_bounds_dates_none_plan_and_unsafe_replay_are_rejected():
    ctx = context()
    for changed in (
        replace(full_decision(ctx), amount_safe_to_pay=Decimal("101")),
        replace(full_decision(ctx), amount_safe_to_pay=Decimal("99")),
        replace(full_decision(ctx), earliest_date_for_full_payment=None),
        replace(full_decision(ctx), payment_plan=()),
        replace(full_decision(ctx), payment_plan=(Payment(TODAY, Decimal("231")),)),
    ):
        with pytest.raises(DecisionValidationError):
            validate_decision_plan(changed, ctx)

    stale = replace(
        ctx,
        baseline=replace(ctx.baseline, amount_safe_to_pay=Decimal("99")),
    )
    with pytest.raises(DecisionValidationError, match="independent recomputation"):
        validate_decision_plan(
            replace(full_decision(ctx), amount_safe_to_pay=Decimal("99")), stale
        )


def test_partial_structure_installment_match_and_wait_preference_rejected():
    ctx = context()
    wrong_partial = replace(
        full_decision(ctx),
        affordability_status=AffordabilityStatus.AFFORDABLE_WITH_PLAN,
        recommended_payment_method=PaymentMethod.PARTIAL_PAYMENT,
        payment_plan=(Payment(TODAY, Decimal("40")), Payment(TODAY, Decimal("60"))),
    )
    with pytest.raises(DecisionValidationError):
        validate_decision_plan(wrong_partial, ctx)

    wrong_installment = replace(
        wrong_partial,
        recommended_payment_method=PaymentMethod.INSTALLMENTS,
        payment_plan=(
            Payment(TODAY, Decimal("50")),
            Payment(TODAY + timedelta(days=30), Decimal("50")),
        ),
    )
    with pytest.raises(DecisionValidationError, match="supplied option"):
        validate_decision_plan(wrong_installment, ctx)

    wait = replace(
        full_decision(ctx),
        affordability_status=AffordabilityStatus.AFFORDABLE_LATER,
        recommended_payment_method=PaymentMethod.WAIT,
    )
    with pytest.raises(DecisionValidationError):
        validate_decision_plan(wait, ctx)


def test_explanations_cover_installments_changes_not_affordable_and_investment():
    opt = installment_option()
    ctx = context(options=(opt,))
    installment = Decision(
        "request_1",
        Decimal("100"),
        AffordabilityStatus.AFFORDABLE_WITH_PLAN,
        PaymentMethod.INSTALLMENTS,
        (
            Payment(TODAY, Decimal("55.00")),
            Payment(TODAY + timedelta(days=30), Decimal("55.00")),
        ),
        TODAY,
        (),
        "",
    )
    text = build_decision_explanation(
        installment,
        request=ctx.request,
        profile=ctx.profile,
        selected_option=opt,
    )
    assert "2 installments of USD 55.00" in text and "2026-09-01" in text

    change = SpendingChange(SpendingChangeType.STOP, "event_3")
    source = SpendingChangeSource(
        "event_3",
        "Video subscription",
        "fun",
        Flexibility.STOPPABLE,
        Decimal("20"),
        None,
        ("event_1", "event_2", "event_3"),
        (),
        (),
    )
    verified = VerifiedSpendingChange(change, source, TODAY, Decimal("20"))
    changed = replace(
        full_decision(ctx),
        affordability_status=AffordabilityStatus.AFFORDABLE_WITH_PLAN,
        spending_changes_needed=(change,),
    )
    assert "stop Video subscription" in build_decision_explanation(
        changed,
        request=ctx.request,
        profile=ctx.profile,
        verified_changes=(verified,),
    )
    with pytest.raises(ExplanationError, match="provenance"):
        build_decision_explanation(changed, request=ctx.request, profile=ctx.profile)

    fallback_ctx = context(prof=profile(balance="20"))
    fallback = Decision(
        "request_1",
        Decimal("0"),
        AffordabilityStatus.NOT_AFFORDABLE,
        PaymentMethod.NOT_RECOMMENDED,
        (),
        None,
        (),
        "",
    )
    assert (
        "deadline"
        not in build_decision_explanation(
            fallback, request=fallback_ctx.request, profile=fallback_ctx.profile
        ).casefold()
    )
    investment = replace(fallback_ctx.request, request_type=RequestType.INVESTMENT)
    assert "not investment performance" in build_decision_explanation(
        fallback, request=investment, profile=fallback_ctx.profile
    )


@pytest.mark.parametrize(
    "payload",
    [
        "",
        "NaN",
        "Infinity",
        '{"answer": true}',
        "```text```",
        "**bold**",
        "line1\nline2",
        "bad|field",
    ],
)
def test_malformed_explanations_fail_closed(payload):
    with pytest.raises(OutputValidationError):
        decision_to_output_row(replace(full_decision(), decision_explanation=payload))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("amount_safe_to_pay", "NaN"),
        ("affordability_status", "yes"),
        ("recommended_payment_method", "cash"),
        ("payment_plan", "2026-09-01,100"),
        ("earliest_date_for_full_payment", "09/01/2026"),
        ("spending_changes_needed", "stop,event_1"),
    ],
)
def test_malformed_output_fields_fail_closed(field, value):
    row = decision_to_output_row(full_decision())
    row[field] = value
    with pytest.raises(OutputValidationError):
        parse_output_row(row)


def test_full_row_rejects_fabricated_explanation_and_noncanonical_money():
    ctx = context()
    row = decision_to_output_row(full_decision(ctx))
    row["decision_explanation"] = "A guaranteed windfall makes this risk-free."
    with pytest.raises(OutputValidationError, match="deterministic template"):
        validate_output_rows((row,), (ctx,))

    row = decision_to_output_row(full_decision(ctx))
    row["amount_safe_to_pay"] = "0100.00"
    with pytest.raises(OutputValidationError, match="canonically formatted"):
        validate_output_rows((row,), (ctx,))


def test_installment_output_requires_supplied_option_precision():
    option = installment_option()
    ctx = context(
        prof=profile(methods=(PaymentMethod.INSTALLMENTS,)), options=(option,)
    )
    decision = Decision(
        "request_1",
        Decimal("100"),
        AffordabilityStatus.AFFORDABLE_WITH_PLAN,
        PaymentMethod.INSTALLMENTS,
        (
            Payment(TODAY, Decimal("55.00")),
            Payment(TODAY + timedelta(days=30), Decimal("55.00")),
        ),
        TODAY,
        (),
        "",
    )
    decision = replace(
        decision,
        decision_explanation=build_decision_explanation(
            decision,
            request=ctx.request,
            profile=ctx.profile,
            selected_option=option,
        ),
    )
    row = decision_to_output_row(decision)
    row["payment_plan"] = row["payment_plan"].replace("55.00", "55")
    with pytest.raises(OutputValidationError, match="canonically formatted"):
        validate_output_rows((row,), (ctx,))


def test_exact_header_id_order_and_duplicate_gates():
    ctx = context()
    row = decision_to_output_row(full_decision(ctx))
    wrong_header = dict(row)
    wrong_header["extra"] = "x"
    with pytest.raises(OutputValidationError, match="columns"):
        parse_output_row(wrong_header)
    with pytest.raises(OutputValidationError, match="row count"):
        validate_output_rows((), (ctx,))
    second = replace(ctx, request=replace(ctx.request, request_id="request_2"))
    with pytest.raises(OutputValidationError, match="out of order|duplicate"):
        validate_output_rows((row, row), (ctx, second))


def test_atomic_write_validates_before_replace_and_quotes_commas(tmp_path):
    ctx = context()
    target = tmp_path / "output.csv"
    target.write_text("previous-valid-output\n", encoding="utf-8")
    bad = replace(full_decision(ctx), amount_safe_to_pay=Decimal("99"))
    with pytest.raises(OutputValidationError):
        write_output_csv_atomic((bad,), (ctx,), target)
    assert target.read_text(encoding="utf-8") == "previous-valid-output\n"

    good = full_decision(ctx)
    write_output_csv_atomic((good,), (ctx,), target)
    assert validate_output_csv(target, (ctx,)) == (good,)
