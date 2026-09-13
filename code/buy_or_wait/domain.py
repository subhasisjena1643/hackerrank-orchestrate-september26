"""Typed, immutable domain contracts for Buy or Wait?."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path


class Currency(StrEnum):
    EUR = "EUR"
    IDR = "IDR"
    INR = "INR"
    USD = "USD"
    ZAR = "ZAR"


class EventType(StrEnum):
    DEBT_PAYMENT = "debt_payment"
    EXPENSE = "expense"
    INCOME = "income"
    INVESTMENT_PURCHASE = "investment_purchase"
    INVESTMENT_SALE = "investment_sale"
    INVESTMENT_VALUATION = "investment_valuation"
    REFUND = "refund"
    SUBSCRIPTION = "subscription"


class Direction(StrEnum):
    CREDIT = "credit"
    DEBIT = "debit"
    NON_CASH = "non_cash"


class EventStatus(StrEnum):
    CANCELLED = "cancelled"
    FAILED = "failed"
    PENDING = "pending"
    SCHEDULED = "scheduled"
    SETTLED = "settled"
    UNREALIZED = "unrealized"


class Flexibility(StrEnum):
    FIXED = "fixed"
    REDUCIBLE = "reducible"
    REDUCIBLE_OR_STOPPABLE = "reducible_or_stoppable"
    STOPPABLE = "stoppable"


class RequestType(StrEnum):
    PURCHASE = "purchase"
    TRAVEL = "travel"
    EDUCATION = "education"
    FAMILY_TRANSFER = "family_transfer"
    DEBT_REPAYMENT = "debt_repayment"
    INVESTMENT = "investment"
    HOUSING = "housing"
    EMERGENCY_EXPENSE = "emergency_expense"
    OTHER = "other"


class PaymentMethod(StrEnum):
    FULL_PAYMENT = "full_payment"
    PARTIAL_PAYMENT = "partial_payment"
    INSTALLMENTS = "installments"
    WAIT = "wait"
    NOT_RECOMMENDED = "not_recommended"


class SourceType(StrEnum):
    BANK = "bank"
    EMPLOYER = "employer"
    FINANCIAL_SERVICE = "financial_service"
    MERCHANT = "merchant"
    SERVICE_PROVIDER = "service_provider"


class AffordabilityStatus(StrEnum):
    AFFORDABLE_NOW = "affordable_now"
    AFFORDABLE_WITH_PLAN = "affordable_with_plan"
    AFFORDABLE_LATER = "affordable_later"
    NOT_AFFORDABLE = "not_affordable"


class SpendingChangeType(StrEnum):
    STOP = "stop"
    REDUCE_TO = "reduce_to"


@dataclass(frozen=True, slots=True)
class Profile:
    user_id: str
    home_currency: Currency
    current_available_balance: Decimal
    minimum_balance_to_keep: Decimal
    financial_priorities: tuple[str, ...]
    expense_categories_to_protect: tuple[str, ...]
    expense_categories_user_is_willing_to_reduce: tuple[str, ...]
    expense_categories_user_is_willing_to_stop: tuple[str, ...]
    payment_methods_user_will_consider: tuple[PaymentMethod, ...]
    max_installment_months: int | None


@dataclass(frozen=True, slots=True)
class FinancialEvent:
    event_id: str
    user_id: str
    event_type: EventType
    description: str
    category: str
    direction: Direction
    amount: Decimal | None
    currency: Currency
    event_date: date
    settlement_date: date | None
    status: EventStatus
    linked_event_id: str | None
    flexibility: Flexibility
    minimum_allowed_amount: Decimal | None


@dataclass(frozen=True, slots=True)
class ExchangeRate:
    rate_date: date
    from_currency: Currency
    to_currency: Currency
    rate: Decimal


@dataclass(frozen=True, slots=True)
class Request:
    request_id: str
    user_id: str
    request_date: date
    request_type: RequestType
    requested_amount: Decimal
    desired_completion_date: date
    allows_partial_payment: bool
    request_text: str


@dataclass(frozen=True, slots=True)
class PaymentOption:
    payment_option_id: str
    request_id: str
    payment_method: PaymentMethod
    payment_amount: Decimal
    number_of_payments: int
    first_payment_date: date
    payment_frequency_days: int | None
    financing_fee: Decimal
    total_payable_amount: Decimal


@dataclass(frozen=True, slots=True)
class MessageRecord:
    message_id: str
    user_id: str
    request_id: str | None
    related_event_id: str | None
    sent_at: datetime
    source_type: SourceType
    message_text: str


@dataclass(frozen=True, slots=True)
class ImageRecord:
    image_id: str
    user_id: str
    request_id: str | None
    related_event_id: str | None
    file_path: Path


@dataclass(frozen=True, slots=True)
class EvidenceFact:
    source_record_id: str
    fact_type: str
    value: str
    effective_date: date | None = None
    related_event_id: str | None = None


@dataclass(frozen=True, slots=True)
class CashFlow:
    cash_flow_id: str
    flow_date: date
    amount: Decimal
    direction: Direction
    category: str
    source_event_ids: tuple[str, ...]
    description: str = ""


@dataclass(frozen=True, slots=True)
class SpendingChange:
    change_type: SpendingChangeType
    event_id: str
    new_amount: Decimal | None = None


@dataclass(frozen=True, slots=True)
class Payment:
    payment_date: date
    amount: Decimal


@dataclass(frozen=True, slots=True)
class CandidatePlan:
    payment_method: PaymentMethod
    payments: tuple[Payment, ...]
    total_payable_amount: Decimal
    spending_changes: tuple[SpendingChange, ...] = ()
    payment_option_id: str | None = None


@dataclass(frozen=True, slots=True)
class ForecastResult:
    candidate_plan: CandidatePlan
    daily_balances: tuple[tuple[date, Decimal], ...]
    minimum_projected_balance: Decimal
    is_safe: bool
    failure_reason: str | None = None


@dataclass(frozen=True, slots=True)
class Decision:
    request_id: str
    amount_safe_to_pay: Decimal
    affordability_status: AffordabilityStatus
    recommended_payment_method: PaymentMethod
    payment_plan: tuple[Payment, ...]
    earliest_date_for_full_payment: date | None
    spending_changes_needed: tuple[SpendingChange, ...]
    decision_explanation: str


@dataclass(frozen=True, slots=True)
class RequestContext:
    request: Request
    profile: Profile
    financial_events: tuple[FinancialEvent, ...]
    payment_options: tuple[PaymentOption, ...]
    messages: tuple[MessageRecord, ...]
    images: tuple[ImageRecord, ...]
    exchange_rates: tuple[ExchangeRate, ...]
    evidence_facts: tuple[EvidenceFact, ...] = ()


@dataclass(frozen=True, slots=True)
class SampleOutput:
    """Solved sample labels, intentionally separate from prediction contexts."""

    request_id: str
    amount_safe_to_pay: Decimal
    affordability_status: AffordabilityStatus
    recommended_payment_method: PaymentMethod
    payment_plan: str
    earliest_date_for_full_payment: date | None
    spending_changes_needed: str
    decision_explanation: str


@dataclass(frozen=True, slots=True)
class Dataset:
    profiles: tuple[Profile, ...]
    financial_events: tuple[FinancialEvent, ...]
    exchange_rates: tuple[ExchangeRate, ...]
    evaluation_requests: tuple[Request, ...]
    sample_requests: tuple[Request, ...]
    sample_outputs: tuple[SampleOutput, ...]
    payment_options: tuple[PaymentOption, ...]
    messages: tuple[MessageRecord, ...]
    images: tuple[ImageRecord, ...]
