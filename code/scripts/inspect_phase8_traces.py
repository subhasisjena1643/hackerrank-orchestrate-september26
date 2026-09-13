"""Print compact sample-user cash-flow traces without loading answer columns."""

from pathlib import Path

from buy_or_wait.data_loader import load_dataset
from buy_or_wait.evidence.messages import extract_message_evidence
from buy_or_wait.finance.cashflows import construct_future_cash_flows
from buy_or_wait.finance.lifecycle import resolve_event_lifecycles
from buy_or_wait.finance.recurrence_policies import BASELINE_POLICY


dataset = load_dataset(Path("dataset"), include_sample_outputs=False)
profiles = {profile.user_id: profile for profile in dataset.profiles}
for request in dataset.sample_requests[:6]:
    profile = profiles[request.user_id]
    events = tuple(
        event for event in dataset.financial_events if event.user_id == request.user_id
    )
    messages = tuple(
        message for message in dataset.messages if message.user_id == request.user_id
    )
    evidence = extract_message_evidence(
        messages,
        user_id=request.user_id,
        request_id=request.request_id,
        request_date=request.request_date,
        home_currency=profile.home_currency,
        known_event_ids=(event.event_id for event in events),
    ).evidence
    resolved = resolve_event_lifecycles(
        events,
        snapshot_date=request.request_date,
        home_currency=profile.home_currency,
        exchange_rates=dataset.exchange_rates,
        message_evidence=evidence,
        allow_unresolved_amounts=True,
    )
    trace = construct_future_cash_flows(
        resolved,
        request_date=request.request_date,
        policy=BASELINE_POLICY,
        message_evidence=evidence,
    )
    preview = ";".join(
        f"{flow.flow_date}:{flow.direction.value}:{flow.amount}:{flow.category}:"
        f"{'S' if flow.synthetic else 'E'}"
        for flow in trace.cash_flows[:6]
    )
    print(
        f"{request.request_id}/{request.user_id} flows={len(trace.cash_flows)} "
        f"explicit={sum(not flow.synthetic for flow in trace.cash_flows)} "
        f"inferred={sum(flow.synthetic for flow in trace.cash_flows)} "
        f"dedup={len(trace.suppressed)} :: {preview}"
    )
