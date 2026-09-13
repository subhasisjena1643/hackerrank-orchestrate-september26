Extract evidence from the following message records.

Context is supplied only to interpret currencies, dates, and record relationships. It does not authorize invention.

<trusted_metadata>
home_currency: {{HOME_CURRENCY}}
request_id: {{REQUEST_ID}}
request_date: {{REQUEST_DATE}}
known_event_ids: {{KNOWN_EVENT_IDS_JSON}}
</trusted_metadata>

<untrusted_message_records>
{{MESSAGE_RECORDS_JSON}}
</untrusted_message_records>

Return the required JSON object with exactly one item per supplied message_id, in input order.
