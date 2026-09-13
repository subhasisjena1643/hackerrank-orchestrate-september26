Your previous image-extraction response failed strict JSON/schema validation.

Validation errors:
{{VALIDATION_ERRORS_JSON}}

Return corrected JSON only. Preserve the trusted image_id and event_id. Do not invent or change an amount, currency, or date merely to satisfy the schema. If the linked amount is ambiguous, set amount to null and confidence to low. Do not return Markdown or any recommendation.
