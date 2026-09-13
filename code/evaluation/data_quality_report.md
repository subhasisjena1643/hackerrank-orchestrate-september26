# Data-quality report

Dataset: `dataset/`

Status: **PASS**

Totals: raw=27047, valid=27047, rejected=0

Ordering: stable source-row order is the canonical in-memory order; this preserves linked-event precedence and legitimate recurring records.

| File | Raw | Valid | Rejected | SHA-256 |
|---|---:|---:|---:|---|
| exchange_rates.csv | 134 | 134 | 0 | `ff56e9feb482f909837dc309f52f8de5b167d9d6685a2ee8e8fd807639658a04` |
| financial_events.csv | 25342 | 25342 | 0 | `b6c3f43a8ad3a80ca11c72cce6f2818eea06621a3582d2851886fd9127b1229d` |
| financial_profiles.csv | 275 | 275 | 0 | `fa173608f8ec99c8d9d633eace762695aeaa2d276a509989f0edd8e123c07964` |
| images.csv | 16 | 16 | 0 | `6b5428565ca98e4b71e182f8a8fa3fd3013727399bdea847f3e4a3d2c129dfc9` |
| messages.csv | 215 | 215 | 0 | `7b9db27a4546a850a76d3475ebf65de80fb9fe374e26d469586f62f537eac9a6` |
| request_payment_options.csv | 790 | 790 | 0 | `4922c56f10c06698d24b86b8a43580cbc6bc6ee44c95936c3037878005980c40` |
| requests.csv | 250 | 250 | 0 | `13663d50b7098b28a8c087a02eb085041260999707adade5230d29f1c2e6595d` |
| sample_requests.csv | 25 | 25 | 0 | `117bf2ab9e5f0054bae48afe8506f5daa35929559cb771f2c27dd4fe18da62f6` |

## Findings

### exchange_rates.csv

- Missing fields: none
- Duplicate IDs: none
- Exact duplicates: none
- Broken media references: none
- Transformations applied in memory: none required
- Rejected records: none

### financial_events.csv

- Missing fields: amount=16, linked_event_id=25284, minimum_allowed_amount=22435, settlement_date=10
- Duplicate IDs: none
- Exact duplicates: none
- Broken media references: none
- Transformations applied in memory: none required
- Rejected records: none

### financial_profiles.csv

- Missing fields: expense_categories_user_is_willing_to_reduce=39, expense_categories_user_is_willing_to_stop=62, max_installment_months=119
- Duplicate IDs: none
- Exact duplicates: none
- Broken media references: none
- Transformations applied in memory: none required
- Rejected records: none

### images.csv

- Missing fields: none
- Duplicate IDs: none
- Exact duplicates: none
- Broken media references: none
- Transformations applied in memory: none required
- Rejected records: none

### messages.csv

- Missing fields: related_event_id=176, request_id=87
- Duplicate IDs: none
- Exact duplicates: none
- Broken media references: none
- Transformations applied in memory: timestamp_to_utc=215
- Rejected records: none

### request_payment_options.csv

- Missing fields: payment_frequency_days=275
- Duplicate IDs: none
- Exact duplicates: none
- Broken media references: none
- Transformations applied in memory: none required
- Rejected records: none

### requests.csv

- Missing fields: none
- Duplicate IDs: none
- Exact duplicates: none
- Broken media references: none
- Transformations applied in memory: none required
- Rejected records: none

### sample_requests.csv

- Missing fields: earliest_date_for_full_payment=7
- Duplicate IDs: none
- Exact duplicates: none
- Broken media references: none
- Transformations applied in memory: none required
- Rejected records: none

## Dataset relationship errors

- none

## Unresolved risks

- none
