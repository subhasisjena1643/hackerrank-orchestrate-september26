# Phase 9 Baseline Sample Diff

Policy hash: `743bad6adb94c55064191e2888e7cc2c1433d11ae008e6ae04a0cee7cdd5199b`

This is a baseline-only diagnostic. It compares only safe amount and earliest
full-payment date; it does not tune policy parameters or generate recommendations.
The five image-linked sample amounts use the Phase 6 source-image values.

| request_id | expected safe | baseline safe | safe diff | expected earliest | baseline earliest |
|---|---:|---:|---:|---|---|
| request_01 | 25256 | 7497.45 | -17758.55 | 2024-03-03 |  |
| request_02 | 17229139.2 | 19058514.24 | 1829375.04 | 2025-09-15 | 2025-09-15 |
| request_03 | 873000 | 1160613.49 | 287613.49 | 2019-11-15 | 2019-10-15 |
| request_04 | 8401800 | 8952018.79 | 550218.79 | 2024-06-15 | 2024-06-15 |
| request_05 | 737 | 15488 | 14751 |  | 2025-11-06 |
| request_06 | 603.3 | 612.97 | 9.67 | 2026-01-15 | 2026-01-15 |
| request_07 | 87170.56 | 95762.92 | 8592.36 | 2024-10-23 | 2024-10-15 |
| request_08 | 284.57 | 0 | -284.57 | 2025-04-15 |  |
| request_09 | 166.61 | 166.61 | 0.00 | 2026-07-04 | 2026-07-04 |
| request_10 | 12700 | 266700 | 254000 |  | 2024-12-06 |
| request_11 | 12510645 | 13110000 | 599355 | 2025-07-15 | 2025-05-03 |
| request_12 | 65164 | 65164 | 0 | 2026-04-05 | 2026-04-05 |
| request_13 | 433.4 | 0 | -433.4 | 2024-05-15 |  |
| request_14 | 597.74 | 0 | -597.74 |  |  |
| request_15 | 83.05 | 0 | -83.05 |  |  |
| request_16 | 122500 | 122500 | 0 | 2023-08-12 | 2023-08-12 |
| request_17 | 243849.58 | 243199.23 | -650.35 | 2026-03-15 | 2026-04-15 |
| request_18 | 462 | 552.05 | 90.05 | 2026-09-15 | 2026-08-15 |
| request_19 | 28820 | 18871.38 | -9948.62 | 2024-09-15 | 2024-09-15 |
| request_20 | 5400 | 12342.43 | 6942.43 |  |  |
| request_21 | 1543.35 | 1574.4 | 31.05 | 2026-04-15 | 2026-04-03 |
| request_22 | 475.46 | 445.55 | -29.91 | 2025-01-15 | 2025-01-15 |
| request_23 | 9152 | 3633.31 | -5518.69 | 2025-07-15 | 2025-07-15 |
| request_24 | 13420 | 18216.83 | 4796.83 |  |  |
| request_25 | 1425000 | 2244365.68 | 819365.68 |  |  |
