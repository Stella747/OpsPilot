# OpsPilot synthetic dataset card

## Purpose

Provide a realistic, reproducible-looking sample for developing and demonstrating incident retrieval and report generation without using operational data.

## Contents

- 15 historical incident records across 10 fictional services.
- 34 fictional timestamped application log records.
- 16 labeled retrieval queries: 14 expected historical matches and 2 unrelated negative cases.
- All records are explicitly marked `SYNTHETIC`.

## Limitations

These hand-authored examples are illustrative and intentionally small. The expected matches and root causes are scenario labels, not verified truth about real incidents. Retrieval evaluation results should not be generalized to production incident data. The records contain no real company, customer, or production details.
