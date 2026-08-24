# Human-review protocol for the synthetic exact-cell dataset

## Purpose

This workflow reduces manual annotation without replacing human judgment. It
constructs 500 review candidates from real T²-RAGBench financial tables. The
financial values and tables are never invented. Only the question wrapper,
exact-cell mapping and controlled hard-negative package are generated.

The generated set requires human verification before use as labelled data. The
primary researcher completed the first-pass review on 24 August 2026 and
approved all 500 items. A blind independent-review sample remains required
before confirmatory evaluation.

## Deterministic construction

Run:

```sh
make synthetic-500
```

The generator uses the pinned dataset revision and project seed. It:

1. verifies every raw source file against its manifest hash;
2. excludes all contexts used in the earlier 30-question pilot and 15-question
   screening batch;
3. parses Markdown tables while removing FinQA's synthetic display-index column;
4. retains only numeric cells with a meaningful concept and an explicit,
   single-year column header;
5. rejects questions with configured calculation or comparison phrases;
6. requires every meaningful row-label token to appear in the source question;
7. requires words such as `average`, `percentage`, `increase`, `maximum` and
   `minimum` to be visible in the target row, column or raw value;
8. allows question years to refer only to the target period or filing report
   year, preventing accidental multi-period calculations;
9. requires the target value, year and concept to align with the source
   dataset's original answer and question;
10. requires the target numeric value to be unique across every numeric cell in
   the full table;
11. requires one natural wrong-period cell and one natural wrong-concept cell;
12. selects at most one question per table and six per company within each
   subset; and
13. ranks candidates by a seeded SHA-256 key before taking the configured
    subset totals.

Question wording is a deterministic table-grounded wrapper around the original
dataset question. This preserves financial scope that may exist outside a flat
row header. It does not use a moving LLM or external API.

## Current candidate set

- Total: 500
- ConvFinQA: 176
- TAT-DQA: 324
- FinQA: 0

The split was rebalanced by two questions after strengthening the uniqueness
test to cover every numeric cell in the full source table, not only cells that
passed the other eligibility filters. This keeps the total at 500 without
weakening the quality rules.

FinQA is excluded from this 500-candidate construction because its surviving
automatically matched questions were predominantly maximum, minimum,
percentage-change or other arithmetic questions. A coincidentally matching cell
is not treated as a direct-lookup label. FinQA can still be used elsewhere in
the project for arithmetic or external analyses, but it is not forced into this
exact-cell direct-lookup set.

## Human review

The generated review pack is
`results/synthetic/synthetic_questions_500_review.md`. It contains one compact
table of all questions followed by a collapsible source-table section for each
item. The version-controlled review log is
`annotations/synthetic_questions_500_review.yaml`.

For every item, the researcher records:

- `approve` when the question uniquely identifies the proposed target cell and
  both hard negatives are genuinely wrong;
- `edit` when the mapping is valid but the question needs revised wording;
- `reject` when the target or negative mapping is semantically wrong; or
- `unsure` when adjudication is required.

Codex's `approve` suggestion was advisory. The primary researcher inspected the
complete pack and recorded 500 `approve` decisions. This first-pass review was
not blind and does not replace the later independent-review sample.

## Representation round trip

`make fact-roundtrip` parses every complete source table before consulting the
approved target or hard-negative labels. It then creates and serializes the
table-wide `FinancialFact` collection and verifies the labelled cells by stable
cell ID. The current run materialises 8,082 facts and recovers 500/500 targets
and 1,000/1,000 hard negatives. Full details and limitations are recorded in
[`fact_roundtrip_report.md`](fact_roundtrip_report.md).

## Use restrictions

- Do not report the 500 candidates as a gold benchmark before review.
- Do not estimate real-world hallucination or error prevalence from synthetic
  hard negatives.
- Do not split questions randomly. After approval, group by company, filing and
  table family before assigning development and confirmatory partitions.
- Do not allow the target label or generated mismatch type to enter retrieval
  input features.
