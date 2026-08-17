# T²-RAGBench Feasibility Audit

## Material Passport

- **Artifact ID:** T2-AUDIT-20260815
- **Artifact type:** Dataset feasibility audit
- **Verification status:** ANALYZED
- **Audit date:** 15 August 2026
- **Dataset:** `G4KMU/t2-ragbench`
- **Pinned revision:** `adf7fe1541ac37351ce1142544d8e3b43010ed92`
- **Release:** Cleaned release without VQAonBD
- **Data examined:** All seven structured JSONL metadata files
- **Source PDFs/images examined:** No
- **Audit config hash:** `6f7afee2fea3e1c8e611cbc7236695d50fdee8681ccc2db24ff0c395cff0f9b8`
- **Code Git commit recorded by the run:** `affed0fce0251a597f584864ddcda9a9c544b403`
- **Raw result:** `results/data_audit/t2_ragbench.json` (gitignored)

## Verdict

**CONDITIONAL GO.** The cleaned release is large enough to construct a 300–500-question exact-cell test set. The automatic screen found thousands of plausible candidates, including many questions asking for a value from a multi-year table. However, T²-RAGBench does not directly label exact cells or reliably distinguish direct extraction from arithmetic and comparison questions. Human screening and, where possible, joins to original source annotations remain necessary. Its differences in source schema and table encoding are also useful for testing a schema-flexible representation, but the audit does not yet prove generalisation to unseen table structures.

## What was downloaded

The official repository currently contains approximately 2.8 GB including PDFs and images. This audit downloaded only the structured metadata required for schema and eligibility inspection:

- 7 JSONL files;
- 165,007,817 bytes in total;
- 23,088 rows;
- exact file sizes and SHA-256 digests recorded in [`configs/datasets/t2_ragbench.yaml`](../configs/datasets/t2_ragbench.yaml).

The source revision, rather than the moving `main` branch, is pinned. The [current dataset card](https://huggingface.co/datasets/G4KMU/t2-ragbench) states that VQAonBD was removed because its reformulated questions were low quality. This explains why older descriptions report 32,908 rows while the cleaned release and the [peer-reviewed EACL paper](https://aclanthology.org/2026.eacl-long.8/) report 23,088.

## Integrity results

Every downloaded file matched its configured byte count, row count, and SHA-256 digest.

| Subset | Split | Rows | Distinct contexts |
|---|---|---:|---:|
| ConvFinQA | turn_0 | 3,458 | 1,806 |
| FinQA | train | 6,251 | 2,110 |
| FinQA | dev | 883 | 299 |
| FinQA | test | 1,147 | 380 |
| TAT-DQA | train | 9,063 | 2,172 |
| TAT-DQA | dev | 1,142 | 274 |
| TAT-DQA | test | 1,144 | 277 |
| **Total** |  | **23,088** | **7,318** |

The observed totals match the peer-reviewed dataset description.

## Schema findings

### FinQA and ConvFinQA

These subsets have 21 fields and provide a separate Markdown `table` field. They also provide company name, ticker, report year, page number, sector, industry, headquarters, CIK, and source filename.

Important inconsistencies still require normalisation:

- FinQA stores `report_year` and `page_number` as integers;
- ConvFinQA stores them as strings;
- five ConvFinQA questions are empty;
- one FinQA test question is empty;
- 74 FinQA rows have an empty `original_answer`, although `program_answer` is present.

### TAT-DQA

TAT-DQA has only 11 fields. It does not provide a separate table column or page number. Instead, one or more Markdown tables appear inside the `context` string.

- 11,076 of 11,349 rows contain a detected Markdown table;
- 273 rows have no detected Markdown table;
- 25,228 table blocks were detected, confirming that many contexts contain multiple tables;
- five questions are empty across its train and test files.

This means TAT-DQA requires table-block identification before cell addressing. A source filename alone is not a sufficient table identity.

## Automatic exact-cell candidate screen

The audit parsed displayed numeric values from Markdown cells and compared them with `program_answer` and `original_answer`. This is a high-recall screening method, not gold annotation.

| Subset | Rows | Rows with Markdown tables | Any answer–cell match | Unique answer–cell match | Requested year in multi-year table |
|---|---:|---:|---:|---:|---:|
| ConvFinQA | 3,458 | 3,458 | 1,116 | 1,016 | 879 |
| FinQA | 8,281 | 8,281 | 587 | 505 | 426 |
| TAT-DQA | 11,349 | 11,076 | 3,167 | 2,294 | 2,891 |
| **Total** | **23,088** | **22,815** | **4,870** | **3,815** | **4,196** |

Across the full release:

- 98.82% of rows contain a detected Markdown table;
- 21.09% have at least one displayed cell matching an answer value;
- 16.52% have exactly one matching numeric cell under the automatic parser;
- 4.57% have multiple matching cells;
- 18.17% ask for a year that appears in a table containing at least two years.

These counts demonstrate a sufficiently large candidate pool, but they must not be reported as the number of valid direct-value questions.

## Why human screening is still required

The automatic candidates include clear direct extraction questions, such as:

- the amount of income before income taxes in 2019;
- net cash from operating activities in a specified fiscal year;
- total revenue in a specified reporting year.

They also include ineligible questions whose computed answer happens to occur somewhere in the table, such as:

- whether one cumulative return exceeded another, with answer `1`;
- the change in a liability between two years;
- an average over several years;
- how many years satisfy a condition.

Conversely, the screen can miss a valid cell when answer normalisation changes scale, sign, percentage form, or display format. Therefore:

> **An automatic answer–cell match is a candidate-generation signal, not an exact-cell gold label.**

## Implications for the experiment

1. Use `context_id`, not only `file_name`, as the corpus context key. The audit found 7,318 contexts but only 5,512 distinct filenames.
2. Give every TAT-DQA Markdown block a stable table ID before assigning cell IDs.
3. Represent each eligible cell as a `FinancialFact` with entity, concept, period, unit, value, source and provenance.
4. Preserve complete raw row-header and column-header paths. Normalised fields must not replace the original structural evidence.
5. Store any additional table context as arbitrary name–value dimensions. Do not force FinQA, ConvFinQA and TAT-DQA into one closed list of scope fields.
6. Normalise report-year and page types across subsets while retaining raw values and source-specific provenance.
7. Exclude empty questions before sampling.
8. Join original FinQA, ConvFinQA and TAT-family operation annotations where legally and technically possible. These annotations can distinguish direct extraction from arithmetic more reliably than wording heuristics.
9. Manually verify every development and test label against the complete table and later against the PDF where necessary.
10. Preserve multiple valid facts instead of forcing one target when the same answer is legitimately repeated.
11. Define table-structure families before inspecting model errors, then hold out companies, filings and structure families for generalisation analysis.
12. Evaluate the flattened B0 baseline using chunk evidence coverage and downstream answers. Do not derive a false exact-cell ranking from the fact that a retrieved chunk contains a target value.

## Week 4 pilot sampling recommendation

Begin with 30 questions:

- 10 from ConvFinQA;
- 10 from FinQA;
- 10 from TAT-DQA.

Within each subset, prioritise questions that:

- have one automatic answer–cell match;
- mention a requested year that appears in a multi-year table;
- contain at least one other numeric cell as a natural distractor;
- ask directly for a reported amount rather than a change, ratio, average, count, comparison, or yes/no decision.

Record every exclusion reason, unexpected table dimension and raw header path. Time the annotation work. The full 300–500-question plan is feasible only if the pilot has acceptable ambiguity and annotation time, and if `FinancialFact` can preserve the heterogeneous source structures without ad hoc schema changes.

## Pilot follow-up — 17 August 2026

A purposive 30-question first pass has now been completed: 10 questions each
from ConvFinQA, FinQA and TAT-DQA. The validator reproduced 30 gold cells and
91 natural near-miss cells from the pinned raw files and generated stable IDs
for all of them. This confirms that the three source schemas can supply a small
exact-cell development sample.

This does **not** close the Week 4 feasibility gate. Active annotation time was
not measured, the screening/exclusion log is incomplete, the primary reviewer
approved AI-assisted summaries rather than recording full-source inspection,
and no blind independent annotation has been completed. The selected source
families are development material and must not enter the later confirmatory
test set. Full results and claim limits are in
[`pilot_feasibility_report.md`](pilot_feasibility_report.md).

## Reproduction

After acquiring the configured raw files, run:

```sh
make audit-data
```

The command validates all file checksums and counts before reading the dataset. It writes a stamped JSON envelope containing the Git commit, stable config hash, timestamp, full resolved configuration, per-source statistics, and a deterministic list of screening candidates. The separate `make pilot` workflow validates the first-pass exact-cell labels after the raw metadata are present. Retrieval and evaluation conditions remain intentional stubs.

## Limitations of this audit

- PDFs and images were not downloaded or visually checked.
- Markdown parsing does not fully reconstruct merged cells or malformed hierarchical headers.
- Numeric equality alone cannot determine whether a question requires arithmetic.
- The 140 saved pilot suggestions are screening candidates, not approved labels.
- The audit does not yet construct `FinancialFact` dimensions or validate unseen-structure generalisation.
- The audit establishes data feasibility, not retrieval performance.
