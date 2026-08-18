# Exact-cell pilot annotation guide

## Purpose

This pilot checks whether T²-RAGBench can support the thesis's exact-cell
retrieval experiment. It is not a model evaluation. The first-pass output is a
small, researcher-approved development answer key containing 30 direct-value
questions: 10 from ConvFinQA, 10 from FinQA and 10 from TAT-DQA. Researcher
approval of AI-assisted pre-annotations is not an independent second annotation.
If the researcher delegates later source decisions under a standing rule, those
decisions are recorded separately as delegated AI review. They do not count as
individual full-source inspection by the researcher.

For every included question, the answer key identifies the complete set of
source cells that count as correct and several real nearby cells that could
tempt a retrieval system into returning the wrong value.

## The key distinction

A question is a **direct lookup** when the requested answer is already printed
in one table cell. Understanding the company, metric, period, unit and other
conditions is still necessary, but no new number must be calculated.

A question is **derived** when the answerer must add, subtract, divide, count,
compare or otherwise combine source values. A total calculated by the company
is still a direct lookup if that total is already printed in a cell and the
question asks for the reported total.

Example:

| Question | Classification |
|---|---|
| What was net cash from operating activities in 2009? | Direct lookup when the 2009 total is printed in the table. |
| How much did net cash increase from 2008 to 2009? | Derived because two cells must be subtracted. |

## Eligibility rules

Include a question only when all of the following are true:

1. Its answer is explicitly printed in at least one source-table cell.
2. No arithmetic, counting, comparison or yes/no judgement is required.
3. The requested company, concept, period, unit and other stated dimensions can
   be mapped to the source table without guesswork.
4. Every cell that could validly answer the question can be listed.
5. The source location and table coordinates can be reproduced from the pinned
   dataset snapshot.

Exclude a question when any of the following apply:

- the answer must be calculated or compared;
- the answer is supported only by prose;
- the matched number occurs in a cell unrelated to the question;
- the table is too damaged to identify the requested row or column reliably;
- a unit, period, scope or other necessary condition cannot be resolved; or
- the complete set of valid answer cells cannot be determined.

An automatic numeric match is only a screening suggestion. It is never treated
as a gold label without human review.

## Review procedure

For each candidate:

1. Read the question and write down its requested company, concept, period,
   unit and any additional dimensions.
2. Read the whole table plus nearby source text. Do not judge from the answer
   number alone.
3. Decide `INCLUDE` or `EXCLUDE` and record a short reason.
4. For an included question, record every valid source cell.
5. Record two to four genuine source cells that are plausible but wrong.
6. Describe why each alternative is wrong, using field errors such as wrong
   company, concept, period, unit or another table-specific dimension.
7. Record malformed headers, inferred structure, duplicate values and other
   ambiguity.
8. Record the annotation time in seconds.

## Cell coordinates and identity

- Every non-separator Markdown row is counted from `0` in source order. The
  first header row is therefore row `0`; tables with multiple header rows have
  a later first data-row index.
- Markdown separator lines are syntax and are not counted.
- The left-hand stub or row-label column is column `0`.
- The first value column is column `1`.
- FinQA's exported Markdown contains a synthetic display-index column (`0`,
  `1`, `2`, ...). That generated column is removed before semantic column
  indices are assigned because it is not a financial source cell. When this
  transform applies, the raw Markdown column index is also retained.
- Raw row and column header text must be preserved, even when a cleaned or
  inferred interpretation is also recorded.
- Raw and normalised values must both be preserved.

The stable cell ID is the configured SHA-256 digest of a canonical address
containing the pinned dataset revision, subset, split, context ID, table index,
row index and column index. The readable address fields are stored beside the
digest so a reviewer can inspect it. IDs are derived and round-trip checked by
the validator; annotators do not type hashes manually into the source YAML.

## Multiple valid cells

If the same question has more than one genuinely valid source cell, list all of
them. Do not mark a valid duplicate as a hard negative. Exact-cell retrieval is
correct when the returned cell ID belongs to this complete valid set.

## Pilot completion checks

The annotation content passes its first validation only when:

- 30 included direct-value questions are present, balanced 10/10/10 across the
  three source subsets;
- every included question has at least one reproducible gold cell;
- every gold cell preserves raw value, raw headers and source provenance;
- natural near misses have been checked against all valid gold cells;
- known exclusions and ambiguity are retained rather than silently discarded;
- stable IDs are generated for gold cells and natural near misses; and
- raw files, table hashes, coordinates and values pass the automated validator.

The wider feasibility gate is not complete until prospective screening and
exclusion decisions are logged, fresh annotation timing is measured, and a
blind second annotator independently checks a configured sample before
adjudication. All pilot companies, filings, tables and related question
families are development material and must be excluded from the later
confirmatory test split.

Run `make review-pack` to generate a single human-readable file containing all
30 questions, complete relevant tables, proposed cells, alternatives, source
notes and expandable extracted context. A question counts as individually
human-inspected only after the researcher actually reads its source section and
records an INCLUDE, EXCLUDE or UNSURE decision.

Run `make screening-batch` to reproduce the first prospective timing batch.
The command uses a fixed seeded rank to select five previously unused automatic
candidates from each source subset. It writes the human-readable pack to
`results/pilot/prospective_screening_batch_01.md` and preserves decisions in
`annotations/prospective_screening_batch_01.yaml`. The researcher must start a
timer before the first item, review every item in the fixed order, record all
exclusions and reasons, and report total active minutes. Automatic cell matches
are deliberately not pre-filtered by a human or model.
