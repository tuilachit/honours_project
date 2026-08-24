# Exact-cell feasibility pilot report

## Verdict

**The 30 proposed source-cell mappings reproduce correctly, but the wider
feasibility gate has not passed yet.** The pinned T²-RAGBench files contain
enough varied examples to build a small exact-cell development set. This first
pass does not yet establish independent semantic-label reliability, annotation
cost, dataset prevalence, model accuracy or scalability to a 300–500-question
benchmark.

## Scope and provenance

- **Pilot ID:** `t2_exact_cell_feasibility_v1`
- **Dataset:** `G4KMU/t2-ragbench`
- **Pinned revision:** `adf7fe1541ac37351ce1142544d8e3b43010ed92`
- **Annotation status:** `first_pass_review_complete`
- **Review process:** AI-assisted candidate preparation and an initial
  delegated source review, followed by the primary researcher's full review of
  the generated 30-question source pack
- **Intended use:** project development only
- **Generated result:** `results/pilot/pilot_validation.json` (gitignored)

The earlier delegated review remains in the provenance history. The final
source decision is now based on the primary researcher's inspection of all 30
source sections. This is still not blind independent annotation because the
pack displayed the proposed cells and alternatives.

## Validated content

| Check | Result |
|---|---:|
| Included direct-value questions | 30 |
| ConvFinQA / FinQA / TAT-DQA | 10 / 10 / 10 |
| Reproducible gold cells | 30 |
| Unique stable gold-cell IDs | 30 |
| Reproducible natural near misses | 91 |
| Unique stable near-miss cell IDs | 91 |
| Questions containing annotation notes | 23 |
| Recorded exclusions in the final YAML | 0 |
| Questions with measured active annotation time | 0 |
| Full-source reviewed by the primary researcher | 30 |
| Included by the primary researcher | 30 |
| Historical delegated AI source reviews | 26 |

The 23 questions with notes must not be interpreted as 23 ambiguous labels.
The notes mix malformed structure, resolved interpretation, duplicate values,
dataset conflicts and routine coordinate-transform details.

### Source split composition

| Source | Train | Dev | Test | Other | Total |
|---|---:|---:|---:|---:|---:|
| ConvFinQA | 0 | 0 | 0 | 10 `turn_0/all` | 10 |
| FinQA | 7 | 2 | 1 | 0 | 10 |
| TAT-DQA | 4 | 3 | 3 | 0 | 10 |

Original dataset split names do not make this a held-out test set because the
researcher has now seen the questions, cells and negatives. Every selected
company, filing, document, table and related question family must stay in the
project-development split or be excluded from confirmatory evaluation.

## What the validator checked

The `make pilot` workflow:

1. verifies the dataset revision and manifest hash;
2. verifies byte counts and SHA-256 hashes for all seven required JSONL files;
3. locates each question and checks its exact source text, context, file, split
   and available page value, plus both stored dataset-answer fields;
4. reproduces each selected table hash;
5. applies the configured FinQA synthetic-index transform;
6. round-trip checks every semantic and raw Markdown coordinate and value;
7. rejects duplicate gold addresses, repeated negatives and gold/negative
   overlap;
8. derives stable SHA-256 IDs for gold and near-miss cells; and
9. writes the standard result envelope with Git commit, config hash, UTC
   timestamp and full resolved config.

A separate read-only integrity pass found four question strings that had been
cleaned during annotation. They were restored to the exact raw wording before
the successful validation run. No answer, coordinate or cell value changed.
The automated validator does not decide whether a human-interpreted header,
query dimension or near-miss reason is semantically correct; those labels still
require full-source human inspection and independent review.

## What this pilot supports

The evidence supports this narrow conclusion:

> The pinned T²-RAGBench release contains at least 30 purposively selected
> direct-value questions, balanced across its three source subsets, that can be
> mapped reproducibly to one proposed gold cell and two to four plausible wrong
> cells.

It also demonstrates real structural cases relevant to the thesis, including
malformed or promoted headers, hierarchical headers, split row labels,
duplicate displayed values, closely related totals and components, missing
units, multiple table blocks and FinQA's synthetic display index.

The near-miss field counts describe how the alternatives were deliberately
chosen. They are **not observed retrieval-error rates** and must not be used to
claim that wrong-period or wrong-concept errors are prevalent.

## Open feasibility gates

| Gate | Current status | Required action |
|---|---|---|
| Active annotation time | Batch 1 prepared; not yet measured | Review the fixed 15-candidate batch with a stopwatch and record total active minutes. |
| Complete screening trail | Fixed batch and 15 advisory Codex pre-labels recorded; human decisions pending | Critique all 15 suggestions and record every final decision and exclusion reason. |
| Synthetic construction | 500 deterministic candidates generated from real tables; human decisions pending | Review, edit or reject every proposed question, target and hard-negative package. |
| Full human source inspection | Complete: 30 of 30 | No remaining action for this first-pass source review. |
| Independent agreement | Not performed | Blindly annotate a configured stratified sample before adjudication. |
| `FinancialFact` round trip | Not implemented | Materialise target and negative facts without using oracle labels as retrieval inputs. |
| B0 and B1 pilot runs | Not implemented | Run genuine chunk and cell rankings before judging method feasibility. |
| Scale decision | Not ready | Use prospective timing, ambiguity and agreement evidence to estimate 300–500-label effort. |

## Reproduction

After acquiring the pinned structured metadata, run:

```sh
make pilot
make review-pack
make screening-batch
make synthetic-500
```

The source annotations are in
[`../annotations/pilot_annotations.yaml`](../annotations/pilot_annotations.yaml),
and the procedure is defined in
[`pilot_annotation_guide.md`](pilot_annotation_guide.md). The output status is
`validated_with_open_feasibility_gates`; it must not be relabelled as a passed
pilot until the missing evidence above is collected.

The second command writes the gitignored human review artifact to
`results/pilot/human_source_review_pack.md`.

The third command reproduces the fixed prospective batch and writes its
gitignored review pack to `results/pilot/prospective_screening_batch_01.md`.
Its version-controlled decision log is
[`../annotations/prospective_screening_batch_01.yaml`](../annotations/prospective_screening_batch_01.yaml).

The fourth command generates the 500-question human-review candidate set. Its
method and use restrictions are defined in
[`synthetic_dataset_protocol.md`](synthetic_dataset_protocol.md).
