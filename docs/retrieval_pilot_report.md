# B0/B1/B2/M1 Development Retrieval Pilot

## Purpose

This pilot checks whether the flattened, hybrid fact, structured lookup and three-route candidate
union interfaces work before training a project-specific reranker. It is a development feasibility
test, not a confirmatory experiment.

The run used 500 first-pass-approved synthetic questions grounded in 500 real T²-RAGBench
tables. The fact corpus contained 8,082 losslessly round-tripped `FinancialFact` records.
B0 is retained as the earlier reference run. B1 was rerun with the same 20, 50 and 100 final
candidate budgets as M1. The result artifacts record their base commit and dirty-worktree
fingerprints below.

## Frozen components

- Dense encoder: [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- Generic reranker: [`cross-encoder/ms-marco-MiniLM-L6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2), revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`.
- Sparse route: deterministic BM25.
- Fusion: reciprocal-rank fusion.
- Structured route: deterministic automatic query extraction and schema-flexible compatibility
  scoring over common fields, observed dimensions and raw header context.
- Runtime: CPU, seeded deterministic inference. M1 runs its three retrieval routes concurrently
  per question and performs one generic reranker scoring pass for all three budget analyses.

## Conditions

| Condition | Retrieved unit | Meaning of success |
|---|---|---|
| B0 | One complete flattened source table | A retrieved table contains the labelled cell |
| B1 | One `FinancialFact`, followed by a generic reranker | The retrieved fact has the labelled source-cell ID |
| B2 | One `FinancialFact`, retrieved and ranked by automatic structured context | The retrieved fact has the labelled source-cell ID |
| M1 | Deduplicated dense, sparse and structured `FinancialFact` union, followed by the unchanged B1 generic reranker | The retrieved fact has the labelled source-cell ID |

B0 deliberately preserves each complete table. It is not a 2,000-character chunking
experiment. B0 and B1 therefore answer different questions and their percentages must not be
subtracted to claim a direct accuracy improvement.

## Results

| Condition and metric | @1 | @5 | @10 | @20 | MRR |
|---|---:|---:|---:|---:|---:|
| B0 table evidence coverage | 44.6% | 72.0% | 80.8% | 87.2% | — |
| B1 exact-cell retrieval | 78.2% | 94.4% | 96.2% | 97.6% | 0.8562 |
| B2 exact-cell retrieval | 80.8% | 95.0% | 96.2% | 97.2% | 0.8725 |
| M1 exact-cell retrieval | 78.2% | 94.4% | 96.2% | 98.6% | 0.8568 |

The primary B1 and M1 rows use a final reranker input budget of 100. Their matched sensitivity
results are:

| Budget | Condition | Hit@1 | Recall@5 | Recall@10 | Recall@20 | Candidate miss | Reranking miss |
|---:|---|---:|---:|---:|---:|---:|---:|
| 20 | B1 | 78.4% | 95.0% | 96.4% | 97.4% | 2.6% | 0.0% |
| 20 | M1 | 78.2% | 94.6% | 96.6% | 98.2% | 1.8% | 0.0% |
| 50 | B1 | 78.2% | 94.6% | 96.2% | 97.6% | 2.2% | 0.2% |
| 50 | M1 | 78.2% | 94.4% | 96.4% | 98.8% | 0.4% | 0.8% |
| 100 | B1 | 78.2% | 94.4% | 96.2% | 97.6% | 2.0% | 0.4% |
| 100 | M1 | 78.2% | 94.4% | 96.2% | 98.6% | 0.4% | 1.0% |

B1 stage errors:

- Candidate-generation miss: 2.0%.
- Additional reranking miss: 0.4%.
- Final top-20 miss: 2.4%.

B2 stage errors:

- Candidate-generation miss: 2.0%.
- Additional structured reranking miss: 0.8%.
- Final top-20 miss: 2.8%.

M1 stage errors at the primary 100-candidate budget:

- Candidate-generation miss: 0.4%.
- Additional reranking miss: 1.0%.
- Final top-20 miss: 1.4%.

B2 is 2.6 percentage points higher than B1 at Hit@1 on this development set. This is a
descriptive comparison between two exact-cell conditions, not a superiority claim. The automatic
extractor was corrected after inspecting an initial engineering run on the same 500 questions, so
these questions are development material and cannot serve as an untouched confirmatory test.

M1 improves B1's candidate coverage at every budget and improves Recall@20 by 1.0 percentage
point at the primary budget. It does not improve Hit@1. At a budget of 100, B1 and M1 select the
same top cell for all 500 questions. The added structured candidates therefore reach the union,
but the unchanged generic reranker does not move them to rank one. This result supports the M2
gate: test a context-aware reranker before claiming that the full retrieval method improves
wrong-cell selection.

Across the raw route outputs, dense retrieval contains a gold cell for 483 questions, sparse for
481 and structured lookup for 490. At least one route contains a gold cell for all 500 questions.
Structured lookup uniquely covers 10 questions missed by both dense and sparse. Reciprocal-rank
fusion retains a gold cell for 491 questions at budget 20 and 498 questions at budgets 50 and 100;
two available gold candidates are therefore pruned from the top-100 fused union.

## Runtime

| Condition | Index build | Candidate retrieval | Reranking | Total |
|---|---:|---:|---:|---:|
| B0 | 10.27 s | 5.20 s, including evaluation | — | 15.47 s |
| B1 | 37.11 s | 50.66 s | 311.49 s | 399.72 s |
| B2 | 0.12 s | 53.45 s, plus 0.06 s query extraction | 0.16 s | 53.89 s |
| M1 | 39.73 s | Three concurrent routes, plus 0.40 s fusion | 314.68 s | 448.93 s |

The generic reranker consumed 78% of B1 wall time and 70% of M1 wall time. M1's cumulative
per-route retrieval times were 93.28 seconds for dense, 42.71 seconds for sparse and 76.61 seconds
for structured lookup. These route timings overlap because the routes execute concurrently, so
they must not be added to infer wall time. M1 took 12.3% more wall time than the matched B1 run.
Later experiments should report accuracy and cost together.

## B2 error profile

The corrected B2 run made 96 top-rank errors. Among those errors, 90 selected a cell from the same
calendar year, 51 selected the same raw concept and 66 selected the same raw entity. Thirty-nine
errors matched the same raw entity, concept and year simultaneously. Qualitative inspection found
remaining distinctions involving exact dates, reported versus adjusted bases, business segments,
table-axis roles and duplicate common fields with different source-cell identities. This is the
near-miss context that later schema-flexible reranking must address rather than hiding behind
document or table relevance.

The automatic extractor identified an explicit segment, region or product phrase in 23 questions.
None of those phrases appeared as a matching arbitrary dimension in the corresponding fact header
paths. B2 therefore treated them as unknown with no penalty, rather than guessing a field match.
This is both a limitation of the structured baseline and evidence that some decisive context sits
outside the local cell headers in this development corpus.

## Reproducibility stamps

Matched B1 run:

- Base commit: `ef83e4c3a54379d0d54b4007402a12dcd5c9e5d7`.
- Dirty worktree fingerprint: `5fba30d4704a1d44d948aaf0c2bd1986a8b81668b89c9324eb01438406a49508`.
- Resolved configuration hash: `3e537565af1e40d6bd61aff7c8c1870cf09981172c6c1e8cfcbae3f77a6aa797`.
- Result-file SHA-256: `f39f6fcfb064a1ff3828bdb8c09570e6726ef066845a1b718e269a0819946ef2`.
- Result file: `results/runs/b1_development_pilot.json`.

M1 run:

- Base commit: `ef83e4c3a54379d0d54b4007402a12dcd5c9e5d7`.
- Dirty worktree fingerprint: `5fba30d4704a1d44d948aaf0c2bd1986a8b81668b89c9324eb01438406a49508`.
- Resolved configuration hash: `506cccbc6ad58b85c0017946f83ac37178d4114c62719e1c33eb3f1e25b71dd4`.
- Canonical question-content SHA-256: `56427212b320bd908a3ba892ef90dd80b68079a729bdd9579da4372603957946`.
- Canonical fact-content SHA-256: `1c5d8a63700243fc53defe6a5092bfb1f31fff6642cde615e147366b9c883846`.
- Result-file SHA-256: `e212ef232bc37a19adace7bbcecc09932713dc5f569d14e416398f2b42826463`.
- Result file: `results/runs/m1_development_pilot.json`.

Corrected B2 run:

- Base commit: `ef83e4c3a54379d0d54b4007402a12dcd5c9e5d7`.
- Dirty worktree fingerprint: `130df876474a1250327a03910b1deb6663a1a8a7cd114f06db22fcd9be86e155`.
- Resolved configuration hash: `ffcdc87069385409f8838bcddb62cfd8c03e5ed2c39a1b34109dc77e78a23ea9`.
- Canonical question-content SHA-256: `56427212b320bd908a3ba892ef90dd80b68079a729bdd9579da4372603957946`.
- Canonical fact-content SHA-256: `1c5d8a63700243fc53defe6a5092bfb1f31fff6642cde615e147366b9c883846`.
- Result-file SHA-256: `bea57792410815e37ca7aec051b5103bc6af315b0e8f3bdd01f3d8556de3cdfd`.
- Result file: `results/runs/b2_development_pilot.json`.

The worktree fingerprint records the exact tracked and untracked code state without implying that
the current changes were committed or pushed.

## What this establishes

- Complete-table and exact-fact evidence can be indexed and retrieved with stable identities.
- The exact target cell survived parsing, fact construction, serialization, indexing, fusion,
  and reranking.
- B1's remaining failures mostly originate in first-stage candidate generation, not reranking.
- Automatic structured lookup is a viable strong baseline on the closed development corpus.
- M1's structured route improves gold-cell candidate coverage, including 10 questions with no gold
  cell in either dense or sparse retrieval.
- M1's generic reranker does not improve Hit@1 despite the stronger candidate pool, isolating the
  next bottleneck at ranking rather than route availability.
- Structured matching masks query dimensions that are absent from the fact header paths instead of
  turning them into invented constraints.
- Every route, fusion contribution, budget ranking and final ranking is retained in the stamped
  result files for error analysis.

## What this does not establish

- It does not show that B1 outperforms B0 because the success units differ.
- It does not estimate performance on unseen companies, filings, or table structures.
- It does not measure final answer accuracy or hallucination reduction.
- The questions came from the same closed table corpus used for retrieval and may contain
  lexical alignment.
- Approval was first-pass human review; independent blind review is still pending.
- B2 was iterated against these same development questions, so its reported value is unsuitable
  for model selection and final hypothesis testing.
- M1 has not been tested on held-out companies, filings or table-structure families, and its
  matched-budget differences are descriptive rather than confirmatory.

## Next experiment gate

Implement M2 using the same M1 candidate union and train the schema-flexible, context-aware
reranker with ordinary negatives. Keep the held-out company, filing and table-structure splits
sealed, compare M2 against the matched B1, B2 and M1 development conditions, and inspect whether
the added context converts M1's candidate-coverage gain into a Hit@1 gain. M3 natural financial
hard-negative training remains a later ablation.
