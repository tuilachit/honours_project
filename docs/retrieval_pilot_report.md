# B0/B1 Development Retrieval Pilot

## Purpose

This pilot checks whether the retrieval interfaces work before implementing the proposed
structured routes. It is a development feasibility test, not a confirmatory experiment.

The run used 500 first-pass-approved synthetic questions grounded in 500 real T²-RAGBench
tables. The fact corpus contained 8,082 losslessly round-tripped `FinancialFact` records.
Every input and result was generated under Git commit
`116ba94e259802dc56301f98f51122456f8a2e6b`.

## Frozen components

- Dense encoder: [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- Generic reranker: [`cross-encoder/ms-marco-MiniLM-L6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2), revision `233902d25c440f23af6f7d6e94d2946bac0bee0a`.
- Sparse route: deterministic BM25.
- Fusion: reciprocal-rank fusion.
- Runtime: CPU, one worker, seeded deterministic inference.

## Conditions

| Condition | Retrieved unit | Meaning of success |
|---|---|---|
| B0 | One complete flattened source table | A retrieved table contains the labelled cell |
| B1 | One `FinancialFact`, followed by a generic reranker | The retrieved fact has the labelled source-cell ID |

B0 deliberately preserves each complete table. It is not a 2,000-character chunking
experiment. B0 and B1 therefore answer different questions and their percentages must not be
subtracted to claim a direct accuracy improvement.

## Results

| Condition and metric | @1 | @5 | @10 | @20 | MRR |
|---|---:|---:|---:|---:|---:|
| B0 table evidence coverage | 44.6% | 72.0% | 80.8% | 87.2% | — |
| B1 exact-cell retrieval | 78.2% | 94.4% | 96.2% | 97.6% | 0.8562 |

B1 stage errors:

- Candidate-generation miss: 2.0%.
- Additional reranking miss: 0.4%.
- Final top-20 miss: 2.4%.

## Runtime

| Condition | Index build | Candidate retrieval | Reranking | Total |
|---|---:|---:|---:|---:|
| B0 | 10.27 s | 5.20 s, including evaluation | — | 15.47 s |
| B1 | 30.92 s | 35.53 s | 260.71 s | 327.40 s |

The generic B1 reranker consumed about 80% of B1 runtime. Later experiments should report
accuracy and cost together.

## What this establishes

- Complete-table and exact-fact evidence can be indexed and retrieved with stable identities.
- The exact target cell survived parsing, fact construction, serialization, indexing, fusion,
  and reranking.
- B1's remaining failures mostly originate in first-stage candidate generation, not reranking.
- Every route and final ranking is retained in the stamped result files for error analysis.

## What this does not establish

- It does not show that B1 outperforms B0 because the success units differ.
- It does not estimate performance on unseen companies, filings, or table structures.
- It does not measure final answer accuracy or hallucination reduction.
- The questions came from the same closed table corpus used for retrieval and may contain
  lexical alignment.
- Approval was first-pass human review; independent blind review is still pending.

## Next experiment gate

Implement B2 as a structured-field lookup baseline over the same `FinancialFact` corpus. Then
M1 can test whether unioning dense, sparse, and structured candidates improves exact-cell
retrieval under a matched candidate budget. Training and M3 hard-negative work should wait
until these non-trained baselines are measured and their errors are inspected.
