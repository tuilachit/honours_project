# Schema-Flexible Exact-Cell Retrieval for Financial RAG

Evaluation scaffold for the honours thesis **Beyond Relevant Tables: Diagnosing and Reducing Wrong-Cell Retrieval in Financial RAG**.

A financial RAG system may retrieve a relevant filing or table but still select the wrong number. The error can arise from a small difference in context, such as a different reporting period, unit, entity, segment, accounting basis, or a table-specific dimension that was not known when the system was designed.

## Research question

> In a large collection of financial filings, does schema-flexible, context-aware exact-cell retrieval reduce wrong-cell retrieval errors compared with conventional hybrid RAG and structured database lookup?

The generalisation subquestion is:

> Does the method continue to work on unseen companies, filings, and table structures?

## Core representation

Each addressable table value is represented as a `FinancialFact` containing:

- entity, concept, period, unit and value;
- source and complete provenance;
- complete raw row-header and column-header paths; and
- arbitrary additional dimensions stored as name–value pairs.

Canonical dimension names are versioned and learned from training data only, while the original dimension names, values and header paths are retained. This avoids forcing every financial table into one fixed list of fields. Unknown dimensions remain explicit rather than being guessed or discarded.

## Experimental conditions

- **B0:** flattened chunks with dense and sparse hybrid retrieval.
- **B1:** `FinancialFact` retrieval with dense and sparse search plus a generic reranker.
- **B2:** automatic query-context extraction plus structured database lookup.
- **M1:** parallel dense, sparse and structured candidate retrieval with a generic reranker and no project-specific training.
- **M2:** M1 plus a schema-flexible, context-aware reranker trained with ordinary negatives.
- **M3:** M2 trained with naturally confusing financial hard negatives; this is the complete proposed method.

B0 does not pretend that a multi-cell chunk is an exact-cell ranking. It is evaluated using chunk evidence coverage and a fixed chunk-to-answer generator. B1–M3 return addressable `FinancialFact` candidates and are evaluated using exact-cell Hit@1 and related cell-level metrics. The same pinned generator and prompt provide the optional downstream comparison; deterministic fact rendering is a separate B1–M3 diagnostic.

## Data

The primary corpus is the cleaned T²-RAGBench release, pinned in [`configs/datasets/t2_ragbench.yaml`](configs/datasets/t2_ragbench.yaml). FinanceBench is a pinned external validation set described in [`configs/datasets/financebench.yaml`](configs/datasets/financebench.yaml). Its filing PDFs require separate acquisition and integrity verification before exact-cell annotation.

## Repository status

This repository is currently an **experiment scaffold with implemented data-audit, pilot-validation, prospective-screening and synthetic-candidate construction workflows**. The primary researcher completed first-pass review of all 500 generated candidates and approved 500/500; an independent blind review sample remains required before confirmatory use. The reproducibility contracts, condition configurations and typed interfaces define the intended study, but retrieval, reranking, training, evaluation and figure-generation algorithms intentionally raise `NotImplementedError` until future implementation tasks.

The T²-RAGBench feasibility audit validates the pinned metadata files. The exact-cell pilot validator then checks the first-pass annotation file against those raw records, verifies every table hash and source-cell value, derives stable cell IDs, and writes a provenance-stamped result. The current pilot is development material with open feasibility gates; see [`docs/pilot_feasibility_report.md`](docs/pilot_feasibility_report.md).

## Project layout

- `src/data/`: deduplicated T²-RAGBench and FinanceBench corpus loaders and grouped splits;
- `src/tables/`: structure-preserving parsing and normalisation contracts;
- `src/facts/`: `FinancialFact` construction, evidence packaging and serialisation;
- `src/query/`: schema-flexible query-context extraction;
- `src/retrieval/`: dense, sparse, structured, fusion, reranking and training contracts;
- `src/generation/`: deterministic fact rendering and fixed-prompt generation;
- `src/eval/`: exact-cell, chunk-coverage, dynamic field-error, efficiency and statistical metrics;
- `configs/`: shared parameters, six scientific conditions, workflows and pinned manifests;
- `scripts/`: reproducible command-line entrypoints; and
- `tests/` and `results/`: contract checks and provenance-stamped outputs.

## Reproducibility contract

- Python 3.11 with dependencies managed by `uv` and pinned in `uv.lock`;
- one base YAML merged with one condition or workflow YAML;
- all paths, seeds, budgets, thresholds, models and revisions held in configuration;
- immutable dataset revisions and file manifests;
- stable resolved-configuration hashes; and
- result envelopes stamped with provenance and the full resolved configuration.

The six scientific profiles live in `configs/`. Operational overrides for data acquisition, the pilot and figures live in `configs/workflows/`.

```sh
make setup
make test
make audit-data
make pilot
make review-pack
make screening-batch
make synthetic-500
make fact-roundtrip
```

The run and evaluation targets are present as explicit workflow contracts. They will become executable end to end only after the intentionally stubbed algorithms are implemented; the scaffold does not generate placeholder scientific results. Raw data and generated result envelopes remain gitignored, while the pinned manifests, first-pass annotations, fixed prospective screening log and methods report are version controlled.
