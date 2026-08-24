# Week 4 Study Design Freeze

**Project:** Beyond Relevant Tables: Diagnosing and Reducing Wrong-Cell Retrieval in Financial RAG<br>
**Candidate:** Loc Nguyen<br>
**Supervisor:** A/Prof Wei Liu<br>
**Frozen:** 15 August 2026<br>
**Revised contract:** 16 August 2026<br>
**Status:** Approved research direction, as reported by the candidate; detailed method pending supervisor review

## Why this file exists

This is the authoritative specification for the feasibility pilot and scaffold migration. It prevents the project from changing research questions while data suitability and interfaces are being established. The fuller rationale remains in [`research_proposal.md`](research_proposal.md).

If another project document conflicts with this file during the pilot, this file controls. Any later change requires a dated entry in the change log.

## Problem

A financial RAG system can retrieve the correct document or a relevant table but still select the wrong number. Financial tables use heterogeneous layouts and dimensions. Some distinguish periods, units or entities; others introduce table-specific contexts such as product, geography, scenario, maturity band, accounting basis or another locally defined dimension. A fixed schema can lose these distinctions, while an embedding can rank cells as semantically similar even when one decisive context differs.

## Primary research question

> In a large collection of financial filings, does schema-flexible, context-aware exact-cell retrieval reduce wrong-cell retrieval errors compared with conventional hybrid RAG and structured database lookup?

## Generalisation subquestion

> Does the method continue to work on unseen companies, filings, and table structures?

## Unit of analysis

For B1–M3, the unit of analysis is one question paired with the complete set of addressable `FinancialFact` records that validly answer it.

For B0, the retrieval unit is a flattened chunk rather than a cell. B0 therefore receives chunk evidence-coverage and downstream answer metrics. It must not be reported as producing an exact-cell ranking merely because a retrieved chunk contains a correct cell.

## FinancialFact contract

Every table value in the exact-cell conditions is represented as a `FinancialFact` containing:

- entity;
- concept;
- period;
- unit;
- raw and normalised value;
- source and complete provenance;
- stable fact and cell identity;
- complete raw row-header path;
- complete raw column-header path; and
- zero or more arbitrary dimensions represented as name–value pairs.

Canonical dimension names use a versioned normaliser fitted on training data only. Every fact also retains the raw dimension name, raw value and source header path. The arbitrary dimension map is essential: the system must not assume that one fixed list of scope fields covers every table. An unknown or missing dimension remains explicitly unknown; it is never guessed merely to complete a record.

## Primary scope

Include questions where:

- the answer is explicitly present in one financial table cell;
- the source document and complete table can be identified;
- the target row and column can be identified without arithmetic;
- the relevant question and cell contexts can be annotated with reasonable confidence; and
- the evidence comes from a public English-language financial report or filing.

Exclude from the primary exact-cell experiment:

- arithmetic, ratios, aggregation and multi-step programs;
- multi-table reasoning;
- narrative questions without a target numeric cell;
- VLM training or PDF parsing as a research contribution;
- GraphRAG as the proposed architecture;
- claim-level NLI verification, arithmetic verification and conformal abstention;
- production deployment or financial advice.

Excluded items may be counted and described, but they must not be mixed into the confirmatory exact-cell test.

## Datasets

### Primary corpus

Use **T²-RAGBench** from the official Hugging Face repository `G4KMU/t2-ragbench`.

- Use the cleaned 23,088-row release that excludes VQAonBD.
- Pin the immutable revision `adf7fe1541ac37351ce1142544d8e3b43010ed92`.
- Validate all configured row counts, byte counts and file hashes before processing.
- Use its heterogeneous FinQA, ConvFinQA and TAT-DQA table forms to construct the human-verified exact-cell subset and test schema variation.

### External validation

Use **FinanceBench** from Hugging Face repository `PatronusAI/financebench`, configuration `default`, split `train`, pinned at revision `e04404e3a97f69f79c14d42f24981a1c9c3bcd18`. The split contains 150 metadata rows. Filing PDFs are not treated as verified merely because metadata were downloaded; they require separate acquisition and integrity verification before exact-cell annotation.

### Supporting data only

Original FinQA, ConvFinQA and TAT-family annotations may help reconstruct target cells or form training data. Derived questions and documents remain grouped in one split. FinDER is supplementary rather than a primary exact-cell dataset because its released schema lacks exact-cell addresses.

## Experimental conditions

| ID | Condition | Training role | Primary retrieval evaluation |
|---|---|---|---|
| B0 | Flattened chunks with dense + sparse hybrid retrieval | None | Chunk coverage and downstream answer metrics |
| B1 | `FinancialFact` dense + sparse retrieval with a generic reranker | No project training | Exact-cell Hit@1 and related cell metrics |
| B2 | Automatic query context + structured lookup | None | Exact-cell Hit@1 and related cell metrics |
| M1 | Parallel dense, sparse and structured candidate union + generic reranker | No project training | Exact-cell Hit@1 and related cell metrics |
| M2 | M1 + schema-flexible context-aware reranker | Ordinary negatives | Exact-cell Hit@1 and related cell metrics |
| M3 | M2 + natural finance hard negatives | Natural confusing financial cells | Exact-cell Hit@1 and related cell metrics |

M3 is the complete proposed method. M1 isolates candidate union, M2 isolates context-aware reranking and M3 tests whether natural finance hard negatives add value. The co-primary comparisons are M3 versus B1 and M3 versus B2.

Cell-level and graph-based RAG already exist. This study does not claim to be the first system to retrieve table cells. Its contribution is the controlled comparison of a schema-flexible representation and context-aware retrieval against both hybrid RAG and structured lookup, with explicit testing on unseen table structures.

## Evaluation

### B0

- evidence coverage at configured cutoffs;
- context recall at configured cutoffs;
- fixed-generator downstream answer accuracy, if enabled; and
- retrieval and generation cost.

### B1–M3

Primary metric:

- exact-cell Hit@1.

Secondary metrics:

- exact-cell Recall@5, Recall@10 and Recall@20;
- mean reciprocal rank;
- candidate-generation and reranking miss rates;
- wrong-entity, wrong-concept, wrong-period, wrong-unit and wrong-arbitrary-dimension rates;
- deterministic rendered-answer accuracy;
- latency, index size, memory and throughput.

Exact-cell success is determined only by whether the ranked fact's source cell ID belongs to the complete gold source-cell set. Fact IDs are retained as representation lineage, not as an alternative correctness target. If several gold cells are valid, field-error attribution uses the configured closest-gold mismatch rule with stable cell-ID tie-breaking.

B0 and B1 differ in both retrieval unit and reranking, so their difference is descriptive rather than a clean one-component ablation. The confirmatory comparisons remain M3 versus B1 and M3 versus B2.

Splits must be grouped by company and filing. Related questions, source tables, paraphrases and derived negatives remain together. Results must be reported separately for unseen companies, filings and table-structure families.

All cell-ranking conditions use the same configured final reranker input budget. Because M1–M3 deliberately add a third retrieval route, report route-level cost and repeat the main comparison at configured final candidate counts of 20, 50 and 100.

## Week 4 feasibility gate

Before implementing M1–M3, demonstrate that:

1. the pinned T²-RAGBench revision can be acquired reproducibly;
2. its schema, source variations and split counts are recorded;
3. at least 30 direct-value pilot questions can be mapped to exact cells;
4. target cells and natural near misses can be represented as stable `FinancialFact` records without discarding unanticipated dimensions;
5. B0 can run on the pilot and produce per-question chunk rankings and coverage results;
6. B1 can produce genuine cell rankings on the same pilot;
7. annotation time and ambiguity support a 300–500-question exact-cell set.

If these conditions fail, revise the dataset construction or claims before implementing the full method. Do not silently force heterogeneous tables into a fixed schema to pass the gate.

As of 24 August 2026, items 1–4 pass. The table-wide round-trip rebuilt 8,082
numeric `FinancialFact` records without using gold labels as parser inputs and
recovered all 500 approved targets and 1,000 natural alternatives exactly.
Items 5–6 have not been implemented. Item 7 cannot be judged because active
annotation time and blind independent agreement were not collected. The set is
therefore a validated first-pass development artifact, not yet a confirmatory
benchmark.

## Reproducibility rules

- Python 3.11 and `uv` with a pinned lockfile;
- one base YAML merged with one condition or workflow override;
- all model names and revisions, dataset revisions, paths, seeds, budgets, thresholds and training settings in YAML;
- immutable raw data and hashed derived artifacts;
- every result stamped with Git commit, stable configuration hash, UTC timestamp, full resolved configuration, dataset revision and dataset-manifest hash;
- raw per-question rankings and score components retained;
- no paid or moving external API dependency in the primary experiment;
- `make eval` regenerates reported outputs from a fresh clone after data acquisition once the scientific algorithms are implemented.

During the interface-only scaffold phase, unimplemented scientific operations must raise `NotImplementedError`. The scaffold must not create placeholder results that appear to be experiments.

## Change log

| Date | Change | Reason | Approved by |
|---|---|---|---|
| 2026-08-15 | Initial Week 4 design freeze | Begin dataset-feasibility pilot without further architectural expansion | Candidate; supervisor direction reported approved |
| 2026-08-16 | Replaced fixed field routing with schema-flexible `FinancialFact`; separated B0 chunk metrics from exact-cell metrics; clarified M1–M3 training | Prevent false comparability and support unseen table structures | Pending detailed supervisor review |
| 2026-08-17 | Recorded the validated 30-question first pass and its open gates | Separate reproducible cell-label validation from benchmark and scale claims | Primary researcher approved labels; methods gates remain open |
| 2026-08-24 | Added a deterministic, human-reviewable 500-question construction workflow using real tables, source-aligned questions and controlled hard negatives | Reduce annotation burden without treating automatic labels as gold; preserve researcher criticism and later blind review | Primary researcher reviewed and approved 500/500; independent review pending |
| 2026-08-24 | Passed the table-wide `FinancialFact` round-trip on 8,082 facts, including all 500 targets and 1,000 natural hard negatives | Verify that the proposed representation preserves approved evidence before retrieval implementation | Automated reproducibility gate passed; B0/B1 and independent review remain open |
