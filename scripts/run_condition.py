"""Run one retrieval condition over the approved development stress set."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import TypeVar, cast

import numpy as np
import torch

from src.config import Config, load_config
from src.eval.retrieval import (
    compute_chunk_evidence_coverage,
    compute_exact_cell_metrics,
    compute_stage_miss_metrics,
)
from src.facts.evidence import build_fact_evidence, build_flattened_evidence
from src.facts.serialize import deserialize_financial_fact
from src.hashing import sha256_json
from src.query.extract import extract_query_context
from src.results import write_result_json
from src.retrieval.dense import build_dense_index, retrieve_dense
from src.retrieval.fusion import (
    fuse_fact_candidates,
    fuse_route_candidates,
    project_fact_candidates,
)
from src.retrieval.reranker import rerank_generic_budgets
from src.retrieval.sparse import build_sparse_index, retrieve_sparse
from src.retrieval.structured import (
    build_structured_index,
    rank_structured_lookup,
    retrieve_structured,
)
from src.types import (
    FactCandidate,
    FinancialFact,
    GoldCellLabel,
    IndexArtifact,
    QueryContext,
    Question,
    RankedFact,
    RetrievalCandidate,
    RetrievalRoute,
    SourceTable,
)

REPOSITORY = Path(__file__).resolve().parents[1]
T = TypeVar("T")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _string(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _integer(mapping: Mapping[str, object], key: str) -> int:
    value = mapping.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return value


def _config_section(config: Config, key: str) -> Mapping[str, object]:
    return _mapping(config.get(key), key)


def _resolve(path_value: object, label: str) -> Path:
    if not isinstance(path_value, str):
        raise ValueError(f"{label} must be a path string")
    path = Path(path_value)
    return path if path.is_absolute() else REPOSITORY / path


def _read_result(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    envelope = _mapping(payload, str(path))
    return _mapping(envelope.get("result"), f"{path} result")


def _load_inputs(
    config: Config,
) -> tuple[
    list[Question],
    list[SourceTable],
    list[FinancialFact],
    dict[str, GoldCellLabel],
    dict[str, str],
]:
    pilot = _config_section(config, "retrieval_pilot")
    question_path = _resolve(pilot.get("input_questions_path"), "input_questions_path")
    fact_path = _resolve(pilot.get("input_facts_path"), "input_facts_path")
    questions_result = _read_result(question_path)
    facts_result = _read_result(fact_path)
    if questions_result.get("status") != pilot.get("required_question_status"):
        raise ValueError("Question artifact has not reached the required review status")
    if facts_result.get("status") != pilot.get("required_fact_status"):
        raise ValueError("Fact artifact has not passed the required round-trip status")
    raw_questions = questions_result.get("questions")
    raw_facts = facts_result.get("facts")
    if not isinstance(raw_questions, list) or not isinstance(raw_facts, list):
        raise ValueError("Input artifacts do not contain question and fact lists")
    if len(raw_questions) != _integer(pilot, "expected_question_count"):
        raise ValueError("Question count differs from the configured pilot contract")
    if len(raw_facts) != _integer(pilot, "expected_fact_count"):
        raise ValueError("Fact count differs from the configured pilot contract")
    facts = [
        deserialize_financial_fact(_mapping(item, "serialized fact"), config) for item in raw_facts
    ]
    facts_by_source: dict[tuple[str, int], list[FinancialFact]] = defaultdict(list)
    for fact in facts:
        context_id = fact.source_address.context_id
        table_index = fact.source_address.metadata.get("table_index")
        if context_id is None or not isinstance(table_index, int):
            raise ValueError("Every pilot fact must retain context and table identity")
        facts_by_source[(context_id, table_index)].append(fact)

    questions: list[Question] = []
    tables: list[SourceTable] = []
    labels: dict[str, GoldCellLabel] = {}
    for raw_question in raw_questions:
        item = _mapping(raw_question, "synthetic question")
        source = _mapping(item.get("source"), "question source")
        target = _mapping(item.get("target_cell"), "target cell")
        question_id = _string(item, "question_id")
        context_id = _string(source, "context_id")
        table_index = _integer(source, "table_index")
        table_facts = facts_by_source.get((context_id, table_index), [])
        if not table_facts:
            raise ValueError(f"No FinancialFacts found for {question_id}")
        table_ids = {fact.source_address.table_id for fact in table_facts}
        if len(table_ids) != 1 or None in table_ids:
            raise ValueError(f"Ambiguous table identity for {question_id}")
        page_raw = source.get("page_number")
        page_number = int(page_raw) if isinstance(page_raw, str | int) else None
        table_id = cast(str, next(iter(table_ids)))
        tables.append(
            SourceTable(
                dataset=_string(source, "dataset"),
                raw_text=_string(item, "source_table_markdown"),
                table_id=table_id,
                document_id=_string(source, "file_name"),
                context_id=context_id,
                page_number=page_number,
                source_format="markdown",
                metadata={
                    "fact_ids": [fact.fact_id for fact in table_facts],
                    "cell_ids": [fact.source_address.cell_id for fact in table_facts],
                    "subset": _string(source, "subset"),
                    "table_index": table_index,
                },
            )
        )
        questions.append(
            Question(
                question_id=question_id,
                dataset=_string(source, "dataset"),
                text=_string(item, "question"),
                gold_answer=_string(item, "answer"),
                relevant_document_ids=(_string(source, "file_name"),),
                metadata={"context_id": context_id, "table_id": table_id},
            )
        )
        labels[question_id] = GoldCellLabel(
            question_id=question_id,
            valid_cell_ids=(_string(target, "cell_id"),),
            annotator_id="first_pass_human_review",
            adjudicated=False,
            metadata={"evaluation_use": "project_development_only"},
        )
    if len(tables) != _integer(pilot, "expected_table_count"):
        raise ValueError("Source-table count differs from the configured pilot contract")
    return questions, tables, facts, labels, {
        "questions": sha256_json(raw_questions),
        "facts": sha256_json(raw_facts),
    }


def _candidate_trace(candidate: RetrievalCandidate, rank: int) -> Mapping[str, object]:
    singular_fact_id = (
        candidate.evidence.fact_ids[0] if len(candidate.evidence.fact_ids) == 1 else None
    )
    return {
        "rank": rank,
        "evidence_id": candidate.evidence.evidence_id,
        "fact_id": singular_fact_id,
        "contained_fact_count": len(candidate.evidence.fact_ids),
        "cell_id": candidate.evidence.metadata.get("cell_id"),
        "table_id": candidate.evidence.metadata.get("table_id"),
        "route_scores": {route.value: score for route, score in candidate.route_scores.items()},
        "route_ranks": {route.value: value for route, value in candidate.route_ranks.items()},
        "fusion_score": candidate.metadata.get("fusion_score"),
    }


def _ranked_trace(ranked: RankedFact) -> Mapping[str, object]:
    return {
        "rank": ranked.rank,
        "fact_id": ranked.fact.fact_id,
        "cell_id": ranked.fact.source_address.cell_id,
        "raw_value": ranked.fact.raw_value,
        "score": ranked.score,
        "component_scores": dict(ranked.component_scores),
    }


def _fact_candidate_trace(
    candidate: FactCandidate,
    rank: int | None = None,
) -> Mapping[str, object]:
    trace: dict[str, object] = {
        "rank": (
            rank
            if rank is not None
            else candidate.route_ranks.get(RetrievalRoute.STRUCTURED)
        ),
        "evidence_id": candidate.evidence_id,
        "fact_id": candidate.fact.fact_id,
        "cell_id": candidate.fact.source_address.cell_id,
        "raw_value": candidate.fact.raw_value,
        "route_scores": {route.value: score for route, score in candidate.route_scores.items()},
        "route_ranks": {route.value: rank for route, rank in candidate.route_ranks.items()},
    }
    structured_components = candidate.metadata.get("structured_components")
    route_metadata = candidate.metadata.get("route_metadata")
    if structured_components is None and isinstance(route_metadata, Mapping):
        structured_metadata = route_metadata.get(RetrievalRoute.STRUCTURED.value)
        if isinstance(structured_metadata, Mapping):
            structured_components = structured_metadata.get("structured_components")
    if structured_components is not None:
        trace["structured_components"] = structured_components
    return trace


def _query_context_trace(query_context: QueryContext) -> Mapping[str, object]:
    return {
        "question_id": query_context.question_id,
        "constraints": [
            {
                "name": constraint.name,
                "state": constraint.state.value,
                "value": constraint.value,
                "raw_name": constraint.raw_name,
                "raw_value": constraint.raw_value,
                "normalization_version": constraint.normalization_version,
                "aliases": list(constraint.aliases),
                "metadata": dict(constraint.metadata),
            }
            for constraint in query_context.constraints
        ],
        "metadata": dict(query_context.metadata),
    }


def _aggregate(per_question: Sequence[Mapping[str, float]]) -> Mapping[str, float]:
    keys = sorted({key for metrics in per_question for key in metrics})
    return {key: fmean(metrics[key] for metrics in per_question if key in metrics) for key in keys}


def _route_rankings(
    question: Question,
    dense_index: IndexArtifact,
    sparse_index: IndexArtifact,
    config: Config,
) -> tuple[list[RetrievalCandidate], list[RetrievalCandidate], list[RetrievalCandidate]]:
    dense = retrieve_dense(question, dense_index, config)
    sparse = retrieve_sparse(question, sparse_index, config)
    fused = fuse_route_candidates(
        {RetrievalRoute.DENSE: dense, RetrievalRoute.SPARSE: sparse}, config
    )
    return dense, sparse, fused


def _run_b0(
    questions: Sequence[Question],
    tables: Sequence[SourceTable],
    labels: Mapping[str, GoldCellLabel],
    config: Config,
) -> Mapping[str, object]:
    started = perf_counter()
    evidence = list(build_flattened_evidence(tables, config))
    dense_index = build_dense_index(evidence, config)
    sparse_index = build_sparse_index(evidence, config)
    index_seconds = perf_counter() - started
    retrieval_started = perf_counter()
    traces: list[Mapping[str, object]] = []
    metric_rows: list[Mapping[str, float]] = []
    for question in questions:
        dense, sparse, fused = _route_rankings(question, dense_index, sparse_index, config)
        metrics = compute_chunk_evidence_coverage(fused, labels[question.question_id], config)
        metric_rows.append(metrics)
        traces.append(
            {
                "question_id": question.question_id,
                "metrics": metrics,
                "dense": [_candidate_trace(item, rank) for rank, item in enumerate(dense, 1)],
                "sparse": [_candidate_trace(item, rank) for rank, item in enumerate(sparse, 1)],
                "fused": [_candidate_trace(item, rank) for rank, item in enumerate(fused, 1)],
            }
        )
    return {
        "evidence_count": len(evidence),
        "aggregate_metrics": _aggregate(metric_rows),
        "question_rankings": traces,
        "index_artifacts": {"dense": dense_index, "sparse": sparse_index},
        "timings_seconds": {
            "index_build": index_seconds,
            "retrieval_and_evaluation": perf_counter() - retrieval_started,
            "total": perf_counter() - started,
        },
    }


def _run_b1(
    questions: Sequence[Question],
    facts: Sequence[FinancialFact],
    labels: Mapping[str, GoldCellLabel],
    config: Config,
) -> Mapping[str, object]:
    started = perf_counter()
    evidence = list(build_fact_evidence(facts, config))
    facts_by_id = {fact.fact_id: fact for fact in facts}
    dense_index = build_dense_index(evidence, config)
    sparse_index = build_sparse_index(evidence, config)
    index_seconds = perf_counter() - started
    budgets = _sensitivity_budgets(config)
    primary_budget = max(budgets)
    retrieval_seconds = 0.0
    reranking_seconds = 0.0
    traces: list[Mapping[str, object]] = []
    metrics_by_budget: dict[int, list[Mapping[str, float]]] = {
        budget: [] for budget in budgets
    }
    for question in questions:
        route_started = perf_counter()
        dense, sparse, fused = _route_rankings(question, dense_index, sparse_index, config)
        retrieval_seconds += perf_counter() - route_started
        projected = project_fact_candidates(fused, facts_by_id, config)
        rerank_started = perf_counter()
        rankings = rerank_generic_budgets(question, projected, budgets, config)
        reranking_seconds += perf_counter() - rerank_started
        budget_metrics: dict[int, Mapping[str, float]] = {}
        for budget in budgets:
            metrics = dict(
                compute_exact_cell_metrics(
                    rankings[budget],
                    labels[question.question_id],
                    config,
                )
            )
            metrics.update(
                compute_stage_miss_metrics(
                    projected[:budget],
                    rankings[budget],
                    labels[question.question_id],
                    config,
                )
            )
            metrics_by_budget[budget].append(metrics)
            budget_metrics[budget] = metrics
        traces.append(
            {
                "question_id": question.question_id,
                "metrics": budget_metrics[primary_budget],
                "budget_metrics": budget_metrics,
                "dense": [_candidate_trace(item, rank) for rank, item in enumerate(dense, 1)],
                "sparse": [_candidate_trace(item, rank) for rank, item in enumerate(sparse, 1)],
                "fused": [_candidate_trace(item, rank) for rank, item in enumerate(fused, 1)],
                "reranked": [_ranked_trace(item) for item in rankings[primary_budget]],
                "budget_rankings": {
                    budget: [_ranked_trace(item) for item in rankings[budget]]
                    for budget in budgets
                },
            }
        )
    aggregate_by_budget = {
        budget: _aggregate(metrics_by_budget[budget]) for budget in budgets
    }
    return {
        "evidence_count": len(evidence),
        "aggregate_metrics": aggregate_by_budget[primary_budget],
        "budget_metrics": aggregate_by_budget,
        "question_rankings": traces,
        "index_artifacts": {"dense": dense_index, "sparse": sparse_index},
        "timings_seconds": {
            "index_build": index_seconds,
            "candidate_retrieval": retrieval_seconds,
            "reranking": reranking_seconds,
            "total": perf_counter() - started,
        },
    }


def _run_b2(
    questions: Sequence[Question],
    facts: Sequence[FinancialFact],
    labels: Mapping[str, GoldCellLabel],
    config: Config,
) -> Mapping[str, object]:
    started = perf_counter()
    facts_by_id = {fact.fact_id: fact for fact in facts}
    structured_index = build_structured_index(facts, config)
    index_seconds = perf_counter() - started
    extraction_seconds = 0.0
    retrieval_seconds = 0.0
    reranking_seconds = 0.0
    traces: list[Mapping[str, object]] = []
    metric_rows: list[Mapping[str, float]] = []
    for question in questions:
        extraction_started = perf_counter()
        query_context = extract_query_context(question, config)
        extraction_seconds += perf_counter() - extraction_started
        retrieval_started = perf_counter()
        structured = retrieve_structured(
            question,
            query_context,
            structured_index,
            facts_by_id,
            config,
        )
        retrieval_seconds += perf_counter() - retrieval_started
        rerank_started = perf_counter()
        reranked = rank_structured_lookup(question, query_context, structured, config)
        reranking_seconds += perf_counter() - rerank_started
        metrics = dict(compute_exact_cell_metrics(reranked, labels[question.question_id], config))
        metrics.update(
            compute_stage_miss_metrics(
                structured,
                reranked,
                labels[question.question_id],
                config,
            )
        )
        metric_rows.append(metrics)
        traces.append(
            {
                "question_id": question.question_id,
                "query_context": _query_context_trace(query_context),
                "metrics": metrics,
                "structured": [_fact_candidate_trace(item) for item in structured],
                "reranked": [_ranked_trace(item) for item in reranked],
            }
        )
    return {
        "evidence_count": len(facts),
        "aggregate_metrics": _aggregate(metric_rows),
        "question_rankings": traces,
        "index_artifacts": {"structured": structured_index},
        "timings_seconds": {
            "index_build": index_seconds,
            "query_context_extraction": extraction_seconds,
            "candidate_retrieval": retrieval_seconds,
            "reranking": reranking_seconds,
            "total": perf_counter() - started,
        },
    }


def _sensitivity_budgets(config: Config) -> tuple[int, ...]:
    retrieval = _config_section(config, "retrieval")
    sensitivity = _mapping(
        retrieval.get("budget_matched_sensitivity"),
        "retrieval.budget_matched_sensitivity",
    )
    values = sensitivity.get("final_candidate_counts")
    if sensitivity.get("enabled") is not True or not isinstance(values, list):
        raise ValueError("M1 requires enabled budget-matched sensitivity")
    if not values or not all(
        isinstance(value, int) and not isinstance(value, bool) for value in values
    ):
        raise ValueError("M1 sensitivity budgets must be an integer list")
    budgets = tuple(cast(list[int], values))
    if any(value <= 0 for value in budgets) or len(set(budgets)) != len(budgets):
        raise ValueError("M1 sensitivity budgets must be unique positive integers")
    return budgets


def _timed_call(operation: Callable[[], T]) -> tuple[T, float]:
    started = perf_counter()
    return operation(), perf_counter() - started


def _retrieve_m1_routes(
    question: Question,
    query_context: QueryContext,
    dense_index: IndexArtifact,
    sparse_index: IndexArtifact,
    structured_index: IndexArtifact,
    facts_by_id: Mapping[str, FinancialFact],
    config: Config,
) -> tuple[
    list[RetrievalCandidate],
    list[RetrievalCandidate],
    list[FactCandidate],
    Mapping[str, float],
]:
    with ThreadPoolExecutor(max_workers=3) as executor:
        dense_future = executor.submit(
            _timed_call,
            lambda: retrieve_dense(question, dense_index, config),
        )
        sparse_future = executor.submit(
            _timed_call,
            lambda: retrieve_sparse(question, sparse_index, config),
        )
        structured_future = executor.submit(
            _timed_call,
            lambda: retrieve_structured(
                question,
                query_context,
                structured_index,
                facts_by_id,
                config,
            ),
        )
        dense, dense_seconds = dense_future.result()
        sparse, sparse_seconds = sparse_future.result()
        structured, structured_seconds = structured_future.result()
    return dense, sparse, structured, {
        "dense": dense_seconds,
        "sparse": sparse_seconds,
        "structured": structured_seconds,
    }


def _run_m1(
    questions: Sequence[Question],
    facts: Sequence[FinancialFact],
    labels: Mapping[str, GoldCellLabel],
    config: Config,
) -> Mapping[str, object]:
    started = perf_counter()
    evidence = list(build_fact_evidence(facts, config))
    facts_by_id = {fact.fact_id: fact for fact in facts}

    route_started = perf_counter()
    dense_index = build_dense_index(evidence, config)
    dense_index_seconds = perf_counter() - route_started
    route_started = perf_counter()
    sparse_index = build_sparse_index(evidence, config)
    sparse_index_seconds = perf_counter() - route_started
    route_started = perf_counter()
    structured_index = build_structured_index(facts, config)
    structured_index_seconds = perf_counter() - route_started
    index_seconds = dense_index_seconds + sparse_index_seconds + structured_index_seconds

    budgets = _sensitivity_budgets(config)
    primary_budget = max(budgets)
    extraction_seconds = 0.0
    dense_seconds = 0.0
    sparse_seconds = 0.0
    structured_seconds = 0.0
    union_seconds = 0.0
    reranking_seconds = 0.0
    candidate_counts: dict[str, list[int]] = {
        "dense": [],
        "sparse": [],
        "structured": [],
        "fused": [],
    }
    route_presence_counts: dict[str, int] = {
        "dense": 0,
        "sparse": 0,
        "structured": 0,
    }
    route_overlap_counts: defaultdict[str, int] = defaultdict(int)
    gold_route_coverage: dict[str, int] = {
        "dense": 0,
        "sparse": 0,
        "structured": 0,
        "any_route": 0,
        "structured_only_vs_dense_sparse": 0,
        **{f"fused_at_{budget}": 0 for budget in budgets},
    }
    traces: list[Mapping[str, object]] = []
    metrics_by_budget: dict[int, list[Mapping[str, float]]] = {
        budget: [] for budget in budgets
    }
    for question in questions:
        extraction_started = perf_counter()
        query_context = extract_query_context(question, config)
        extraction_seconds += perf_counter() - extraction_started

        dense, sparse, structured, route_timings = _retrieve_m1_routes(
            question,
            query_context,
            dense_index,
            sparse_index,
            structured_index,
            facts_by_id,
            config,
        )
        dense_seconds += route_timings["dense"]
        sparse_seconds += route_timings["sparse"]
        structured_seconds += route_timings["structured"]

        union_started = perf_counter()
        dense_facts = project_fact_candidates(dense, facts_by_id, config)
        sparse_facts = project_fact_candidates(sparse, facts_by_id, config)
        fused = fuse_fact_candidates(
            {
                RetrievalRoute.DENSE: dense_facts,
                RetrievalRoute.SPARSE: sparse_facts,
                RetrievalRoute.STRUCTURED: structured,
            },
            config,
        )
        union_seconds += perf_counter() - union_started

        rerank_started = perf_counter()
        rankings = rerank_generic_budgets(question, fused, budgets, config)
        reranking_seconds += perf_counter() - rerank_started
        budget_metrics: dict[int, Mapping[str, float]] = {}
        for budget in budgets:
            metrics = dict(
                compute_exact_cell_metrics(
                    rankings[budget],
                    labels[question.question_id],
                    config,
                )
            )
            metrics.update(
                compute_stage_miss_metrics(
                    fused[:budget],
                    rankings[budget],
                    labels[question.question_id],
                    config,
                )
            )
            metrics_by_budget[budget].append(metrics)
            budget_metrics[budget] = metrics

        candidate_counts["dense"].append(len(dense))
        candidate_counts["sparse"].append(len(sparse))
        candidate_counts["structured"].append(len(structured))
        candidate_counts["fused"].append(len(fused))
        gold_cells = set(labels[question.question_id].valid_cell_ids)
        route_cell_ids = {
            RetrievalRoute.DENSE: {
                candidate.fact.source_address.cell_id for candidate in dense_facts
            },
            RetrievalRoute.SPARSE: {
                candidate.fact.source_address.cell_id for candidate in sparse_facts
            },
            RetrievalRoute.STRUCTURED: {
                candidate.fact.source_address.cell_id for candidate in structured
            },
        }
        route_gold_hits = {
            route: bool(gold_cells.intersection(cell_ids))
            for route, cell_ids in route_cell_ids.items()
        }
        for route, hit in route_gold_hits.items():
            gold_route_coverage[route.value] += int(hit)
        gold_route_coverage["any_route"] += int(any(route_gold_hits.values()))
        gold_route_coverage["structured_only_vs_dense_sparse"] += int(
            route_gold_hits[RetrievalRoute.STRUCTURED]
            and not route_gold_hits[RetrievalRoute.DENSE]
            and not route_gold_hits[RetrievalRoute.SPARSE]
        )
        for budget in budgets:
            fused_cells = {
                candidate.fact.source_address.cell_id for candidate in fused[:budget]
            }
            gold_route_coverage[f"fused_at_{budget}"] += int(
                bool(gold_cells.intersection(fused_cells))
            )
        for candidate in fused:
            routes = set(candidate.route_ranks)
            for route in (
                RetrievalRoute.DENSE,
                RetrievalRoute.SPARSE,
                RetrievalRoute.STRUCTURED,
            ):
                route_presence_counts[route.value] += int(route in routes)
            route_overlap_counts["+".join(sorted(route.value for route in routes))] += 1
        traces.append(
            {
                "question_id": question.question_id,
                "query_context": _query_context_trace(query_context),
                "metrics": budget_metrics[primary_budget],
                "budget_metrics": budget_metrics,
                "dense": [_candidate_trace(item, rank) for rank, item in enumerate(dense, 1)],
                "sparse": [
                    _candidate_trace(item, rank) for rank, item in enumerate(sparse, 1)
                ],
                "structured": [
                    _fact_candidate_trace(item, rank)
                    for rank, item in enumerate(structured, 1)
                ],
                "fused": [
                    _fact_candidate_trace(item, rank) for rank, item in enumerate(fused, 1)
                ],
                "budget_rankings": {
                    budget: [_ranked_trace(item) for item in rankings[budget]]
                    for budget in budgets
                },
            }
        )

    aggregate_by_budget = {
        budget: _aggregate(metrics_by_budget[budget]) for budget in budgets
    }
    return {
        "evidence_count": len(evidence),
        "aggregate_metrics": aggregate_by_budget[primary_budget],
        "budget_metrics": aggregate_by_budget,
        "mean_candidate_counts": {
            route: fmean(counts) for route, counts in candidate_counts.items()
        },
        "route_presence_counts": route_presence_counts,
        "route_overlap_counts": dict(sorted(route_overlap_counts.items())),
        "gold_route_coverage": gold_route_coverage,
        "question_rankings": traces,
        "index_artifacts": {
            "dense": dense_index,
            "sparse": sparse_index,
            "structured": structured_index,
        },
        "timings_seconds": {
            "index_build": index_seconds,
            "dense_index_build": dense_index_seconds,
            "sparse_index_build": sparse_index_seconds,
            "structured_index_build": structured_index_seconds,
            "query_context_extraction": extraction_seconds,
            "dense_retrieval": dense_seconds,
            "sparse_retrieval": sparse_seconds,
            "structured_retrieval": structured_seconds,
            "candidate_union": union_seconds,
            "reranking": reranking_seconds,
            "total": perf_counter() - started,
        },
    }


def _set_reproducible_runtime(config: Config) -> None:
    project = _config_section(config, "project")
    runtime = _config_section(config, "runtime")
    seed = _integer(project, "seed")
    workers = _integer(runtime, "workers")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(workers)
    torch.use_deterministic_algorithms(runtime.get("deterministic_algorithms") is True)


def _validate_execution_contract(config: Config, condition_id: str) -> None:
    """Fail rather than stamp a config that the pilot did not actually execute."""

    retrieval = _config_section(config, "retrieval")
    routes = _mapping(retrieval.get("routes"), "retrieval.routes")
    union = _mapping(retrieval.get("candidate_union"), "retrieval.candidate_union")
    if _config_section(config, "indexes").get("overwrite") is not True:
        raise NotImplementedError("Pilot index builders require indexes.overwrite: true")
    reranker = _config_section(config, "reranker")
    if condition_id in {"b0", "b1"}:
        expected_routes = {"dense": True, "sparse": True, "structured": False}
        if dict(routes) != expected_routes:
            raise NotImplementedError(f"Pilot routes must equal {expected_routes}")
        dense = _mapping(retrieval.get("dense"), "retrieval.dense")
        sparse = _mapping(retrieval.get("sparse"), "retrieval.sparse")
        if (
            dense.get("similarity") != "cosine"
            or dense.get("normalize_embeddings") is not True
        ):
            raise NotImplementedError(
                "Pilot dense retrieval requires normalized cosine similarity"
            )
        if sparse.get("algorithm") != "bm25":
            raise NotImplementedError("Pilot sparse retrieval requires BM25")
        if union.get("enabled") is not True or union.get("strategy") != "reciprocal_rank_fusion":
            raise NotImplementedError("Pilot candidate union requires reciprocal-rank fusion")
        expected_reranker = (
            (False, "none") if condition_id == "b0" else (True, "generic_cross_encoder")
        )
    elif condition_id == "b2":
        expected_routes = {"dense": False, "sparse": False, "structured": True}
        if dict(routes) != expected_routes:
            raise NotImplementedError(f"B2 routes must equal {expected_routes}")
        if union.get("enabled") is not False or union.get("strategy") != "structured_lookup":
            raise NotImplementedError("B2 requires direct structured lookup")
        structured = _mapping(retrieval.get("structured"), "retrieval.structured")
        if (
            structured.get("ranking_strategy")
            != "concept_then_context_compatibility"
            or structured.get("unknown_field_policy") != "no_penalty"
        ):
            raise NotImplementedError(
                "B2 structured lookup policy must use the implemented ranking and "
                "unknown-field behavior"
            )
        query_context = _config_section(config, "query_context")
        extractor = _mapping(query_context.get("extractor"), "query_context.extractor")
        if (
            query_context.get("enabled") is not True
            or query_context.get("mode") != "automatic"
            or extractor.get("implementation") != "deterministic_regex_v1"
            or extractor.get("revision") != "1"
        ):
            raise NotImplementedError("B2 requires the pinned automatic query extractor")
        expected_reranker = (True, "structured_metric_matcher")
    elif condition_id == "m1":
        expected_routes = {"dense": True, "sparse": True, "structured": True}
        if dict(routes) != expected_routes:
            raise NotImplementedError(f"M1 routes must equal {expected_routes}")
        if (
            union.get("enabled") is not True
            or union.get("strategy") != "reciprocal_rank_fusion"
            or union.get("deduplicate_by") != "stable_source_cell_id"
            or union.get("top_k") != 100
        ):
            raise NotImplementedError("M1 requires the frozen source-cell RRF union")
        dense = _mapping(retrieval.get("dense"), "retrieval.dense")
        sparse = _mapping(retrieval.get("sparse"), "retrieval.sparse")
        structured = _mapping(retrieval.get("structured"), "retrieval.structured")
        if (
            dense.get("similarity") != "cosine"
            or dense.get("normalize_embeddings") is not True
            or sparse.get("algorithm") != "bm25"
            or structured.get("ranking_strategy")
            != "concept_then_context_compatibility"
            or structured.get("unknown_field_policy") != "no_penalty"
        ):
            raise NotImplementedError("M1 requires the frozen three-route implementations")
        query_context = _config_section(config, "query_context")
        extractor = _mapping(query_context.get("extractor"), "query_context.extractor")
        if (
            query_context.get("enabled") is not True
            or query_context.get("mode") != "automatic"
            or extractor.get("implementation") != "deterministic_regex_v1"
            or extractor.get("revision") != "1"
        ):
            raise NotImplementedError("M1 requires the pinned automatic query extractor")
        budgets = _sensitivity_budgets(config)
        if budgets != (20, 50, 100):
            raise NotImplementedError("M1 requires candidate budgets 20, 50 and 100")
        if reranker.get("input_top_k") != 100 or reranker.get("output_top_k") != 20:
            raise NotImplementedError("M1 requires a 100-input, 20-output generic reranker")
        training = _config_section(config, "training")
        generation = _config_section(config, "generation")
        evaluation = _config_section(config, "evaluation")
        if (
            training.get("enabled") is not False
            or training.get("project_training") is not False
            or training.get("negative_strategy") != "none"
            or generation.get("llm_enabled") is not False
            or evaluation.get("downstream_answer_metrics_enabled") is not False
        ):
            raise NotImplementedError("M1 requires the retrieval-only no-training contract")
        expected_reranker = (True, "generic_cross_encoder")
    else:
        raise NotImplementedError("Only B0, B1, B2 and M1 development pilots are implemented")
    if (reranker.get("enabled"), reranker.get("kind")) != expected_reranker:
        raise NotImplementedError(f"{condition_id} reranker contract is {expected_reranker}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--condition-config", required=True, type=Path)
    return parser


def run_condition(config: Config) -> Path:
    """Execute an implemented development condition under the validity boundary."""

    if "retrieval_pilot" not in config:
        raise NotImplementedError("A configured development retrieval pilot is required")
    pilot = _config_section(config, "retrieval_pilot")
    if pilot.get("enabled") is not True:
        raise ValueError("retrieval_pilot.enabled must be true")
    if pilot.get("confirmatory_claims_allowed") is not False:
        raise ValueError("The approved stress set is development-only, not confirmatory")
    condition_id = _string(_config_section(config, "condition"), "id")
    _validate_execution_contract(config, condition_id)
    _set_reproducible_runtime(config)
    questions, tables, facts, labels, input_hashes = _load_inputs(config)
    if condition_id == "b0":
        result = _run_b0(questions, tables, labels, config)
    elif condition_id == "b1":
        result = _run_b1(questions, facts, labels, config)
    elif condition_id == "b2":
        result = _run_b2(questions, facts, labels, config)
    elif condition_id == "m1":
        result = _run_m1(questions, facts, labels, config)
    else:
        raise NotImplementedError("Only the approved B0/B1/B2/M1 pilots are implemented")
    payload = {
        "status": "development_retrieval_pilot_complete",
        "condition_id": condition_id,
        "scope": pilot.get("scope"),
        "confirmatory_claims_allowed": False,
        "question_count": len(questions),
        "input_sha256": input_hashes,
        **result,
    }
    output_path = _resolve(pilot.get("output_path"), "retrieval_pilot.output_path")
    write_result_json(output_path, payload, resolved_config=config, repository=REPOSITORY)
    return output_path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config, _ = load_config(cast(Path, args.base_config), cast(Path, args.condition_config))
    output_path = run_condition(config)
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
