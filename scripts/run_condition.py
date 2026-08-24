"""Run one retrieval condition over the approved development stress set."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import cast

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
from src.hashing import sha256_file
from src.results import write_result_json
from src.retrieval.dense import build_dense_index, retrieve_dense
from src.retrieval.fusion import fuse_route_candidates, project_fact_candidates
from src.retrieval.reranker import rerank_generic
from src.retrieval.sparse import build_sparse_index, retrieve_sparse
from src.types import (
    FinancialFact,
    GoldCellLabel,
    IndexArtifact,
    Question,
    RankedFact,
    RetrievalCandidate,
    RetrievalRoute,
    SourceTable,
)

REPOSITORY = Path(__file__).resolve().parents[1]


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
        "questions": sha256_file(question_path),
        "facts": sha256_file(fact_path),
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
    retrieval_seconds = 0.0
    reranking_seconds = 0.0
    traces: list[Mapping[str, object]] = []
    metric_rows: list[Mapping[str, float]] = []
    for question in questions:
        route_started = perf_counter()
        dense, sparse, fused = _route_rankings(question, dense_index, sparse_index, config)
        retrieval_seconds += perf_counter() - route_started
        projected = project_fact_candidates(fused, facts_by_id, config)
        rerank_started = perf_counter()
        reranked = rerank_generic(question, projected, config)
        reranking_seconds += perf_counter() - rerank_started
        metrics = dict(compute_exact_cell_metrics(reranked, labels[question.question_id], config))
        metrics.update(
            compute_stage_miss_metrics(fused, reranked, labels[question.question_id], config)
        )
        metric_rows.append(metrics)
        traces.append(
            {
                "question_id": question.question_id,
                "metrics": metrics,
                "dense": [_candidate_trace(item, rank) for rank, item in enumerate(dense, 1)],
                "sparse": [_candidate_trace(item, rank) for rank, item in enumerate(sparse, 1)],
                "fused": [_candidate_trace(item, rank) for rank, item in enumerate(fused, 1)],
                "reranked": [_ranked_trace(item) for item in reranked],
            }
        )
    return {
        "evidence_count": len(evidence),
        "aggregate_metrics": _aggregate(metric_rows),
        "question_rankings": traces,
        "index_artifacts": {"dense": dense_index, "sparse": sparse_index},
        "timings_seconds": {
            "index_build": index_seconds,
            "candidate_retrieval": retrieval_seconds,
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
    expected_routes = {"dense": True, "sparse": True, "structured": False}
    if dict(routes) != expected_routes:
        raise NotImplementedError(f"Pilot routes must equal {expected_routes}")
    dense = _mapping(retrieval.get("dense"), "retrieval.dense")
    sparse = _mapping(retrieval.get("sparse"), "retrieval.sparse")
    union = _mapping(retrieval.get("candidate_union"), "retrieval.candidate_union")
    if dense.get("similarity") != "cosine" or dense.get("normalize_embeddings") is not True:
        raise NotImplementedError("Pilot dense retrieval requires normalized cosine similarity")
    if sparse.get("algorithm") != "bm25":
        raise NotImplementedError("Pilot sparse retrieval requires BM25")
    if union.get("enabled") is not True or union.get("strategy") != "reciprocal_rank_fusion":
        raise NotImplementedError("Pilot candidate union requires reciprocal-rank fusion")
    if _config_section(config, "indexes").get("overwrite") is not True:
        raise NotImplementedError("Pilot index builders require indexes.overwrite: true")
    reranker = _config_section(config, "reranker")
    expected_reranker = (
        (False, "none") if condition_id == "b0" else (True, "generic_cross_encoder")
    )
    if (reranker.get("enabled"), reranker.get("kind")) != expected_reranker:
        raise NotImplementedError(f"{condition_id} reranker contract is {expected_reranker}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--condition-config", required=True, type=Path)
    return parser


def run_condition(config: Config) -> Path:
    """Execute B0 or B1 while preserving the development-only validity boundary."""

    if "retrieval_pilot" not in config:
        raise NotImplementedError("A configured B0/B1 retrieval pilot is required")
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
    else:
        raise NotImplementedError("Only the approved B0/B1 development pilot is implemented")
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
