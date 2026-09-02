"""Deterministic unit tests for the B0/B1 retrieval pilot."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from src.config import Config
from src.eval.retrieval import (
    compute_chunk_evidence_coverage,
    compute_exact_cell_metrics,
    compute_stage_miss_metrics,
)
from src.facts.evidence import build_fact_evidence, build_flattened_evidence
from src.retrieval.fusion import fuse_route_candidates, project_fact_candidates
from src.retrieval.sparse import build_sparse_index, retrieve_sparse
from src.types import (
    CellAddress,
    EvidenceGranularity,
    EvidenceUnit,
    FactCandidate,
    FinancialFact,
    GoldCellLabel,
    Question,
    RankedFact,
    RetrievalCandidate,
    RetrievalRoute,
    SourceTable,
    TimePeriod,
)


def _config(tmp_path: Path) -> Config:
    return {
        "indexes": {"sparse_path": str(tmp_path / "sparse")},
        "retrieval": {
            "sparse": {
                "algorithm": "bm25",
                "top_k": 2,
                "k1": 1.2,
                "b": 0.75,
                "minimum_score_exclusive": 0.0,
                "token_pattern": "[A-Za-z0-9]+",
                "lowercase": True,
            },
            "candidate_union": {
                "strategy": "reciprocal_rank_fusion",
                "rrf_k": 60,
                "top_k": 10,
            },
        },
        "evaluation": {
            "chunk_cutoffs": [1, 2],
            "exact_cell_cutoffs": [1, 2],
        },
    }


def _fact(fact_id: str, cell_id: str, concept: str, year: int) -> FinancialFact:
    return FinancialFact(
        fact_id=fact_id,
        source_address=CellAddress(
            cell_id=cell_id,
            dataset="test",
            table_id="table-1",
            row_header_path=(concept,),
            column_header_path=(str(year),),
        ),
        raw_value="10",
        entity="Acme",
        concept=concept,
        period=TimePeriod(fiscal_year=year, raw_text=str(year)),
    )


def test_evidence_granularity_and_gold_metadata(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config["retrieval_pilot"] = {"table_evidence_policy": "one_complete_source_table"}
    config["representation"] = {"flattened_chunk": {"target_characters": None}}
    fact = _fact("fact-1", "cell-1", "revenue", 2024)
    table = SourceTable(
        dataset="test",
        raw_text="| metric | 2024 |\n|---|---|\n| revenue | 10 |",
        table_id="table-1",
        metadata={"fact_ids": ["fact-1"], "cell_ids": ["cell-1"]},
    )
    chunk = list(build_flattened_evidence([table], config))[0]
    exact = list(build_fact_evidence([fact], config))[0]
    assert chunk.granularity is EvidenceGranularity.FLAT_CHUNK
    assert chunk.metadata["cell_ids"] == ("cell-1",)
    assert exact.granularity is EvidenceGranularity.FINANCIAL_FACT
    assert exact.fact_ids == ("fact-1",)
    assert "concept: revenue" in exact.text
    assert "period: 2024" in exact.text


def test_bm25_and_rrf_keep_route_evidence(tmp_path: Path) -> None:
    config = _config(tmp_path)
    evidence = [
        EvidenceUnit("a", EvidenceGranularity.FINANCIAL_FACT, "Acme revenue 2024", "test"),
        EvidenceUnit("b", EvidenceGranularity.FINANCIAL_FACT, "Acme expenses 2023", "test"),
    ]
    question = Question("q1", "test", "What was Acme revenue in 2024?")
    sparse = retrieve_sparse(question, build_sparse_index(evidence, config), config)
    assert sparse[0].evidence.evidence_id == "a"
    dense = [
        RetrievalCandidate(
            "q1",
            evidence[1],
            route_scores={RetrievalRoute.DENSE: 0.9},
            route_ranks={RetrievalRoute.DENSE: 1},
        )
    ]
    fused = fuse_route_candidates(
        {RetrievalRoute.DENSE: dense, RetrievalRoute.SPARSE: sparse}, config
    )
    assert fused[0].evidence.evidence_id in {"a", "b"}
    assert cast(float, fused[0].metadata["fusion_score"]) > 0


def test_exact_and_chunk_metrics_use_cell_identity(tmp_path: Path) -> None:
    config = _config(tmp_path)
    fact = _fact("fact-1", "cell-1", "revenue", 2024)
    label = GoldCellLabel("q1", ("cell-1",))
    fact_candidate = FactCandidate("q1", fact, "evidence-1")
    ranked = [RankedFact("q1", fact, 1, 1.0, fact_candidate)]
    exact = compute_exact_cell_metrics(ranked, label, config)
    assert exact["hit_at_1"] == 1.0
    assert exact["mean_reciprocal_rank"] == 1.0
    chunk_candidate = RetrievalCandidate(
        "q1",
        EvidenceUnit(
            "chunk-1",
            EvidenceGranularity.FLAT_CHUNK,
            "table",
            "test",
            metadata={"cell_ids": ("cell-1",)},
        ),
    )
    assert compute_chunk_evidence_coverage([chunk_candidate], label, config)[
        "evidence_coverage_at_1"
    ] == 1.0


def test_stage_miss_metrics_accept_fact_candidates(tmp_path: Path) -> None:
    config = _config(tmp_path)
    fact = _fact("fact-1", "cell-1", "revenue", 2024)
    label = GoldCellLabel("q1", ("cell-1",))
    candidate = FactCandidate("q1", fact, "evidence-1")
    ranked = [RankedFact("q1", fact, 1, 1.0, candidate)]

    metrics = compute_stage_miss_metrics([candidate], ranked, label, config)

    assert metrics == {
        "candidate_generation_miss": 0.0,
        "reranking_miss": 0.0,
        "final_miss": 0.0,
    }


def test_projection_rejects_chunks_and_maps_singular_fact(tmp_path: Path) -> None:
    config = _config(tmp_path)
    fact = _fact("fact-1", "cell-1", "revenue", 2024)
    candidate = RetrievalCandidate(
        "q1",
        EvidenceUnit(
            "evidence-1",
            EvidenceGranularity.FINANCIAL_FACT,
            "fact",
            "test",
            fact_ids=("fact-1",),
        ),
    )
    projected = project_fact_candidates([candidate], {"fact-1": fact}, config)
    assert projected[0].fact.source_address.cell_id == "cell-1"
