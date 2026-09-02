"""Behavioral tests for the M1 three-route candidate union."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from threading import Barrier
from typing import cast

import numpy as np
import pytest

import scripts.run_condition as condition_runner
import src.retrieval.dense as dense_module
import src.retrieval.reranker as reranker_module
from src.config import Config
from src.facts.evidence import build_fact_evidence
from src.retrieval.fusion import fuse_fact_candidates
from src.types import (
    CellAddress,
    FactCandidate,
    FactDimension,
    FinancialFact,
    GoldCellLabel,
    MeasurementUnit,
    Question,
    RetrievalCandidate,
    RetrievalRoute,
    TimePeriod,
)


class _FakeEncoder:
    def encode(self, texts: list[str], **_: object) -> np.ndarray:
        return np.asarray(
            [
                [
                    float("revenue" in text.lower()),
                    float("2024" in text),
                    float("2023" in text),
                ]
                for text in texts
            ],
            dtype=np.float32,
        )


class _FakeReranker:
    def __init__(self, scores: list[float] | None = None) -> None:
        self.scores = scores
        self.calls = 0
        self.scored_pair_count = 0

    def predict(self, pairs: list[tuple[str, str]], **_: object) -> np.ndarray:
        self.calls += 1
        self.scored_pair_count += len(pairs)
        if self.scores is not None:
            return np.asarray(self.scores[: len(pairs)], dtype=np.float64)
        return np.asarray(
            [float("period: 2024" in evidence) for _, evidence in pairs],
            dtype=np.float64,
        )


def _fact(fact_id: str, cell_id: str, year: int) -> FinancialFact:
    return FinancialFact(
        fact_id=fact_id,
        source_address=CellAddress(
            cell_id=cell_id,
            dataset="test",
            table_id="table-1",
            row_header_path=("revenue",),
            column_header_path=(str(year),),
        ),
        raw_value="$10 million",
        entity="Acme Corporation",
        concept="revenue",
        period=TimePeriod(fiscal_year=year, raw_text=str(year)),
        measurement=MeasurementUnit(currency="$", scale=Decimal("1000000")),
        dimensions=(
            FactDimension(
                name="column_header_0",
                value=str(year),
                raw_name="column_header_0",
                raw_value=str(year),
                header_path=(str(year),),
            ),
        ),
    )


def _config(tmp_path: Path) -> Config:
    return {
        "project": {"seed": 20260815},
        "runtime": {"device": "cpu", "workers": 1, "deterministic_algorithms": True},
        "models": {
            "dense_encoder": {"name": "fake-dense", "revision": "1"},
            "generic_reranker": {"name": "fake-reranker", "revision": "1"},
        },
        "indexes": {
            "dense_path": str(tmp_path / "dense"),
            "sparse_path": str(tmp_path / "sparse"),
            "structured_path": str(tmp_path / "structured"),
            "overwrite": True,
        },
        "query_context": {
            "enabled": True,
            "mode": "automatic",
            "extractor": {
                "implementation": "deterministic_regex_v1",
                "revision": "1",
            },
        },
        "retrieval": {
            "routes": {"dense": True, "sparse": True, "structured": True},
            "dense": {
                "model_profile": "dense_encoder",
                "similarity": "cosine",
                "top_k": 2,
                "batch_size": 2,
                "normalize_embeddings": True,
            },
            "sparse": {
                "algorithm": "bm25",
                "top_k": 2,
                "k1": 1.2,
                "b": 0.75,
                "minimum_score_exclusive": 0.0,
                "token_pattern": "[A-Za-z0-9]+",
                "lowercase": True,
            },
            "structured": {
                "top_k": 2,
                "ranking_strategy": "concept_then_context_compatibility",
                "exact_match_boost": 1.0,
                "partial_match_boost": 0.5,
                "unknown_field_policy": "no_penalty",
            },
            "candidate_union": {
                "enabled": True,
                "strategy": "reciprocal_rank_fusion",
                "rrf_k": 60,
                "deduplicate_by": "stable_source_cell_id",
                "top_k": 100,
            },
            "budget_matched_sensitivity": {
                "enabled": True,
                "final_candidate_counts": [20, 50, 100],
            },
        },
        "reranker": {
            "enabled": True,
            "kind": "generic_cross_encoder",
            "model_profile": "generic_reranker",
            "input_top_k": 100,
            "output_top_k": 2,
            "batch_size": 16,
            "max_input_tokens": 512,
        },
        "evaluation": {"exact_cell_cutoffs": [1, 5, 10, 20]},
    }


def test_generic_budget_rankings_share_one_model_scoring_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    facts = [
        _fact("fact-a", "cell-a", 2022),
        _fact("fact-b", "cell-b", 2023),
        _fact("fact-c", "cell-c", 2024),
    ]
    candidates = [
        FactCandidate("q1", fact, f"evidence-{index}")
        for index, fact in enumerate(facts, start=1)
    ]
    model = _FakeReranker([0.1, 0.2, 0.9])
    monkeypatch.setattr(reranker_module, "_load_reranker", lambda *_: model)

    rankings = reranker_module.rerank_generic_budgets(
        Question("q1", "test", "What was revenue in 2024?"),
        candidates,
        (2, 3),
        _config(tmp_path),
    )

    assert [item.fact.source_address.cell_id for item in rankings[2]] == [
        "cell-b",
        "cell-a",
    ]
    assert [item.fact.source_address.cell_id for item in rankings[3]] == [
        "cell-c",
        "cell-b",
    ]
    assert model.calls == 1
    assert model.scored_pair_count == 3


def test_fact_union_preserves_route_metadata_when_deduplicating_by_cell(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    dense_fact = _fact("dense-fact", "shared-cell", 2024)
    structured_fact = _fact("structured-fact", "shared-cell", 2024)
    dense = FactCandidate(
        "q1",
        dense_fact,
        "dense-evidence",
        route_scores={RetrievalRoute.DENSE: 0.8},
        route_ranks={RetrievalRoute.DENSE: 1},
        metadata={"dense_trace": "kept"},
    )
    structured = FactCandidate(
        "q1",
        structured_fact,
        "structured-evidence",
        route_scores={RetrievalRoute.STRUCTURED: 5.0},
        route_ranks={RetrievalRoute.STRUCTURED: 1},
        metadata={"structured_components": {"period": 1.0}},
    )

    fused = fuse_fact_candidates(
        {
            RetrievalRoute.DENSE: [dense],
            RetrievalRoute.STRUCTURED: [structured],
        },
        config,
    )

    assert len(fused) == 1
    route_metadata = cast(dict[str, dict[str, object]], fused[0].metadata["route_metadata"])
    assert route_metadata["dense"]["dense_trace"] == "kept"
    assert route_metadata["structured"]["structured_components"] == {"period": 1.0}


def test_m1_retrieval_routes_overlap_in_time(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    correct = _fact("fact-correct", "cell-z", 2024)
    evidence = next(iter(build_fact_evidence((correct,), config)))
    question = Question("q1", "test", "What was Acme Corporation's revenue in 2024?")
    barrier = Barrier(3)
    encoder = _FakeEncoder()
    reranker = _FakeReranker()
    monkeypatch.setattr(dense_module, "_load_encoder", lambda *_: encoder)
    monkeypatch.setattr(reranker_module, "_load_reranker", lambda *_: reranker)

    def dense_route(*_: object) -> list[RetrievalCandidate]:
        barrier.wait(timeout=1.0)
        return [
            RetrievalCandidate(
                "q1",
                evidence,
                route_scores={RetrievalRoute.DENSE: 1.0},
                route_ranks={RetrievalRoute.DENSE: 1},
            )
        ]

    def sparse_route(*_: object) -> list[RetrievalCandidate]:
        barrier.wait(timeout=1.0)
        return [
            RetrievalCandidate(
                "q1",
                evidence,
                route_scores={RetrievalRoute.SPARSE: 1.0},
                route_ranks={RetrievalRoute.SPARSE: 1},
            )
        ]

    def structured_route(*_: object) -> list[FactCandidate]:
        barrier.wait(timeout=1.0)
        return [
            FactCandidate(
                "q1",
                correct,
                evidence.evidence_id,
                route_scores={RetrievalRoute.STRUCTURED: 1.0},
                route_ranks={RetrievalRoute.STRUCTURED: 1},
            )
        ]

    monkeypatch.setattr(condition_runner, "retrieve_dense", dense_route)
    monkeypatch.setattr(condition_runner, "retrieve_sparse", sparse_route)
    monkeypatch.setattr(condition_runner, "retrieve_structured", structured_route)

    result = condition_runner._run_m1(
        [question],
        [correct],
        {"q1": GoldCellLabel("q1", ("cell-z",))},
        config,
    )

    assert cast(dict[str, float], result["aggregate_metrics"])["hit_at_1"] == 1.0


def test_m1_runner_preserves_three_routes_and_budget_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    correct = _fact("fact-correct", "cell-z", 2024)
    distractor = _fact("fact-distractor", "cell-a", 2023)
    question = Question(
        "q1",
        "test",
        "What was Acme Corporation's revenue in 2024?",
    )
    encoder = _FakeEncoder()
    reranker = _FakeReranker()
    monkeypatch.setattr(dense_module, "_load_encoder", lambda *_: encoder)
    monkeypatch.setattr(reranker_module, "_load_reranker", lambda *_: reranker)

    result = condition_runner._run_m1(
        [question],
        [correct, distractor],
        {"q1": GoldCellLabel("q1", ("cell-z",))},
        config,
    )

    assert cast(dict[str, float], result["aggregate_metrics"])["hit_at_1"] == 1.0
    sensitivity = cast(dict[int, dict[str, float]], result["budget_metrics"])
    assert set(sensitivity) == {20, 50, 100}
    assert all(metrics["hit_at_1"] == 1.0 for metrics in sensitivity.values())
    trace = cast(list[dict[str, object]], result["question_rankings"])[0]
    assert len(cast(list[object], trace["dense"])) == 2
    assert len(cast(list[object], trace["sparse"])) == 2
    assert len(cast(list[object], trace["structured"])) == 2
    fused = cast(list[dict[str, object]], trace["fused"])
    assert set(cast(dict[str, int], fused[0]["route_ranks"])) == {
        RetrievalRoute.DENSE.value,
        RetrievalRoute.SPARSE.value,
        RetrievalRoute.STRUCTURED.value,
    }
    assert set(cast(dict[int, object], trace["budget_rankings"])) == {20, 50, 100}
    assert "route_presence_counts" in result
    assert "route_overlap_counts" in result
    assert "gold_route_coverage" in result
    assert reranker.calls == 1


def test_b1_runner_reports_the_same_budget_sensitivity_with_one_scoring_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(tmp_path)
    correct = _fact("fact-correct", "cell-z", 2024)
    distractor = _fact("fact-distractor", "cell-a", 2023)
    question = Question(
        "q1",
        "test",
        "What was Acme Corporation's revenue in 2024?",
    )
    encoder = _FakeEncoder()
    reranker = _FakeReranker()
    monkeypatch.setattr(dense_module, "_load_encoder", lambda *_: encoder)
    monkeypatch.setattr(reranker_module, "_load_reranker", lambda *_: reranker)

    result = condition_runner._run_b1(
        [question],
        [correct, distractor],
        {"q1": GoldCellLabel("q1", ("cell-z",))},
        config,
    )

    sensitivity = cast(dict[int, dict[str, float]], result["budget_metrics"])
    assert set(sensitivity) == {20, 50, 100}
    assert all(metrics["hit_at_1"] == 1.0 for metrics in sensitivity.values())
    trace = cast(list[dict[str, object]], result["question_rankings"])[0]
    assert set(cast(dict[int, object], trace["budget_rankings"])) == {20, 50, 100}
    assert reranker.calls == 1
