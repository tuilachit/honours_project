"""Deterministic tests for the B2 structured lookup baseline."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import cast

import pytest

import scripts.run_condition as condition_runner
from src.config import Config
from src.query.extract import extract_query_context
from src.retrieval.structured import (
    build_structured_index,
    rank_structured_lookup,
    retrieve_structured,
)
from src.types import (
    CellAddress,
    ConstraintState,
    FactDimension,
    FinancialFact,
    GoldCellLabel,
    IndexArtifact,
    MeasurementUnit,
    Question,
    RetrievalRoute,
    TimePeriod,
)


def _config(tmp_path: Path) -> Config:
    return {
        "indexes": {"structured_path": str(tmp_path / "structured")},
        "query_context": {
            "enabled": True,
            "mode": "automatic",
            "extractor": {
                "implementation": "deterministic_regex_v1",
                "revision": "1",
            },
            "arbitrary_dimensions_enabled": True,
            "missing_dimension_policy": "mask",
            "confidence_threshold": 0.0,
        },
        "retrieval": {
            "structured": {
                "top_k": 10,
                "ranking_strategy": "concept_then_context_compatibility",
                "exact_match_boost": 1.0,
                "partial_match_boost": 0.5,
                "unknown_field_policy": "no_penalty",
            },
            "final_top_k": 5,
        },
        "reranker": {
            "input_top_k": 10,
            "output_top_k": 5,
            "field_features": {
                "use_entity": True,
                "use_concept": True,
                "use_period": True,
                "use_unit": True,
                "use_arbitrary_dimensions": True,
                "use_raw_header_paths": True,
                "missing_value_indicator": True,
            },
        },
        "evaluation": {"exact_cell_cutoffs": [1, 5]},
    }


def _fact(
    fact_id: str,
    cell_id: str,
    *,
    entity: str | None,
    concept: str,
    year: int,
    segment: str,
    dimension_name: str = "segment",
) -> FinancialFact:
    return FinancialFact(
        fact_id=fact_id,
        source_address=CellAddress(
            cell_id=cell_id,
            dataset="test",
            table_id="table-1",
            row_header_path=(concept,),
            column_header_path=(str(year), segment),
        ),
        raw_value="$10 million",
        entity=entity,
        concept=concept,
        period=TimePeriod(fiscal_year=year, raw_text=str(year)),
        measurement=MeasurementUnit(
            currency="$",
            scale=Decimal("1000000"),
            raw_text="$ million",
        ),
        dimensions=(
            FactDimension(
                name=dimension_name,
                value=segment,
                raw_name=dimension_name,
                raw_value=segment,
                header_path=(segment,),
            ),
        ),
    )


def test_automatic_query_context_preserves_explicit_and_unknown_fields(
    tmp_path: Path,
) -> None:
    question = Question(
        "q1",
        "test",
        "What were Acme Corporation's total revenue for the North America segment in 2024?",
    )

    context = extract_query_context(question, _config(tmp_path))
    constraints = {constraint.name: constraint for constraint in context.constraints}

    assert constraints["entity"].state is ConstraintState.EXPLICIT
    assert constraints["entity"].value == "Acme Corporation"
    assert constraints["period"].state is ConstraintState.EXPLICIT
    assert constraints["period"].value == "2024"
    assert constraints["concept"].state is ConstraintState.EXPLICIT
    assert "revenue" in str(constraints["concept"].value).lower()
    assert constraints["segment"].state is ConstraintState.EXPLICIT
    assert constraints["segment"].value == "North America"
    assert constraints["currency"].state is ConstraintState.UNKNOWN
    assert "north" not in str(constraints["concept"].value).split()
    assert "america" not in str(constraints["concept"].value).split()
    assert context.metadata["implementation"] == "deterministic_regex_v1"


@pytest.mark.parametrize(
    ("text", "expected_entity", "expected_period", "required_concept", "excluded_concept"),
    (
        (
            "What was the total value of raw materials and supplies in Lilly's (Eli) "
            "inventories at December 31, 2018?",
            "Lilly",
            "2018",
            {"raw", "materials", "supplies"},
            set(),
        ),
        (
            "Using the financial table, what was the cost of sales for Analog Devices in "
            "2019, as reflected in the total stock-based compensation expense?",
            "Analog Devices",
            "2019",
            {"cost", "sales"},
            {"stock", "compensation"},
        ),
        (
            "According to the reported table, what was Lockheed Martin's backlog at year-end "
            "in 2014, according to their 2015 report?",
            "Lockheed Martin",
            "2014",
            {"backlog"},
            {"report"},
        ),
        (
            "Using the financial table, what was the net income of United Parcel Service, "
            "Inc. and subsidiaries in 2012, as reported in Management's Discussion and Analysis?",
            "United Parcel Service, Inc",
            "2012",
            {"net", "income"},
            {"management", "discussion"},
        ),
    ),
)
def test_query_extraction_isolates_the_requested_fact_from_source_context(
    tmp_path: Path,
    text: str,
    expected_entity: str,
    expected_period: str,
    required_concept: set[str],
    excluded_concept: set[str],
) -> None:
    context = extract_query_context(Question("q", "test", text), _config(tmp_path))
    constraints = {constraint.name: constraint for constraint in context.constraints}
    concept_tokens = set(str(constraints["concept"].value).split())

    assert constraints["entity"].value == expected_entity
    assert constraints["period"].value == expected_period
    assert required_concept.issubset(concept_tokens)
    assert concept_tokens.isdisjoint(excluded_concept)


def test_scale_requires_an_answer_unit_phrase_not_an_unrelated_quantity(tmp_path: Path) -> None:
    unrelated = extract_query_context(
        Question(
            "q1",
            "test",
            "What was the average price per share in 2013 as part of a 100 million share program?",
        ),
        _config(tmp_path),
    )
    answer_unit = extract_query_context(
        Question(
            "q2",
            "test",
            "What was the operating profit, in millions, for Acme in 2013?",
        ),
        _config(tmp_path),
    )
    unrelated_by_name = {constraint.name: constraint for constraint in unrelated.constraints}
    answer_unit_by_name = {constraint.name: constraint for constraint in answer_unit.constraints}

    assert unrelated_by_name["scale"].state is ConstraintState.UNKNOWN
    assert answer_unit_by_name["scale"].value == "1000000"


def test_dimension_extraction_uses_the_nearest_preposition_and_avoids_product_nouns(
    tmp_path: Path,
) -> None:
    segment = extract_query_context(
        Question(
            "q1",
            "test",
            "What was the natural gas marketed volume in billion cubic feet for ONEOK's "
            "energy services segment in 2012?",
        ),
        _config(tmp_path),
    )
    product_noun = extract_query_context(
        Question(
            "q2",
            "test",
            "What percentage of Cisco Systems Inc.'s total revenue in fiscal 2019 was "
            "generated from product sales?",
        ),
        _config(tmp_path),
    )
    segment_by_name = {constraint.name: constraint for constraint in segment.constraints}
    product_by_name = {constraint.name: constraint for constraint in product_noun.constraints}

    assert segment_by_name["segment"].value == "energy services"
    assert "product" not in product_by_name


def test_structured_lookup_prefers_the_complete_context_match(tmp_path: Path) -> None:
    config = _config(tmp_path)
    facts = [
        _fact(
            "fact-correct",
            "cell-z",
            entity="Acme Corporation",
            concept="total revenue",
            year=2024,
            segment="North America",
        ),
        _fact(
            "fact-period",
            "cell-a",
            entity="Acme Corporation",
            concept="total revenue",
            year=2023,
            segment="North America",
        ),
        _fact(
            "fact-entity",
            "cell-b",
            entity="Other Corporation",
            concept="total revenue",
            year=2024,
            segment="North America",
        ),
        _fact(
            "fact-segment",
            "cell-c",
            entity="Acme Corporation",
            concept="total revenue",
            year=2024,
            segment="Europe",
        ),
        _fact(
            "fact-concept",
            "cell-d",
            entity="Acme Corporation",
            concept="operating expenses",
            year=2024,
            segment="North America",
        ),
    ]
    question = Question(
        "q1",
        "test",
        "What were Acme Corporation's total revenue for the North America segment in 2024?",
    )
    context = extract_query_context(question, config)

    index = build_structured_index(facts, config)
    candidates = retrieve_structured(
        question,
        context,
        index,
        {fact.fact_id: fact for fact in facts},
        config,
    )
    ranked = rank_structured_lookup(question, context, candidates, config)

    assert index.route is RetrievalRoute.STRUCTURED
    assert ranked[0].fact.source_address.cell_id == "cell-z"
    assert ranked[0].component_scores["concept"] > 0.5
    assert ranked[0].component_scores["period"] == 1.0
    assert ranked[0].component_scores["entity"] == 1.0
    assert ranked[0].component_scores["dimensions"] == 1.0


def test_structured_lookup_matches_semantic_query_to_generic_column_dimension(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    facts = [
        _fact(
            "fact-wrong-segment",
            "cell-a",
            entity="Acme Corporation",
            concept="total revenue",
            year=2024,
            segment="Europe",
            dimension_name="column_header_1",
        ),
        _fact(
            "fact-correct",
            "cell-z",
            entity="Acme Corporation",
            concept="total revenue",
            year=2024,
            segment="North America",
            dimension_name="column_header_1",
        ),
    ]
    question = Question(
        "q1",
        "test",
        "What were Acme Corporation's total revenue for the North America segment in 2024?",
    )
    context = extract_query_context(question, config)
    index = build_structured_index(facts, config)

    candidates = retrieve_structured(
        question,
        context,
        index,
        {fact.fact_id: fact for fact in facts},
        config,
    )
    ranked = rank_structured_lookup(question, context, candidates, config)

    assert ranked[0].fact.source_address.cell_id == "cell-z"
    assert ranked[0].component_scores["dimensions"] == 1.0
    assert ranked[1].component_scores["dimensions"] == 0.0


def test_unknown_fact_field_is_neutral_but_a_contradiction_is_not(tmp_path: Path) -> None:
    config = _config(tmp_path)
    facts = [
        _fact(
            "fact-unknown",
            "cell-z",
            entity=None,
            concept="revenue",
            year=2024,
            segment="North America",
        ),
        _fact(
            "fact-contradiction",
            "cell-a",
            entity="Other Corporation",
            concept="revenue",
            year=2024,
            segment="North America",
        ),
    ]
    question = Question("q2", "test", "What was Acme's revenue in 2024?")
    context = extract_query_context(question, config)
    index = build_structured_index(facts, config)

    candidates = retrieve_structured(
        question,
        context,
        index,
        {fact.fact_id: fact for fact in facts},
        config,
    )
    ranked = rank_structured_lookup(question, context, candidates, config)

    assert ranked[0].fact.source_address.cell_id == "cell-z"
    assert ranked[0].component_scores["entity"] == 0.0
    assert ranked[1].component_scores["entity"] < 0.0


def test_b2_runner_records_context_rankings_metrics_and_timing(tmp_path: Path) -> None:
    config = _config(tmp_path)
    correct = _fact(
        "fact-correct",
        "cell-z",
        entity="Acme Corporation",
        concept="total revenue",
        year=2024,
        segment="North America",
    )
    distractor = _fact(
        "fact-distractor",
        "cell-a",
        entity="Acme Corporation",
        concept="total revenue",
        year=2023,
        segment="North America",
    )
    question = Question(
        "q1",
        "test",
        "What were Acme Corporation's total revenue for the North America segment in 2024?",
    )

    result = condition_runner._run_b2(
        [question],
        [correct, distractor],
        {"q1": GoldCellLabel("q1", ("cell-z",))},
        config,
    )

    assert cast(dict[str, float], result["aggregate_metrics"])["hit_at_1"] == 1.0
    trace = cast(list[dict[str, object]], result["question_rankings"])[0]
    assert trace["question_id"] == "q1"
    assert cast(dict[str, object], trace["query_context"])["question_id"] == "q1"
    assert len(cast(list[object], trace["structured"])) == 2
    assert cast(list[dict[str, object]], trace["reranked"])[0]["cell_id"] == "cell-z"
    artifact = cast(dict[str, IndexArtifact], result["index_artifacts"])["structured"]
    assert artifact.route is RetrievalRoute.STRUCTURED
    assert cast(dict[str, float], result["timings_seconds"])["total"] >= 0.0
