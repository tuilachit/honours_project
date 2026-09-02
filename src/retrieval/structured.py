"""Deterministic schema-flexible structured indexing and retrieval."""

from __future__ import annotations

import hashlib
import heapq
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import cast

from src.config import Config, hash_config
from src.facts.evidence import build_fact_evidence
from src.types import (
    ConstraintState,
    EvidenceGranularity,
    FactCandidate,
    FinancialFact,
    IndexArtifact,
    QueryContext,
    Question,
    RankedFact,
    RetrievalRoute,
)

_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
_GENERIC_DIMENSION_PATTERN = re.compile(r"^column(?: header)? \d+$")
_CORE_CONSTRAINTS = {"entity", "concept", "period", "currency", "scale", "percentage"}
_MATCH_STOPWORDS = {
    "a",
    "an",
    "and",
    "at",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "the",
    "to",
    "year",
}
_ENTITY_STOPWORDS = {"co", "company", "corp", "corporation", "inc", "ltd", "limited"}


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _structured_config(config: Config) -> Mapping[str, object]:
    retrieval = _mapping(config.get("retrieval"), "retrieval")
    return _mapping(retrieval.get("structured"), "retrieval.structured")


def _normalise(text: str) -> str:
    return " ".join(_tokens(text))


def _tokens(text: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in (item.lower() for item in _TOKEN_PATTERN.findall(text))
        if token not in _MATCH_STOPWORDS
    )


def _index_dir(config: Config) -> Path:
    root = _mapping(config.get("indexes"), "indexes").get("structured_path")
    if not isinstance(root, str):
        raise ValueError("indexes.structured_path must be a path string")
    path = Path(root)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    return path / EvidenceGranularity.FINANCIAL_FACT.value / hash_config(config)


def _fact_entry(fact: FinancialFact, config: Config) -> Mapping[str, object]:
    evidence = next(iter(build_fact_evidence((fact,), config)))
    period_values: list[str] = []
    if fact.period:
        if fact.period.fiscal_year is not None:
            period_values.append(str(fact.period.fiscal_year))
        if fact.period.raw_text:
            period_values.append(fact.period.raw_text)
    measurement = fact.measurement
    dimensions = [
        {
            "name": dimension.name,
            "raw_name": dimension.raw_name,
            "value": dimension.value,
            "raw_value": dimension.raw_value,
            "header_path": list(dimension.header_path),
        }
        for dimension in fact.dimensions
    ]
    return {
        "fact_id": fact.fact_id,
        "evidence_id": evidence.evidence_id,
        "cell_id": fact.source_address.cell_id,
        "entity": fact.entity,
        "concept": fact.concept,
        "period": period_values,
        "currency": measurement.currency if measurement else None,
        "scale": str(measurement.scale) if measurement and measurement.scale is not None else None,
        "percentage": (
            str(measurement.is_percentage).lower()
            if measurement and measurement.is_percentage is not None
            else None
        ),
        "dimensions": dimensions,
        "row_header_path": list(fact.source_address.row_header_path),
        "column_header_path": list(fact.source_address.column_header_path),
    }


@lru_cache(maxsize=4)
def _load_index(location: str) -> tuple[Mapping[str, object], ...]:
    payload = json.loads(Path(location).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Structured index payload must be a mapping")
    raw_entries = payload.get("entries")
    if not isinstance(raw_entries, list) or not all(
        isinstance(entry, Mapping) for entry in raw_entries
    ):
        raise ValueError("Structured index entries must be mappings")
    return tuple(cast(list[Mapping[str, object]], raw_entries))


def _explicit_constraints(query_context: QueryContext) -> Mapping[str, str]:
    values: dict[str, str] = {}
    for constraint in query_context.constraints:
        if constraint.state not in {ConstraintState.EXPLICIT, ConstraintState.IMPLIED}:
            continue
        if constraint.value:
            values[constraint.name] = constraint.value
    return values


def _similarity(query_value: str, candidate_value: str | None) -> float:
    if not candidate_value:
        return 0.0
    query_normalised = _normalise(query_value)
    candidate_normalised = _normalise(candidate_value)
    if not query_normalised or not candidate_normalised:
        return 0.0
    if query_normalised == candidate_normalised:
        return 1.0
    query_tokens = set(query_normalised.split())
    candidate_tokens = set(candidate_normalised.split())
    overlap = query_tokens.intersection(candidate_tokens)
    if not overlap:
        return -1.0
    candidate_coverage = len(overlap) / len(candidate_tokens)
    query_coverage = len(overlap) / len(query_tokens)
    return 0.75 * candidate_coverage + 0.25 * query_coverage


def _entity_similarity(query_value: str, candidate_value: str | None) -> float:
    if not candidate_value:
        return 0.0
    query_tokens = set(_tokens(query_value)).difference(_ENTITY_STOPWORDS)
    candidate_tokens = set(_tokens(candidate_value)).difference(_ENTITY_STOPWORDS)
    if not query_tokens or not candidate_tokens:
        return 0.0
    if query_tokens == candidate_tokens:
        return 1.0
    overlap = query_tokens.intersection(candidate_tokens)
    if not overlap:
        return -1.0
    return 0.75 * len(overlap) / len(candidate_tokens) + 0.25 * len(overlap) / len(
        query_tokens
    )


def _period_similarity(query_value: str, values: object) -> float:
    if not isinstance(values, list) or not values:
        return 0.0
    normalised = {_normalise(str(value)) for value in values}
    return 1.0 if _normalise(query_value) in normalised else -1.0


def _dimension_similarity(
    constraints: Mapping[str, str],
    dimensions: object,
) -> float:
    requested = {
        name: value for name, value in constraints.items() if name not in _CORE_CONSTRAINTS
    }
    if not requested:
        return 0.0
    if not isinstance(dimensions, list):
        return 0.0
    scores: list[float] = []
    for name, value in requested.items():
        named = [
            dimension
            for dimension in dimensions
            if isinstance(dimension, Mapping)
            and any(
                _normalise(str(dimension.get(key, ""))) == _normalise(name)
                for key in ("name", "raw_name")
            )
        ]
        matching = named or [
            dimension
            for dimension in dimensions
            if isinstance(dimension, Mapping)
            and any(
                _GENERIC_DIMENSION_PATTERN.fullmatch(_normalise(str(dimension.get(key, ""))))
                for key in ("name", "raw_name")
            )
        ]
        if not matching:
            scores.append(0.0)
            continue
        best = max(
            _similarity(value, cast(str | None, dimension.get("value")))
            for dimension in matching
        )
        scores.append(max(best, 0.0) if not named else best)
    return sum(scores) / len(scores)


def _entry_components(
    entry: Mapping[str, object],
    query_context: QueryContext,
    constraints: Mapping[str, str] | None = None,
) -> Mapping[str, float]:
    constraints = constraints or _explicit_constraints(query_context)
    concept_text = " ".join(
        [
            str(entry.get("concept") or ""),
            *[str(value) for value in cast(list[object], entry.get("row_header_path", []))],
        ]
    )
    concept = _similarity(constraints.get("concept", ""), concept_text)
    entity = (
        _entity_similarity(constraints["entity"], cast(str | None, entry.get("entity")))
        if "entity" in constraints
        else 0.0
    )
    period = (
        _period_similarity(constraints["period"], entry.get("period"))
        if "period" in constraints
        else 0.0
    )
    unit_scores = [
        _similarity(constraints[name], cast(str | None, entry.get(name)))
        for name in ("currency", "scale", "percentage")
        if name in constraints
    ]
    return {
        "concept": concept,
        "entity": entity,
        "period": period,
        "unit": sum(unit_scores) / len(unit_scores) if unit_scores else 0.0,
        "dimensions": _dimension_similarity(constraints, entry.get("dimensions")),
    }


def _score(components: Mapping[str, float], config: Config) -> float:
    structured = _structured_config(config)
    exact = float(cast(str | int | float, structured["exact_match_boost"]))
    partial = float(cast(str | int | float, structured["partial_match_boost"]))

    def weighted(value: float) -> float:
        if value == 1.0:
            return exact
        if value > 0.0:
            return partial * value
        if value < 0.0:
            return -exact
        return 0.0

    return 4.0 * weighted(components["concept"]) + sum(
        weighted(components[name]) for name in ("entity", "period", "unit", "dimensions")
    )


def build_structured_index(
    facts: Iterable[FinancialFact],
    config: Config,
) -> IndexArtifact:
    """Build an index over core context and observed dynamic dimensions."""

    corpus = tuple(facts)
    if not corpus:
        raise ValueError("Cannot build a structured index over an empty corpus")
    entries = sorted(
        (_fact_entry(fact, config) for fact in corpus),
        key=lambda entry: str(entry["fact_id"]),
    )
    directory = _index_dir(config)
    directory.mkdir(parents=True, exist_ok=True)
    location = directory / "index.json"
    location.write_text(
        json.dumps({"entries": entries}, sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    _load_index.cache_clear()
    return IndexArtifact(
        artifact_id=hashlib.sha256(location.read_bytes()).hexdigest(),
        route=RetrievalRoute.STRUCTURED,
        granularity=EvidenceGranularity.FINANCIAL_FACT,
        location=str(location),
        corpus_size=len(entries),
        config_hash=hash_config(config),
        metadata={"implementation": "schema_flexible_structured_v1"},
    )


def retrieve_structured(
    question: Question,
    query_context: QueryContext,
    index: IndexArtifact,
    facts_by_id: Mapping[str, FinancialFact],
    config: Config,
) -> list[FactCandidate]:
    """Return singular fact candidates without assuming a fixed scope schema."""

    del question
    constraints = _explicit_constraints(query_context)
    scored: list[tuple[float, str, Mapping[str, object], Mapping[str, float]]] = []
    for entry in _load_index(index.location):
        fact_id = entry.get("fact_id")
        cell_id = entry.get("cell_id")
        if not isinstance(fact_id, str) or not isinstance(cell_id, str):
            raise ValueError("Structured index identity is invalid")
        components = _entry_components(entry, query_context, constraints)
        scored.append((_score(components, config), cell_id, entry, components))
    top_k = int(cast(str | int, _structured_config(config)["top_k"]))
    ranked = heapq.nsmallest(top_k, scored, key=lambda item: (-item[0], item[1]))
    candidates: list[FactCandidate] = []
    for rank, (score, _, entry, components) in enumerate(ranked, start=1):
        fact_id = cast(str, entry["fact_id"])
        fact = facts_by_id.get(fact_id)
        if fact is None:
            raise ValueError(f"Structured index references unknown fact {fact_id}")
        candidates.append(
            FactCandidate(
                question_id=query_context.question_id,
                fact=fact,
                evidence_id=cast(str, entry["evidence_id"]),
                route_scores={RetrievalRoute.STRUCTURED: score},
                route_ranks={RetrievalRoute.STRUCTURED: rank},
                metadata={
                    "structured_components": dict(components),
                    "structured_score": score,
                },
            )
        )
    return candidates


def rank_structured_lookup(
    question: Question,
    query_context: QueryContext,
    candidates: Sequence[FactCandidate],
    config: Config,
) -> list[RankedFact]:
    """Rank B2 facts by configured structured and metric compatibility rules."""

    del question
    input_top_k = int(
        cast(str | int, _mapping(config.get("reranker"), "reranker")["input_top_k"])
    )
    output_top_k = int(
        cast(str | int, _mapping(config.get("reranker"), "reranker")["output_top_k"])
    )
    rescored: list[tuple[float, FactCandidate, Mapping[str, float]]] = []
    for candidate in candidates[:input_top_k]:
        raw_components = candidate.metadata.get("structured_components")
        if not isinstance(raw_components, Mapping):
            raise ValueError("Structured candidate is missing indexed component scores")
        components = {
            name: float(cast(str | int | float, raw_components[name]))
            for name in ("concept", "entity", "period", "unit", "dimensions")
        }
        rescored.append((_score(components, config), candidate, components))
    ranked = sorted(
        rescored,
        key=lambda item: (-item[0], item[1].fact.source_address.cell_id),
    )[:output_top_k]
    return [
        RankedFact(
            question_id=query_context.question_id,
            fact=candidate.fact,
            rank=rank,
            score=score,
            candidate=candidate,
            component_scores={
                **components,
                "structured_route": float(
                    candidate.route_scores.get(RetrievalRoute.STRUCTURED, 0.0)
                ),
            },
            metadata={"implementation": "structured_metric_matcher_v1"},
        )
        for rank, (score, candidate, components) in enumerate(ranked, start=1)
    ]
