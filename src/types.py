"""Shared domain types for exact-cell financial retrieval experiments."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from enum import StrEnum
from typing import TypeAlias

Metadata: TypeAlias = Mapping[str, object]


class ConstraintState(StrEnum):
    """How a query constraint was expressed or inferred."""

    EXPLICIT = "explicit"
    IMPLIED = "implied"
    ABSENT = "absent"
    UNKNOWN = "unknown"


class EvidenceGranularity(StrEnum):
    """The retrieval granularity represented by an evidence unit."""

    FLAT_CHUNK = "flattened_chunk"
    FINANCIAL_FACT = "financial_fact"


class RetrievalRoute(StrEnum):
    """A first-stage route that contributed a retrieval candidate."""

    DENSE = "dense"
    SPARSE = "sparse"
    STRUCTURED = "structured"


@dataclass(frozen=True, slots=True)
class Question:
    """A dataset question normalised to the common experiment schema."""

    question_id: str
    dataset: str
    text: str
    gold_answer: str | None = None
    relevant_document_ids: tuple[str, ...] = ()
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class SourceTable:
    """A raw table-bearing source record before structural parsing."""

    dataset: str
    raw_text: str
    table_id: str
    document_id: str | None = None
    context_id: str | None = None
    page_number: int | None = None
    source_uri: str | None = None
    source_format: str | None = None
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DatasetExample:
    """One question and its source material in a dataset-neutral representation."""

    example_id: str
    question: Question
    source_table_ids: tuple[str, ...]
    context_id: str | None = None
    gold_labels: tuple[GoldCellLabel, ...] = ()
    split: str | None = None
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DatasetCorpus:
    """Deduplicated questions and source tables for one dataset snapshot."""

    dataset: str
    examples: tuple[DatasetExample, ...]
    source_tables: tuple[SourceTable, ...]
    revision: str
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class CellAddress:
    """Stable provenance and structural coordinates for one source table cell."""

    cell_id: str
    dataset: str
    document_id: str | None = None
    context_id: str | None = None
    page_number: int | None = None
    table_id: str | None = None
    row_index: int | None = None
    column_index: int | None = None
    row_header_path: tuple[str, ...] = ()
    column_header_path: tuple[str, ...] = ()
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ParsedCell:
    """A parsed cell that retains its raw value, headers, and source qualifiers."""

    address: CellAddress
    raw_value: str
    normalized_value: Decimal | None = None
    qualifiers: tuple[str, ...] = ()
    footnotes: tuple[str, ...] = ()
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ParsedTable:
    """A structurally parsed source table with addressable cells."""

    source: SourceTable
    cells: tuple[ParsedCell, ...]
    table_id: str | None = None
    title: str | None = None
    caption: str | None = None
    row_count: int | None = None
    column_count: int | None = None
    qualifiers: tuple[str, ...] = ()
    footnotes: tuple[str, ...] = ()
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TimePeriod:
    """Normalised and raw temporal context for a financial fact."""

    fiscal_year: int | None = None
    fiscal_period: str | None = None
    start_date: str | None = None
    end_date: str | None = None
    instant_date: str | None = None
    period_kind: str | None = None
    raw_text: str | None = None
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MeasurementUnit:
    """Currency, unit, and scale attached to a reported value."""

    currency: str | None = None
    unit: str | None = None
    scale: Decimal | None = None
    is_percentage: bool | None = None
    raw_text: str | None = None
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FactDimension:
    """A canonical dimension that also preserves its source-table wording."""

    name: str
    value: str | None = None
    raw_name: str | None = None
    raw_value: str | None = None
    normalization_version: str | None = None
    header_path: tuple[str, ...] = ()
    qualifiers: tuple[str, ...] = ()
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FinancialFact:
    """An addressable financial value with explicit identity and provenance."""

    fact_id: str
    source_address: CellAddress
    raw_value: str
    entity: str | None = None
    concept: str | None = None
    period: TimePeriod | None = None
    measurement: MeasurementUnit | None = None
    normalized_value: Decimal | None = None
    dimensions: tuple[FactDimension, ...] = ()
    qualifiers: tuple[str, ...] = ()
    footnotes: tuple[str, ...] = ()
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class QueryConstraint:
    """One canonical query constraint with its original wording preserved."""

    name: str
    state: ConstraintState
    value: str | None = None
    raw_name: str | None = None
    raw_value: str | None = None
    normalization_version: str | None = None
    source_header_path: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class QueryContext:
    """The complete set of structured constraints for one question."""

    question_id: str
    constraints: tuple[QueryConstraint, ...] = ()
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvidenceUnit:
    """A retrievable chunk or fact record, optionally covering multiple facts."""

    evidence_id: str
    granularity: EvidenceGranularity
    text: str
    dataset: str
    document_id: str | None = None
    page_number: int | None = None
    fact_ids: tuple[str, ...] = ()
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class IndexArtifact:
    """A persisted retrieval index and the corpus contract used to build it."""

    artifact_id: str
    route: RetrievalRoute
    granularity: EvidenceGranularity
    location: str
    corpus_size: int
    config_hash: str
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ModelArtifact:
    """A persisted trained model with an immutable identity and configuration."""

    artifact_id: str
    location: str
    model_name: str
    model_revision: str
    seed: int
    config_hash: str
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    """A candidate with independent scores and ranks from all contributing routes."""

    question_id: str
    evidence: EvidenceUnit
    route_scores: Mapping[RetrievalRoute, float] = field(default_factory=dict)
    route_ranks: Mapping[RetrievalRoute, int] = field(default_factory=dict)
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FactCandidate:
    """A retrieval candidate proven to represent exactly one addressable fact."""

    question_id: str
    fact: FinancialFact
    evidence_id: str
    route_scores: Mapping[RetrievalRoute, float] = field(default_factory=dict)
    route_ranks: Mapping[RetrievalRoute, int] = field(default_factory=dict)
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RankedFact:
    """A financial fact after candidate fusion or reranking."""

    question_id: str
    fact: FinancialFact
    rank: int
    score: float
    candidate: FactCandidate | None = None
    component_scores: Mapping[str, float] = field(default_factory=dict)
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GoldCellLabel:
    """The complete set of source cells accepted as correct for one question."""

    question_id: str
    valid_cell_ids: tuple[str, ...]
    annotator_id: str | None = None
    adjudicated: bool = False
    notes: str | None = None
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class FieldComparison:
    """One prediction compared with a deterministically selected valid gold cell."""

    predicted_cell_id: str
    reference_cell_id: str
    mismatches: Mapping[str, bool]
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class TrainingPair:
    """A self-contained, leakage-auditable pairwise ranking example."""

    pair_id: str
    question: Question
    query_context: QueryContext
    positive_fact: FinancialFact
    negative_fact: FinancialFact
    negative_type: str
    split: str
    group_id: str
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Answer:
    """A rendered or generated answer with exact evidence provenance."""

    question_id: str
    text: str
    condition: str
    cited_fact_ids: tuple[str, ...] = ()
    cited_evidence_ids: tuple[str, ...] = ()
    metadata: Metadata = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunMetadata:
    """Immutable provenance attached to every persisted result."""

    git_commit: str
    git_dirty: bool
    git_worktree_sha256: str | None
    config_hash: str
    timestamp: str
