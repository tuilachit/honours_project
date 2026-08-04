"""Shared domain types used across every experiment condition."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ClaimType(StrEnum):
    """The verifier route required by an atomic claim."""

    TEXTUAL = "TEXTUAL"
    NUMERIC = "NUMERIC"


@dataclass(frozen=True, slots=True)
class Question:
    """A dataset question normalized to the common input schema."""

    question_id: str
    dataset: str
    text: str
    gold_answer: str | None = None
    relevant_document_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Chunk:
    """A retrievable passage and its source provenance."""

    chunk_id: str
    document_id: str
    text: str
    dataset: str
    page: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Answer:
    """A generated answer with the context used to produce it."""

    question_id: str
    text: str
    condition: str
    cited_chunk_ids: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class AtomicClaim:
    """A minimal verifiable proposition extracted from an answer."""

    claim_id: str
    answer_id: str
    text: str
    claim_type: ClaimType
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """The outcome and evidence emitted by one verifier."""

    claim_id: str
    verifier: str
    supported: bool | None
    support_score: float | None = None
    evidence_chunk_ids: tuple[str, ...] = ()
    rationale: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RunMetadata:
    """Immutable provenance attached to every persisted result."""

    git_commit: str
    config_hash: str
    timestamp: str

