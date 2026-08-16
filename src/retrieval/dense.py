"""Dense indexing and retrieval interfaces."""

from collections.abc import Iterable

from src.config import Config
from src.types import EvidenceUnit, IndexArtifact, Question, RetrievalCandidate


def build_dense_index(
    evidence_units: Iterable[EvidenceUnit],
    config: Config,
) -> IndexArtifact:
    """Build the configured dense index over versioned evidence units."""

    raise NotImplementedError


def retrieve_dense(
    question: Question,
    index: IndexArtifact,
    config: Config,
) -> list[RetrievalCandidate]:
    """Retrieve from the raw question without structured query-context rewriting."""

    raise NotImplementedError
