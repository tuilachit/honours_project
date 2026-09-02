"""Fixed-prompt answer-generation interface for secondary evaluation."""

from collections.abc import Sequence

from src.config import Config
from src.types import Answer, Question, RankedFact, RetrievalCandidate


def generate_answer(
    question: Question,
    ranked_facts: Sequence[RankedFact],
    config: Config,
) -> Answer:
    """Generate an answer from ranked facts using only the configured fixed prompt."""

    raise NotImplementedError


def generate_answer_from_evidence(
    question: Question,
    candidates: Sequence[RetrievalCandidate],
    config: Config,
) -> Answer:
    """Generate B0's secondary answer directly from ranked flattened evidence."""

    raise NotImplementedError
