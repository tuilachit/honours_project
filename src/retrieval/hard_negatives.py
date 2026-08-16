"""Natural finance-specific hard-negative mining interfaces."""

from collections.abc import Mapping, Sequence

from src.config import Config
from src.types import (
    FactCandidate,
    FinancialFact,
    GoldCellLabel,
    QueryContext,
    Question,
    TrainingPair,
)


def mine_natural_hard_negatives(
    questions_by_id: Mapping[str, Question],
    query_contexts_by_id: Mapping[str, QueryContext],
    facts_by_id: Mapping[str, FinancialFact],
    gold_labels: Sequence[GoldCellLabel],
    candidates_by_question_id: Mapping[str, Sequence[FactCandidate]],
    config: Config,
) -> list[TrainingPair]:
    """Create training pairs from genuine nearby facts after excluding valid gold cells."""

    raise NotImplementedError
