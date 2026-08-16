"""Ordinary negative-pair construction interfaces for M2 training."""

from collections.abc import Mapping, Sequence

from src.config import Config
from src.types import FinancialFact, GoldCellLabel, Question, TrainingPair


def build_ordinary_training_pairs(
    questions_by_id: Mapping[str, Question],
    facts_by_id: Mapping[str, FinancialFact],
    gold_labels: Sequence[GoldCellLabel],
    config: Config,
) -> list[TrainingPair]:
    """Build configured non-hard negative pairs without using held-out examples."""

    raise NotImplementedError
