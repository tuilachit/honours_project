"""Deterministic answer rendering from one selected financial fact."""

from src.config import Config
from src.types import Answer, FinancialFact, Question


def render_fact_answer(
    question: Question,
    fact: FinancialFact,
    config: Config,
) -> Answer:
    """Render a fact's value, measurement context, and provenance without generation."""

    raise NotImplementedError
