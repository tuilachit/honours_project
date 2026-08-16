"""Schema-flexible query-constraint extraction interfaces."""

from src.config import Config
from src.types import QueryConstraint, QueryContext, Question


def extract_query_constraints(
    question: Question,
    config: Config,
) -> tuple[QueryConstraint, ...]:
    """Extract configured entity, concept, time, unit, and dimension constraints."""

    raise NotImplementedError


def extract_query_context(question: Question, config: Config) -> QueryContext:
    """Build the complete structured query context for one question."""

    raise NotImplementedError
