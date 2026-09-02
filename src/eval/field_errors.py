"""Schema-flexible field mismatch and error-rate interfaces."""

from collections.abc import Mapping, Sequence

from src.config import Config
from src.types import FieldComparison, FinancialFact


def compare_fact_fields(
    predicted: FinancialFact,
    valid_gold_facts: Sequence[FinancialFact],
    config: Config,
) -> FieldComparison:
    """Compare with the configured closest gold, breaking ties by stable cell ID."""

    raise NotImplementedError


def compute_field_error_rates(
    comparisons: Sequence[FieldComparison],
    config: Config,
) -> Mapping[str, float]:
    """Aggregate core keys and dynamic ``wrong_dimension.<canonical_name>`` keys."""

    raise NotImplementedError
