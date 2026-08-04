"""Answer support-score interface."""

from collections.abc import Sequence

from src.config import Config
from src.types import VerificationResult


def compute_support_score(results: Sequence[VerificationResult], config: Config) -> float:
    """Aggregate claim verification results into an answer support score."""

    raise NotImplementedError

