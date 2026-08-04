"""Verifier miss-rate interfaces."""

from collections.abc import Mapping, Sequence

from src.config import Config
from src.types import AtomicClaim, VerificationResult


def compute_miss_rates(
    claims: Sequence[AtomicClaim],
    predictions: Sequence[VerificationResult],
    gold_labels: Mapping[str, bool],
    config: Config,
) -> Mapping[str, float]:
    """Compute verifier miss-rate disaggregated by claim type."""

    raise NotImplementedError

