"""Deterministic arithmetic verification interface."""

from collections.abc import Sequence

from src.config import Config
from src.types import AtomicClaim, Chunk, VerificationResult


def verify_numeric_claim(
    claim: AtomicClaim,
    evidence: Sequence[Chunk],
    config: Config,
) -> VerificationResult:
    """Verify numeric or threshold claims with deterministic arithmetic."""

    raise NotImplementedError

