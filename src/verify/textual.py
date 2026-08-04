"""NLI plus LLM-judge textual verification interface."""

from collections.abc import Sequence

from src.config import Config
from src.types import AtomicClaim, Chunk, VerificationResult


def verify_textual_claim(
    claim: AtomicClaim,
    evidence: Sequence[Chunk],
    config: Config,
) -> VerificationResult:
    """Verify a claim using the configured NLI and LLM-judge pipeline."""

    raise NotImplementedError

