"""Claim-type classification interface."""

from src.config import Config
from src.types import ClaimType


def classify_claim_type(claim_text: str, config: Config) -> ClaimType:
    """Classify a claim as textual or numeric for verifier routing."""

    raise NotImplementedError

