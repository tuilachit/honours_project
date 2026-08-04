"""Atomic claim decomposition interface."""

from src.config import Config
from src.types import Answer, AtomicClaim


def decompose_claims(answer: Answer, config: Config) -> list[AtomicClaim]:
    """Decompose an answer into minimal independently verifiable claims."""

    raise NotImplementedError

