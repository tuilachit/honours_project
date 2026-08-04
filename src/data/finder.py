"""FinDER dataset adapter."""

from collections.abc import Sequence

from src.config import Config
from src.types import Chunk, Question


def load_finder(config: Config) -> tuple[Sequence[Question], Sequence[Chunk]]:
    """Load FinDER and normalize it to the common schema."""

    raise NotImplementedError

