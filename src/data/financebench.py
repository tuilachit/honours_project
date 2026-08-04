"""FinanceBench open-subset dataset adapter."""

from collections.abc import Sequence

from src.config import Config
from src.types import Chunk, Question


def load_financebench(config: Config) -> tuple[Sequence[Question], Sequence[Chunk]]:
    """Load FinanceBench questions and filing PDFs into the common schema."""

    raise NotImplementedError

