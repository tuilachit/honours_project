"""T2-RAGBench primary-corpus dataset adapter."""

from src.config import Config
from src.types import DatasetCorpus


def load_t2_ragbench(config: Config) -> DatasetCorpus:
    """Load a pinned, deduplicated T2-RAGBench corpus snapshot."""

    raise NotImplementedError
