"""FinanceBench external-validation dataset adapter."""

from src.config import Config
from src.types import DatasetCorpus


def load_financebench(config: Config) -> DatasetCorpus:
    """Load a pinned, deduplicated FinanceBench corpus snapshot."""

    raise NotImplementedError
