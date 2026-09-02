"""Grouped dataset-split construction and leakage-validation interfaces."""

from collections.abc import Mapping

from src.config import Config
from src.types import DatasetCorpus


def build_grouped_splits(
    corpus: DatasetCorpus,
    config: Config,
) -> Mapping[str, tuple[str, ...]]:
    """Group examples by configured company and filing identities before splitting."""

    raise NotImplementedError


def validate_group_isolation(
    split_example_ids: Mapping[str, tuple[str, ...]],
    corpus: DatasetCorpus,
    config: Config,
) -> None:
    """Reject company, filing, table, or derived-family leakage across splits."""

    raise NotImplementedError
