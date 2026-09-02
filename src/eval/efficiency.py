"""Retrieval efficiency measurement interfaces."""

from collections.abc import Mapping, Sequence

from src.config import Config
from src.types import IndexArtifact


def compute_efficiency_metrics(
    query_latencies_seconds: Sequence[float],
    index_artifacts: Sequence[IndexArtifact],
    config: Config,
) -> Mapping[str, float]:
    """Summarize latency, throughput, index size, and configured resource measures."""

    raise NotImplementedError
