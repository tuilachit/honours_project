"""Paired condition-comparison interfaces."""

from collections.abc import Mapping

from src.config import Config


def compare_paired_conditions(
    baseline_hits: Mapping[str, bool],
    proposed_hits: Mapping[str, bool],
    cluster_by_question_id: Mapping[str, str],
    config: Config,
) -> Mapping[str, float | int]:
    """Compare paired question outcomes with the configured clustered analysis."""

    raise NotImplementedError
