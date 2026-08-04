"""Risk-coverage metric interfaces."""

from collections.abc import Mapping, Sequence

from src.config import Config


def compute_risk_coverage(
    support_scores: Sequence[float],
    error_labels: Sequence[bool],
    config: Config,
) -> Mapping[str, float | list[float]]:
    """Compute risk-coverage statistics for calibrated abstention."""

    raise NotImplementedError

