"""Conformal-style abstention calibration interface."""

from collections.abc import Sequence

from src.config import Config


def calibrate_threshold(
    calibration_scores: Sequence[float],
    error_labels: Sequence[bool],
    config: Config,
) -> float:
    """Fit an abstention threshold using only the configured calibration split."""

    raise NotImplementedError


def should_abstain(support_score: float, threshold: float, config: Config) -> bool:
    """Apply a calibrated support threshold to one answer."""

    raise NotImplementedError

