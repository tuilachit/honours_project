"""Human/verifier agreement statistic interfaces."""

from collections.abc import Mapping, Sequence

from src.config import Config


def compute_agreement_statistics(
    annotator_labels: Sequence[Sequence[str]],
    config: Config,
) -> Mapping[str, float]:
    """Compute the configured inter-annotator or verifier agreement statistics."""

    raise NotImplementedError

