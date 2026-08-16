"""Gold exact-cell annotation agreement interfaces."""

from collections.abc import Mapping, Sequence

from src.config import Config
from src.types import GoldCellLabel


def compute_gold_annotation_agreement(
    first_annotator: Sequence[GoldCellLabel],
    second_annotator: Sequence[GoldCellLabel],
    config: Config,
) -> Mapping[str, float]:
    """Measure exact-address and observed field-level agreement before adjudication."""

    raise NotImplementedError
