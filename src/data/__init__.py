"""Deduplicated dataset-corpus and grouped-split interfaces."""

from src.data.financebench import load_financebench
from src.data.splits import build_grouped_splits, validate_group_isolation
from src.data.t2_ragbench import load_t2_ragbench

__all__ = [
    "build_grouped_splits",
    "load_financebench",
    "load_t2_ragbench",
    "validate_group_isolation",
]
