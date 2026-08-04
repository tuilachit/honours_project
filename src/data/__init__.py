"""Dataset adapters returning the shared Question and Chunk schema."""

from src.data.financebench import load_financebench
from src.data.finder import load_finder

__all__ = ["load_financebench", "load_finder"]

