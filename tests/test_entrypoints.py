import argparse
from collections.abc import Callable
from pathlib import Path

import pytest

from scripts.build_pilot_goldset import (
    build_parser as build_pilot_parser,
)
from scripts.build_pilot_goldset import (
    build_pilot_goldset,
)
from scripts.download_data import build_parser as build_data_parser
from scripts.download_data import download_data
from scripts.generate_synthetic_questions import build_parser as build_synthetic_parser
from scripts.make_figures import build_parser as build_figures_parser
from scripts.make_figures import make_figures
from scripts.make_screening_batch import build_parser as build_screening_parser
from scripts.make_source_review_pack import build_parser as build_review_pack_parser
from scripts.run_condition import build_parser as build_condition_parser
from scripts.run_condition import run_condition
from src.config import Config

PARSERS: tuple[Callable[[], argparse.ArgumentParser], ...] = (
    build_data_parser,
    build_synthetic_parser,
    build_pilot_parser,
    build_review_pack_parser,
    build_screening_parser,
    build_condition_parser,
    build_figures_parser,
)


@pytest.mark.parametrize("parser_factory", PARSERS)
def test_entrypoint_requires_base_and_override_configs(
    parser_factory: Callable[[], argparse.ArgumentParser],
) -> None:
    args = parser_factory().parse_args(
        ["--base-config", "configs/base.yaml", "--condition-config", "override.yaml"]
    )

    assert args.base_config == Path("configs/base.yaml")
    assert args.condition_config == Path("override.yaml")


@pytest.mark.parametrize(
    "entrypoint",
    (download_data, run_condition, make_figures),
)
def test_unimplemented_workflow_fails_loudly(entrypoint: Callable[[Config], None]) -> None:
    with pytest.raises(NotImplementedError):
        entrypoint({})


def test_pilot_workflow_rejects_missing_config() -> None:
    with pytest.raises(ValueError, match="pilot must be a mapping"):
        build_pilot_goldset({})
