"""Create thesis figures from stamped result artifacts."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from src.config import Config, load_config


def build_parser() -> argparse.ArgumentParser:
    """Build the figure-generation command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--condition-config", required=True, type=Path)
    return parser


def make_figures(config: Config) -> None:
    """Render every figure declared by the resolved workflow."""

    raise NotImplementedError


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve the workflow configuration and render its figures."""

    args = build_parser().parse_args(argv)
    config, _ = load_config(
        cast(Path, args.base_config),
        cast(Path, args.condition_config),
    )
    make_figures(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
