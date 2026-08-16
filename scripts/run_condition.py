"""Run one configured experimental condition."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from src.config import Config, load_config


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser shared by all condition runs."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--condition-config", required=True, type=Path)
    return parser


def run_condition(config: Config) -> None:
    """Execute one resolved B0--B2 or M1--M3 condition."""

    raise NotImplementedError


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve configuration and dispatch one end-to-end condition run."""

    args = build_parser().parse_args(argv)
    config, _ = load_config(
        cast(Path, args.base_config),
        cast(Path, args.condition_config),
    )
    run_condition(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
