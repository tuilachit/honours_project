"""Build the human-review artifact for the exact-cell feasibility pilot."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from src.config import Config, load_config


def build_parser() -> argparse.ArgumentParser:
    """Build the pilot gold-set command-line parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", required=True, type=Path)
    parser.add_argument("--condition-config", required=True, type=Path)
    return parser


def build_pilot_goldset(config: Config) -> None:
    """Create the configured exact-cell annotation artifact."""

    raise NotImplementedError


def main(argv: Sequence[str] | None = None) -> int:
    """Resolve the workflow configuration and build the pilot artifact."""

    args = build_parser().parse_args(argv)
    config, _ = load_config(
        cast(Path, args.base_config),
        cast(Path, args.condition_config),
    )
    build_pilot_goldset(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
