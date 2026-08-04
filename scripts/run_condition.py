"""Run one configured experimental condition."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute one end-to-end condition run."""

    build_parser().parse_args(argv)
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())

