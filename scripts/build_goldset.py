"""Build the hand-labelling artifact used for verifier evaluation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Create the configured gold-set annotation artifact."""

    build_parser().parse_args(argv)
    raise NotImplementedError


if __name__ == "__main__":
    raise SystemExit(main())

