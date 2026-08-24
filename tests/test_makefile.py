import subprocess
from pathlib import Path

import pytest

TARGETS = (
    "setup",
    "data",
    "audit-data",
    "pilot",
    "review-pack",
    "screening-batch",
    "synthetic-500",
    "fact-roundtrip",
    "run-b0",
    "run-b1",
    "run-b2",
    "run-m1",
    "run-m2",
    "run-m3",
    "eval",
    "figures",
    "test",
    "clean",
)


@pytest.mark.parametrize("target", TARGETS)
def test_make_target_resolves_without_execution(target: str) -> None:
    repository = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        ["make", "--dry-run", target],
        cwd=repository,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
