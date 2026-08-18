from typing import Any

import pytest

from scripts.make_screening_batch import select_candidates


def _candidate(subset: str, question_id: str) -> dict[str, Any]:
    return {
        "subset": subset,
        "split": "train",
        "id": question_id,
    }


def test_selection_is_deterministic_stratified_and_excludes_prior_ids() -> None:
    candidates = [
        _candidate(subset, f"{subset}-{index}")
        for subset in ("A", "B")
        for index in range(5)
    ]
    excluded_ids = {"A-0", "B-0"}
    subset_targets = {"A": 2, "B": 2}

    first = select_candidates(
        candidates,
        excluded_ids=excluded_ids,
        subset_targets=subset_targets,
        seed=42,
    )
    second = select_candidates(
        list(reversed(candidates)),
        excluded_ids=excluded_ids,
        subset_targets=subset_targets,
        seed=42,
    )

    assert [item["id"] for item in first] == [item["id"] for item in second]
    assert {item["id"] for item in first}.isdisjoint(excluded_ids)
    assert [item["subset"] for item in first].count("A") == 2
    assert [item["subset"] for item in first].count("B") == 2


def test_selection_rejects_duplicate_candidate_ids() -> None:
    candidates = [_candidate("A", "same"), _candidate("B", "same")]

    with pytest.raises(ValueError, match="Duplicate automatic candidate ID"):
        select_candidates(
            candidates,
            excluded_ids=set(),
            subset_targets={"A": 1},
            seed=42,
        )
