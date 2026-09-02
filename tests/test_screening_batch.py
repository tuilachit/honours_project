from pathlib import Path
from typing import Any

import pytest
import yaml

from scripts.make_screening_batch import _assistant_suggestions, select_candidates


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


def test_assistant_suggestions_are_loaded_without_becoming_human_decisions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "decisions.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "decisions": [
                    {
                        "question_id": "q1",
                        "assistant_suggestion": "exclude",
                        "assistant_reason": "Requires arithmetic.",
                        "decision": None,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert _assistant_suggestions(path) == {
        "q1": ("exclude", "Requires arithmetic.")
    }


def test_assistant_suggestion_requires_a_reason(tmp_path: Path) -> None:
    path = tmp_path / "decisions.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "decisions": [
                    {
                        "question_id": "q1",
                        "assistant_suggestion": "include",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Missing assistant reason"):
        _assistant_suggestions(path)
