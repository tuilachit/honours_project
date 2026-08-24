from typing import Any

import pytest

from scripts.generate_synthetic_questions import (
    _candidate_review_hash,
    _display_entity,
    _header_path_parts,
    _parse_block,
    _replace_company_mention,
    select_synthetic_questions,
)


def _candidate(
    question_id: str,
    *,
    subset: str,
    company: str,
    question: str,
    rank: str,
) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "question": question,
        "selection_rank": rank,
        "source": {"subset": subset, "company_name": company},
    }


def test_parse_block_removes_separator_and_synthetic_index() -> None:
    block = [
        "| | metric | 2019 | 2018 |",
        "|---:|---|---:|---:|",
        "| 0 | revenue | 10 | 9 |",
        "| 1 | profit | 2 | 1 |",
    ]

    parsed = _parse_block(block, drop_leading_column=True)

    assert parsed is not None
    assert parsed.rows == (
        ("metric", "2019", "2018"),
        ("revenue", "10", "9"),
        ("profit", "2", "1"),
    )


def test_header_path_preserves_each_raw_header_level() -> None:
    rows = (
        ("", "Three months ended", "Three months ended"),
        ("", "2024", "2023"),
        ("Revenue", "10", "9"),
    )

    assert _header_path_parts(rows, 2, 1) == ("Three months ended", "2024")


def test_slug_company_name_is_made_readable_without_llm_rewriting() -> None:
    display = _display_entity("te-connectivity-ltd")

    assert display == "TE Connectivity Ltd."
    assert (
        _replace_company_mention(
            "What was te-connectivity-ltd's revenue?",
            "te-connectivity-ltd",
            display,
        )
        == "What was TE Connectivity Ltd.'s revenue?"
    )


def test_selection_is_ranked_unique_and_company_capped() -> None:
    candidates = [
        _candidate("q1", subset="A", company="One", question="First?", rank="1"),
        _candidate("q2", subset="A", company="One", question="Second?", rank="2"),
        _candidate("q3", subset="A", company="Two", question="FIRST?", rank="3"),
        _candidate("q4", subset="A", company="Two", question="Fourth?", rank="4"),
    ]

    selected = select_synthetic_questions(
        candidates,
        targets={"A": 2, "Unused": 0},
        maximum_per_company=1,
    )

    assert [item["question_id"] for item in selected] == ["q1", "q4"]


def test_selection_rejects_non_positive_company_cap() -> None:
    with pytest.raises(ValueError, match="at least one"):
        select_synthetic_questions([], targets={"A": 0}, maximum_per_company=0)


def test_review_hash_changes_when_question_text_changes() -> None:
    question: dict[str, Any] = {
        "question_id": "q1",
        "question": "What was revenue in 2024?",
        "answer": "10",
        "source": {
            "dataset": "T2-RAGBench",
            "subset": "TAT-DQA",
            "manifest_split": "train",
            "source_record_id": "source-1",
            "context_id": "context-1",
            "table_index": 0,
            "table_sha256": "abc",
        },
        "target_cell": {"cell_id": "cell-1", "raw_value": "10"},
        "natural_hard_negatives": [{"cell_id": "cell-2", "raw_value": "9"}],
    }
    first_hash = _candidate_review_hash(question)

    question["question"] = "According to the table, what was revenue in 2024?"

    assert _candidate_review_hash(question) != first_hash
