import pytest

from scripts.build_pilot_goldset import parse_table_blocks, stable_cell_id


def test_parse_table_blocks_removes_separator_and_synthetic_index() -> None:
    markdown = """\
|   | Metric | 2024 | 2023 |
|---|---|---:|---:|
| 0 | Revenue | 12 | 11 |
| 1 | Profit | 4 | 3 |
"""

    blocks = parse_table_blocks(
        markdown,
        min_pipe_count=2,
        drop_leading_synthetic_column=True,
    )

    assert blocks == (
        (
            ("Metric", "2024", "2023"),
            ("Revenue", "12", "11"),
            ("Profit", "4", "3"),
        ),
    )


def test_parse_table_blocks_preserves_blank_cells() -> None:
    markdown = """\
| Metric | 2024 | 2023 |
|---|---:|---:|
| Revenue | 12 | |
"""

    blocks = parse_table_blocks(
        markdown,
        min_pipe_count=2,
        drop_leading_synthetic_column=False,
    )

    assert blocks[0][1] == ("Revenue", "12", "")


def test_stable_cell_id_is_deterministic_and_address_sensitive() -> None:
    first = stable_cell_id(
        dataset_revision="revision-1",
        subset="FinQA",
        manifest_split="train",
        context_id="context-1",
        table_index=0,
        row_index=2,
        column_index=3,
    )
    second = stable_cell_id(
        dataset_revision="revision-1",
        subset="FinQA",
        manifest_split="train",
        context_id="context-1",
        table_index=0,
        row_index=2,
        column_index=3,
    )
    changed = stable_cell_id(
        dataset_revision="revision-1",
        subset="FinQA",
        manifest_split="train",
        context_id="context-1",
        table_index=0,
        row_index=2,
        column_index=4,
    )

    assert len(first) == 64
    assert first == second
    assert first != changed


def test_stable_cell_id_rejects_negative_coordinates() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        stable_cell_id(
            dataset_revision="revision-1",
            subset="FinQA",
            manifest_split="train",
            context_id="context-1",
            table_index=0,
            row_index=-1,
            column_index=3,
        )
