from decimal import Decimal

from src.config import Config
from src.facts import (
    build_financial_facts,
    deserialize_financial_fact,
    serialize_financial_fact,
)
from src.tables import normalize_table, parse_source_table
from src.tables.normalize import numeric_values, parse_numeric_value
from src.types import SourceTable


def _config() -> Config:
    return {
        "table_parsing": {
            "source_format": "markdown",
            "markdown_min_pipe_count": 2,
            "numeric_cells_only": True,
        },
        "normalisation": {"preserve_raw_values": True},
        "representation": {
            "financial_fact": {
                "schema_version": "test-v1",
                "dimension_name_prefix": "column_header",
                "currency_markers": {"$": "$", "£": "£", "€": "€"},
                "percentage_markers": ["%"],
                "scale_markers": {"million": "1000000", "thousand": "1000"},
                "stable_id_hash_algorithm": "sha256",
            }
        },
    }


def _source() -> SourceTable:
    return SourceTable(
        dataset="T2-RAGBench",
        raw_text="""\
| USD in millions | Year ended | Year ended |
| | 2024 (a) | 2023 |
|---|---:|---:|
| Revenue | $1,200 | $1,100 |
| Margin | 20% | 19% |
""",
        table_id="table-1",
        document_id="filing.pdf",
        context_id="context-1",
        page_number=10,
        source_format="markdown",
        metadata={
            "dataset_revision": "revision-1",
            "subset": "TAT-DQA",
            "manifest_split": "test",
            "context_id": "context-1",
            "table_index": 0,
            "entity": "Example Corp.",
            "drop_leading_synthetic_column": False,
            "cell_id_hash_algorithm": "sha256",
        },
    )


def test_table_wide_fact_build_preserves_address_headers_value_and_unit() -> None:
    config = _config()

    parsed = normalize_table(parse_source_table(_source(), config), config)
    facts = build_financial_facts(parsed, config)

    assert len(facts) == 4
    revenue_2024 = facts[0]
    assert revenue_2024.concept == "Revenue"
    assert revenue_2024.period is not None
    assert revenue_2024.period.fiscal_year == 2024
    assert revenue_2024.period.raw_text == "Year ended / 2024"
    assert revenue_2024.raw_value == "$1,200"
    assert revenue_2024.normalized_value == Decimal("1200")
    assert revenue_2024.measurement is not None
    assert revenue_2024.measurement.currency == "$"
    assert revenue_2024.measurement.scale == Decimal("1000000")
    assert parsed.metadata["gold_used_for_parsing"] is False


def test_financial_fact_serialization_roundtrip_is_exact() -> None:
    config = _config()
    parsed = normalize_table(parse_source_table(_source(), config), config)
    fact = build_financial_facts(parsed, config)[-1]

    restored = deserialize_financial_fact(serialize_financial_fact(fact, config), config)

    assert restored == fact
    assert restored.measurement is not None
    assert restored.measurement.is_percentage is True


def test_numeric_parser_preserves_accounting_negative_semantics() -> None:
    assert numeric_values("-11.0 (11.0)") == [Decimal("-11.0"), Decimal("-11.0")]
    assert parse_numeric_value("-11.0 (11.0)") == Decimal("-11.0")
