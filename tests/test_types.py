from decimal import Decimal

from src.types import (
    CellAddress,
    DatasetCorpus,
    DatasetExample,
    EvidenceGranularity,
    EvidenceUnit,
    FactDimension,
    FinancialFact,
    MeasurementUnit,
    Question,
    SourceTable,
    TimePeriod,
)


def test_financial_fact_keeps_dynamic_dimensions_and_raw_structure() -> None:
    address = CellAddress(
        cell_id="filing-1:table-2:r4:c3",
        dataset="t2-ragbench",
        document_id="filing-1",
        table_id="table-2",
        row_index=4,
        column_index=3,
        row_header_path=("Revenue", "Cloud services"),
        column_header_path=("Year ended", "2024"),
    )
    fact = FinancialFact(
        fact_id="fact-1",
        source_address=address,
        raw_value="$12.4",
        normalized_value=Decimal("12.4"),
        entity="Example Corp",
        concept="Revenue",
        period=TimePeriod(fiscal_year=2024, raw_text="Year ended 2024"),
        measurement=MeasurementUnit(currency="USD", scale=Decimal("1000000")),
        dimensions=(
            FactDimension(name="product_family", value="Cloud services"),
            FactDimension(name="accounting_basis", value="GAAP"),
        ),
    )

    assert fact.source_address.row_header_path == ("Revenue", "Cloud services")
    assert {dimension.name for dimension in fact.dimensions} == {
        "product_family",
        "accounting_basis",
    }


def test_flattened_evidence_can_trace_every_contained_fact() -> None:
    evidence = EvidenceUnit(
        evidence_id="chunk-1",
        granularity=EvidenceGranularity.FLAT_CHUNK,
        text="A flattened multi-year table",
        dataset="t2-ragbench",
        fact_ids=("fact-2023", "fact-2024"),
    )

    assert evidence.fact_ids == ("fact-2023", "fact-2024")


def test_corpus_reuses_one_source_table_across_multiple_questions() -> None:
    table = SourceTable(
        dataset="t2-ragbench",
        raw_text="| Metric | 2024 |",
        table_id="table-1",
        context_id="context-1",
    )
    examples = tuple(
        DatasetExample(
            example_id=f"example-{index}",
            question=Question(
                question_id=f"question-{index}",
                dataset="t2-ragbench",
                text="What was the reported value?",
            ),
            source_table_ids=(table.table_id,),
        )
        for index in (1, 2)
    )
    corpus = DatasetCorpus(
        dataset="t2-ragbench",
        examples=examples,
        source_tables=(table,),
        revision="pinned-revision",
    )

    assert len(corpus.examples) == 2
    assert len(corpus.source_tables) == 1
