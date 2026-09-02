"""Build deterministic retrieval text without consulting gold labels."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from src.config import Config
from src.hashing import sha256_mapping
from src.types import EvidenceGranularity, EvidenceUnit, FinancialFact, SourceTable


def _string_list(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list | tuple) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label} must contain only strings")
    return tuple(value)


def evidence_from_payload(payload: Mapping[str, object]) -> EvidenceUnit:
    """Restore a persisted evidence unit independently of any retrieval route."""

    fact_ids = payload.get("fact_ids")
    metadata = payload.get("metadata")
    if not isinstance(fact_ids, list) or not all(isinstance(item, str) for item in fact_ids):
        raise ValueError("Indexed fact_ids must be a string list")
    page_number = payload.get("page_number")
    if page_number is not None and not isinstance(page_number, int):
        raise ValueError("Indexed page_number must be an integer or null")
    return EvidenceUnit(
        evidence_id=str(payload["evidence_id"]),
        granularity=EvidenceGranularity(str(payload["granularity"])),
        text=str(payload["text"]),
        dataset=str(payload["dataset"]),
        document_id=None if payload.get("document_id") is None else str(payload["document_id"]),
        page_number=page_number,
        fact_ids=tuple(fact_ids),
        metadata=dict(metadata) if isinstance(metadata, Mapping) else {},
    )


def render_financial_fact(fact: FinancialFact) -> str:
    """Render all available identity fields and raw provenance as searchable text."""

    fields: list[tuple[str, str]] = []
    if fact.entity:
        fields.append(("entity", fact.entity))
    if fact.concept:
        fields.append(("concept", fact.concept))
    if fact.period and fact.period.raw_text:
        fields.append(("period", fact.period.raw_text))
    if fact.measurement and fact.measurement.raw_text:
        fields.append(("measurement", fact.measurement.raw_text))
    for dimension in fact.dimensions:
        if dimension.value:
            fields.append((dimension.name, dimension.value))
    if fact.source_address.row_header_path:
        fields.append(("row headers", " > ".join(fact.source_address.row_header_path)))
    if fact.source_address.column_header_path:
        fields.append(("column headers", " > ".join(fact.source_address.column_header_path)))
    fields.append(("reported value", fact.raw_value))
    return " | ".join(f"{name}: {value}" for name, value in fields)


def build_flattened_evidence(
    source_tables: Iterable[SourceTable],
    config: Config,
) -> Iterable[EvidenceUnit]:
    """Build one complete-table B0 unit and attach fact IDs for evaluation only."""

    pilot = config.get("retrieval_pilot")
    if not isinstance(pilot, Mapping) or pilot.get("table_evidence_policy") != (
        "one_complete_source_table"
    ):
        raise ValueError("B0 requires the one_complete_source_table evidence policy")
    representation = config.get("representation")
    if not isinstance(representation, Mapping):
        raise ValueError("representation config must be a mapping")
    flattened = representation.get("flattened_chunk")
    if not isinstance(flattened, Mapping) or flattened.get("target_characters") is not None:
        raise ValueError("Complete-table B0 must disable character-based chunking")
    for table in source_tables:
        fact_ids = _string_list(table.metadata.get("fact_ids", ()), "source-table fact_ids")
        cell_ids = _string_list(table.metadata.get("cell_ids", ()), "source-table cell_ids")
        yield EvidenceUnit(
            evidence_id=sha256_mapping(
                {
                    "granularity": EvidenceGranularity.FLAT_CHUNK.value,
                    "dataset": table.dataset,
                    "table_id": table.table_id,
                    "raw_text": table.raw_text,
                }
            ),
            granularity=EvidenceGranularity.FLAT_CHUNK,
            text=table.raw_text,
            dataset=table.dataset,
            document_id=table.document_id,
            page_number=table.page_number,
            fact_ids=fact_ids,
            metadata={
                "table_id": table.table_id,
                "context_id": table.context_id,
                "cell_ids": cell_ids,
            },
        )


def build_fact_evidence(
    facts: Iterable[FinancialFact],
    config: Config,
) -> Iterable[EvidenceUnit]:
    """Build one canonical evidence unit per source fact."""

    del config
    for fact in facts:
        yield EvidenceUnit(
            evidence_id=sha256_mapping(
                {
                    "granularity": EvidenceGranularity.FINANCIAL_FACT.value,
                    "fact_id": fact.fact_id,
                }
            ),
            granularity=EvidenceGranularity.FINANCIAL_FACT,
            text=render_financial_fact(fact),
            dataset=fact.source_address.dataset,
            document_id=fact.source_address.document_id,
            page_number=fact.source_address.page_number,
            fact_ids=(fact.fact_id,),
            metadata={
                "cell_id": fact.source_address.cell_id,
                "table_id": fact.source_address.table_id,
            },
        )
