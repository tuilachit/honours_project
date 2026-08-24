"""Canonical, schema-flexible financial-fact construction."""

from __future__ import annotations

import re
from collections.abc import Mapping
from decimal import Decimal

from src.config import Config
from src.identity import stable_fact_id
from src.types import FactDimension, FinancialFact, MeasurementUnit, ParsedTable, TimePeriod

YEAR_PATTERN = re.compile(r"(?<!\d)(?:19|20)\d{2}(?!\d)")


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _string(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _measurement(
    raw_value: str,
    table_text: str,
    config: Mapping[str, object],
) -> MeasurementUnit:
    currencies = _mapping(config.get("currency_markers"), "currency markers")
    currency = next(
        (str(value) for token, value in currencies.items() if str(token) in raw_value),
        None,
    )
    percentage_markers = config.get("percentage_markers")
    if not isinstance(percentage_markers, list):
        raise ValueError("percentage_markers must be a list")
    is_percentage = any(str(marker) in raw_value for marker in percentage_markers)
    scales = _mapping(config.get("scale_markers"), "scale markers")
    lower_table = table_text.lower()
    scale: Decimal | None = None
    scale_marker: str | None = None
    for marker, configured_scale in scales.items():
        if str(marker).lower() in lower_table:
            scale = Decimal(str(configured_scale))
            scale_marker = str(marker)
            break
    raw_parts = [
        part for part in (currency, scale_marker, "percent" if is_percentage else None) if part
    ]
    return MeasurementUnit(
        currency=currency,
        unit="percent" if is_percentage else None,
        scale=scale,
        is_percentage=is_percentage,
        raw_text="; ".join(raw_parts) or None,
    )


def build_financial_facts(table: ParsedTable, config: Config) -> tuple[FinancialFact, ...]:
    """Build one addressable fact per parsed numeric cell."""

    representation = _mapping(config.get("representation"), "representation config")
    fact_config = _mapping(representation.get("financial_fact"), "financial_fact config")
    if _string(fact_config, "stable_id_hash_algorithm").lower() != "sha256":
        raise ValueError("Only sha256 fact IDs are supported")
    schema_version = _string(fact_config, "schema_version")
    dimension_prefix = _string(fact_config, "dimension_name_prefix")
    source_metadata = _mapping(table.source.metadata, "source table metadata")
    entity_value = source_metadata.get("entity")
    entity = str(entity_value) if entity_value is not None else None
    facts: list[FinancialFact] = []
    for cell in table.cells:
        concept = " / ".join(cell.address.row_header_path) or None
        raw_period = " / ".join(cell.address.column_header_path) or None
        years = YEAR_PATTERN.findall(raw_period or "")
        period = TimePeriod(
            fiscal_year=int(years[0]) if len(set(years)) == 1 else None,
            raw_text=raw_period,
        )
        measurement = _measurement(cell.raw_value, table.source.raw_text, fact_config)
        dimensions = tuple(
            FactDimension(
                name=f"{dimension_prefix}_{index}",
                value=value,
                raw_name=f"{dimension_prefix}_{index}",
                raw_value=value,
                header_path=cell.address.column_header_path,
            )
            for index, value in enumerate(cell.address.column_header_path)
        )
        identity_payload: dict[str, object] = {
            "cell_id": cell.address.cell_id,
            "entity": entity,
            "concept": concept,
            "period": raw_period,
            "measurement": measurement.raw_text,
            "raw_value": cell.raw_value,
            "dimensions": [(dimension.name, dimension.value) for dimension in dimensions],
        }
        facts.append(
            FinancialFact(
                fact_id=stable_fact_id(schema_version=schema_version, payload=identity_payload),
                source_address=cell.address,
                raw_value=cell.raw_value,
                entity=entity,
                concept=concept,
                period=period,
                measurement=measurement,
                normalized_value=cell.normalized_value,
                dimensions=dimensions,
                qualifiers=cell.qualifiers,
                footnotes=cell.footnotes,
                metadata={"schema_version": schema_version},
            )
        )
    return tuple(facts)
