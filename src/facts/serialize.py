"""Lossless financial-fact serialisation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from decimal import Decimal
from typing import cast

from src.config import Config
from src.types import (
    CellAddress,
    FactDimension,
    FinancialFact,
    MeasurementUnit,
    Metadata,
    TimePeriod,
)


def _fact_config(config: Config) -> Mapping[str, object]:
    representation = config.get("representation")
    if not isinstance(representation, Mapping):
        raise ValueError("representation config must be a mapping")
    fact_config = representation.get("financial_fact")
    if not isinstance(fact_config, Mapping):
        raise ValueError("financial_fact config must be a mapping")
    return fact_config


def _schema_version(config: Config) -> str:
    value = _fact_config(config).get("schema_version")
    if not isinstance(value, str) or not value:
        raise ValueError("financial_fact.schema_version must be a non-empty string")
    return value


def _encode(value: object) -> object:
    if isinstance(value, Decimal):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return _encode(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _encode(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_encode(item) for item in value]
    return value


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _required_string(mapping: Mapping[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value


def _optional_string(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("Optional string field has a non-string value")
    return value


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError("Optional integer field has a non-integer value")
    return value


def _optional_bool(value: object) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError("Optional boolean field has a non-boolean value")
    return value


def _optional_decimal(value: object) -> Decimal | None:
    if value is None:
        return None
    if not isinstance(value, str | int | float):
        raise ValueError("Decimal field must be a string or number")
    return Decimal(str(value))


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    if not all(isinstance(item, str) for item in value):
        raise ValueError(f"{label} must contain only strings")
    return tuple(cast(list[str], value))


def _metadata(value: object, label: str) -> Metadata:
    return dict(_mapping(value, label))


def serialize_financial_fact(fact: FinancialFact, config: Config) -> Mapping[str, object]:
    """Convert a financial fact to a JSON-safe, versioned representation."""

    encoded = _encode(fact)
    if not isinstance(encoded, dict):
        raise TypeError("Encoded financial fact must be a mapping")
    return {"schema_version": _schema_version(config), "fact": encoded}


def deserialize_financial_fact(
    payload: Mapping[str, object],
    config: Config,
) -> FinancialFact:
    """Restore a financial fact and every nested provenance field."""

    if payload.get("schema_version") != _schema_version(config):
        raise ValueError("FinancialFact schema version mismatch")
    fact = _mapping(payload.get("fact"), "financial fact")
    address_payload = _mapping(fact.get("source_address"), "source address")
    address = CellAddress(
        cell_id=_required_string(address_payload, "cell_id"),
        dataset=_required_string(address_payload, "dataset"),
        document_id=_optional_string(address_payload.get("document_id")),
        context_id=_optional_string(address_payload.get("context_id")),
        page_number=_optional_int(address_payload.get("page_number")),
        table_id=_optional_string(address_payload.get("table_id")),
        row_index=_optional_int(address_payload.get("row_index")),
        column_index=_optional_int(address_payload.get("column_index")),
        row_header_path=_string_tuple(address_payload.get("row_header_path"), "row headers"),
        column_header_path=_string_tuple(
            address_payload.get("column_header_path"), "column headers"
        ),
        metadata=_metadata(address_payload.get("metadata"), "address metadata"),
    )
    period_payload = fact.get("period")
    period: TimePeriod | None = None
    if period_payload is not None:
        period_mapping = _mapping(period_payload, "period")
        period = TimePeriod(
            fiscal_year=_optional_int(period_mapping.get("fiscal_year")),
            fiscal_period=_optional_string(period_mapping.get("fiscal_period")),
            start_date=_optional_string(period_mapping.get("start_date")),
            end_date=_optional_string(period_mapping.get("end_date")),
            instant_date=_optional_string(period_mapping.get("instant_date")),
            period_kind=_optional_string(period_mapping.get("period_kind")),
            raw_text=_optional_string(period_mapping.get("raw_text")),
            metadata=_metadata(period_mapping.get("metadata"), "period metadata"),
        )
    measurement_payload = fact.get("measurement")
    measurement: MeasurementUnit | None = None
    if measurement_payload is not None:
        measurement_mapping = _mapping(measurement_payload, "measurement")
        measurement = MeasurementUnit(
            currency=_optional_string(measurement_mapping.get("currency")),
            unit=_optional_string(measurement_mapping.get("unit")),
            scale=_optional_decimal(measurement_mapping.get("scale")),
            is_percentage=_optional_bool(measurement_mapping.get("is_percentage")),
            raw_text=_optional_string(measurement_mapping.get("raw_text")),
            metadata=_metadata(measurement_mapping.get("metadata"), "measurement metadata"),
        )
    raw_dimensions = fact.get("dimensions")
    if not isinstance(raw_dimensions, list):
        raise ValueError("dimensions must be a list")
    dimensions: list[FactDimension] = []
    for raw_dimension in raw_dimensions:
        dimension = _mapping(raw_dimension, "dimension")
        dimensions.append(
            FactDimension(
                name=_required_string(dimension, "name"),
                value=_optional_string(dimension.get("value")),
                raw_name=_optional_string(dimension.get("raw_name")),
                raw_value=_optional_string(dimension.get("raw_value")),
                normalization_version=_optional_string(dimension.get("normalization_version")),
                header_path=_string_tuple(dimension.get("header_path"), "dimension headers"),
                qualifiers=_string_tuple(dimension.get("qualifiers"), "dimension qualifiers"),
                metadata=_metadata(dimension.get("metadata"), "dimension metadata"),
            )
        )
    return FinancialFact(
        fact_id=_required_string(fact, "fact_id"),
        source_address=address,
        raw_value=_required_string(fact, "raw_value"),
        entity=_optional_string(fact.get("entity")),
        concept=_optional_string(fact.get("concept")),
        period=period,
        measurement=measurement,
        normalized_value=_optional_decimal(fact.get("normalized_value")),
        dimensions=tuple(dimensions),
        qualifiers=_string_tuple(fact.get("qualifiers"), "fact qualifiers"),
        footnotes=_string_tuple(fact.get("footnotes"), "fact footnotes"),
        metadata=_metadata(fact.get("metadata"), "fact metadata"),
    )
