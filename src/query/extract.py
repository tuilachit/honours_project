"""Deterministic schema-flexible query-constraint extraction."""

from __future__ import annotations

import re
from collections.abc import Mapping

from src.config import Config
from src.types import ConstraintState, QueryConstraint, QueryContext, Question

_YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")
_POSSESSIVE_ENTITY_PATTERN = re.compile(
    r"(?P<entity>[A-Z][A-Za-z0-9&.-]*"
    r"(?:\s+(?:[A-Z][A-Za-z0-9&.-]*|&)){0,6}"
    r"(?:,\s*(?:Inc|Ltd|Corp)\.)?)(?:'s|'|’s|’)\b"
)
_PREPOSITION_ENTITY_PATTERN = re.compile(
    r"\b(?:for|by|of)\s+"
    r"(?P<entity>[A-Z][A-Za-z0-9&.-]*"
    r"(?:\s+(?:[A-Z][A-Za-z0-9&.-]*|&)){0,7}"
    r"(?:,\s*(?:Inc|Ltd|Corp)\.)?)"
    r"(?=\s+(?:and\s+subsidiaries\s+)?(?:in|during|at|for|as|according|which)\b|[?,]|$)"
)
_SOURCE_CLAUSE_PATTERN = re.compile(
    r",\s*(?:"
    r"as\s+(?:reflected|reported|outlined|disclosed|calculated|presented|shown|used)\b|"
    r"according\s+to\b|"
    r"which\s+(?:includes|reflects|contains)\b|"
    r"as\s+part\s+of\b"
    r")",
    re.IGNORECASE,
)
_DIMENSION_NAMES = (
    "segment",
    "division",
    "region",
    "geography",
    "product",
    "scenario",
    "category",
)
_CONCEPT_STOPWORDS = {
    "a",
    "according",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "during",
    "fiscal",
    "for",
    "financial",
    "from",
    "in",
    "is",
    "of",
    "on",
    "reported",
    "report",
    "reflected",
    "table",
    "the",
    "to",
    "using",
    "was",
    "were",
    "what",
    "year",
}


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _extractor(config: Config) -> tuple[str, str]:
    query_context = _mapping(config.get("query_context"), "query_context")
    if query_context.get("enabled") is not True or query_context.get("mode") != "automatic":
        raise ValueError("Automatic query context must be enabled")
    extractor = _mapping(query_context.get("extractor"), "query_context.extractor")
    implementation = extractor.get("implementation")
    revision = extractor.get("revision")
    if (
        not isinstance(implementation, str)
        or implementation != "deterministic_regex_v1"
        or not isinstance(revision, str)
    ):
        raise ValueError("A pinned deterministic query extractor must be configured")
    return implementation, revision


def _constraint(
    name: str,
    value: str | None,
    *,
    method: str,
    raw_value: str | None = None,
) -> QueryConstraint:
    return QueryConstraint(
        name=name,
        state=ConstraintState.EXPLICIT if value else ConstraintState.UNKNOWN,
        value=value,
        raw_name=name,
        raw_value=raw_value,
        normalization_version="deterministic_regex_v1",
        metadata={"method": method},
    )


def _primary_clause(text: str) -> str:
    match = _SOURCE_CLAUSE_PATTERN.search(text)
    return text if match is None else text[: match.start()]


def _entity(primary_text: str, full_text: str) -> tuple[str | None, str | None]:
    for pattern in (_POSSESSIVE_ENTITY_PATTERN, _PREPOSITION_ENTITY_PATTERN):
        match = pattern.search(primary_text)
        if match is None:
            continue
        raw = match.group("entity").strip(" ,.")
        if raw:
            return raw, match.group(0)
    for pattern in (_PREPOSITION_ENTITY_PATTERN, _POSSESSIVE_ENTITY_PATTERN):
        match = pattern.search(full_text)
        if match is None:
            continue
        raw = match.group("entity").strip(" ,.")
        if raw:
            return raw, match.group(0)
    return None, None


def _dimension(text: str, entity: str | None, name: str) -> str | None:
    pattern = re.compile(
        rf"\b(?:for|in|within|from|of)\s+(?:the\s+)?"
        rf"(?P<value>[A-Za-z0-9&.'’() -]{{1,100}}?)\s+{re.escape(name)}\b",
        re.IGNORECASE,
    )
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    value = matches[-1].group("value").strip(" ,.")
    value = re.split(
        r"\b(?:for|in|within|from|of)\b",
        value,
        flags=re.IGNORECASE,
    )[-1].strip(" ,.")
    if not value:
        return None
    if entity:
        value = re.sub(
            rf"^{re.escape(entity)}(?:'s|'|’s|’)?\s+",
            "",
            value,
            flags=re.IGNORECASE,
        )
    return re.sub(r"^the\s+", "", value, flags=re.IGNORECASE) or None


def _concept(text: str, entity: str | None, dimensions: Mapping[str, str]) -> str | None:
    tokens = [token.lower() for token in _TOKEN_PATTERN.findall(text)]
    excluded = set(_CONCEPT_STOPWORDS)
    excluded.update(year.lower() for year in _YEAR_PATTERN.findall(text))
    if entity:
        excluded.update(token.lower() for token in _TOKEN_PATTERN.findall(entity))
    for name, value in dimensions.items():
        excluded.add(name.lower())
        excluded.update(token.lower() for token in _TOKEN_PATTERN.findall(value))
    content = [
        token
        for token in tokens
        if token not in excluded and not token.isdigit() and len(token) > 1
    ]
    return " ".join(dict.fromkeys(content)) or None


def _currency(text: str) -> str | None:
    lowered = text.lower()
    if "$" in text or "dollar" in lowered or "usd" in lowered:
        return "$"
    if "£" in text or "pound sterling" in lowered or "gbp" in lowered:
        return "£"
    if "€" in text or "euro" in lowered or "eur" in lowered:
        return "€"
    return None


def _scale(text: str) -> str | None:
    lowered = text.lower()
    for marker, value in (
        ("billion", "1000000000"),
        ("million", "1000000"),
        ("thousand", "1000"),
    ):
        if re.search(
            rf"\b(?:in|expressed\s+in|reported\s+in|stated\s+in)\s+{marker}s?\b",
            lowered,
        ):
            return value
    return None


def extract_query_constraints(
    question: Question,
    config: Config,
) -> tuple[QueryConstraint, ...]:
    """Extract configured entity, concept, time, unit, and dimension constraints."""

    _extractor(config)
    primary_text = _primary_clause(question.text)
    entity, entity_span = _entity(primary_text, question.text)
    years = tuple(dict.fromkeys(_YEAR_PATTERN.findall(primary_text)))
    period = years[-1] if years else None
    dimensions = {
        name: value
        for name in _DIMENSION_NAMES
        if (value := _dimension(primary_text, entity, name)) is not None
    }
    constraints = [
        _constraint("entity", entity, method="possessive_or_for_phrase", raw_value=entity_span),
        _constraint(
            "concept",
            _concept(primary_text, entity, dimensions),
            method="content_tokens",
            raw_value=primary_text,
        ),
        _constraint(
            "period",
            period,
            method="last_explicit_year_before_source_clause",
            raw_value=period,
        ),
        _constraint("currency", _currency(question.text), method="currency_marker"),
        _constraint("scale", _scale(primary_text), method="answer_unit_scale_phrase"),
        _constraint(
            "percentage",
            "true" if "%" in question.text or "percent" in question.text.lower() else None,
            method="percentage_marker",
        ),
    ]
    constraints.extend(
        _constraint(name, value, method="named_dimension_phrase", raw_value=value)
        for name, value in sorted(dimensions.items())
    )
    return tuple(constraints)


def extract_query_context(question: Question, config: Config) -> QueryContext:
    """Build the complete structured query context for one question."""

    implementation, revision = _extractor(config)
    constraints = extract_query_constraints(question, config)
    return QueryContext(
        question_id=question.question_id,
        constraints=constraints,
        metadata={
            "implementation": implementation,
            "revision": revision,
            "explicit_constraint_count": sum(
                constraint.state is ConstraintState.EXPLICIT for constraint in constraints
            ),
        },
    )
