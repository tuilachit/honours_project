"""Deterministic BM25 indexing and retrieval."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import cast

from src.config import Config, hash_config
from src.facts.evidence import evidence_from_payload
from src.types import (
    EvidenceGranularity,
    EvidenceUnit,
    IndexArtifact,
    Question,
    RetrievalCandidate,
    RetrievalRoute,
)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _sparse_config(config: Config) -> Mapping[str, object]:
    return _mapping(_mapping(config.get("retrieval"), "retrieval").get("sparse"), "sparse")


def _tokenize(text: str, config: Config) -> list[str]:
    sparse = _sparse_config(config)
    pattern = sparse.get("token_pattern")
    if not isinstance(pattern, str):
        raise ValueError("retrieval.sparse.token_pattern must be a string")
    if sparse.get("lowercase") is True:
        text = text.lower()
    return re.findall(pattern, text)


def _index_dir(config: Config, granularity: EvidenceGranularity) -> Path:
    indexes = _mapping(config.get("indexes"), "indexes")
    root = indexes.get("sparse_path")
    if not isinstance(root, str):
        raise ValueError("indexes.sparse_path must be a path string")
    path = Path(root)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    return path / granularity.value / hash_config(config)


def build_sparse_index(
    evidence_units: Iterable[EvidenceUnit],
    config: Config,
) -> IndexArtifact:
    """Persist evidence and corpus statistics needed for exact BM25 scoring."""

    evidence = tuple(evidence_units)
    if not evidence:
        raise ValueError("Cannot build a sparse index over an empty corpus")
    granularity = evidence[0].granularity
    if any(item.granularity is not granularity for item in evidence):
        raise ValueError("A retrieval index cannot mix evidence granularities")
    tokenized = [_tokenize(item.text, config) for item in evidence]
    document_frequency: Counter[str] = Counter()
    for tokens in tokenized:
        document_frequency.update(set(tokens))
    payload = {
        "evidence": [asdict(item) for item in evidence],
        "term_frequencies": [dict(Counter(tokens)) for tokens in tokenized],
        "document_lengths": [len(tokens) for tokens in tokenized],
        "document_frequency": dict(document_frequency),
        "average_document_length": sum(map(len, tokenized)) / len(tokenized),
    }
    directory = _index_dir(config, granularity)
    directory.mkdir(parents=True, exist_ok=True)
    location = directory / "index.json"
    location.write_text(json.dumps(payload, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    _load_index.cache_clear()
    artifact_id = hashlib.sha256(location.read_bytes()).hexdigest()
    return IndexArtifact(
        artifact_id=artifact_id,
        route=RetrievalRoute.SPARSE,
        granularity=granularity,
        location=str(location),
        corpus_size=len(evidence),
        config_hash=hash_config(config),
    )


@lru_cache(maxsize=4)
def _load_index(location: str) -> Mapping[str, object]:
    payload = json.loads(Path(location).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Sparse index payload must be a mapping")
    return payload


def retrieve_sparse(
    question: Question,
    index: IndexArtifact,
    config: Config,
) -> list[RetrievalCandidate]:
    """Rank indexed evidence with the configured Robertson BM25 formula."""

    payload = _load_index(index.location)
    evidence_payloads = payload["evidence"]
    term_frequencies = payload["term_frequencies"]
    lengths = payload["document_lengths"]
    document_frequency = payload["document_frequency"]
    if not isinstance(evidence_payloads, list) or not isinstance(term_frequencies, list):
        raise ValueError("Sparse index arrays are invalid")
    if not isinstance(lengths, list) or not isinstance(document_frequency, Mapping):
        raise ValueError("Sparse index statistics are invalid")
    sparse = _sparse_config(config)
    k1 = float(cast(str | int | float, sparse["k1"]))
    b = float(cast(str | int | float, sparse["b"]))
    top_k = int(cast(str | int, sparse["top_k"]))
    minimum_score = float(cast(str | int | float, sparse["minimum_score_exclusive"]))
    average_length = float(cast(str | int | float, payload["average_document_length"]))
    corpus_size = len(evidence_payloads)
    query_terms = tuple(dict.fromkeys(_tokenize(question.text, config)))
    scored: list[tuple[float, EvidenceUnit]] = []
    for raw_evidence, raw_tf, raw_length in zip(
        evidence_payloads, term_frequencies, lengths, strict=True
    ):
        if not isinstance(raw_evidence, Mapping) or not isinstance(raw_tf, Mapping):
            raise ValueError("Sparse index entry is invalid")
        score = 0.0
        document_length = float(raw_length)
        for term in query_terms:
            frequency = float(raw_tf.get(term, 0))
            if frequency == 0:
                continue
            df = float(document_frequency.get(term, 0))
            inverse_document_frequency = math.log(1.0 + (corpus_size - df + 0.5) / (df + 0.5))
            denominator = frequency + k1 * (
                1.0 - b + b * document_length / max(average_length, 1.0)
            )
            score += inverse_document_frequency * frequency * (k1 + 1.0) / denominator
        if score > minimum_score:
            scored.append((score, evidence_from_payload(raw_evidence)))
    ranked = sorted(scored, key=lambda item: (-item[0], item[1].evidence_id))[:top_k]
    return [
        RetrievalCandidate(
            question_id=question.question_id,
            evidence=evidence,
            route_scores={RetrievalRoute.SPARSE: score},
            route_ranks={RetrievalRoute.SPARSE: rank},
        )
        for rank, (score, evidence) in enumerate(ranked, start=1)
    ]
