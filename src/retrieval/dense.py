"""Pinned SentenceTransformer dense indexing and retrieval."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any, cast

import numpy as np
from sentence_transformers import SentenceTransformer

from src.config import Config, hash_config
from src.facts.evidence import evidence_from_payload
from src.types import EvidenceUnit, IndexArtifact, Question, RetrievalCandidate, RetrievalRoute


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _dense_config(config: Config) -> Mapping[str, object]:
    return _mapping(_mapping(config.get("retrieval"), "retrieval").get("dense"), "dense")


def _model_profile(config: Config) -> tuple[str, str]:
    profile_name = _dense_config(config).get("model_profile")
    if not isinstance(profile_name, str):
        raise ValueError("retrieval.dense.model_profile must be configured")
    profile = _mapping(_mapping(config.get("models"), "models").get(profile_name), profile_name)
    name, revision = profile.get("name"), profile.get("revision")
    if not isinstance(name, str) or not isinstance(revision, str):
        raise ValueError("Dense model name and immutable revision must be configured")
    return name, revision


def _device(config: Config) -> str:
    value = _mapping(config.get("runtime"), "runtime").get("device")
    if not isinstance(value, str):
        raise ValueError("runtime.device must be a string")
    return value


@lru_cache(maxsize=4)
def _load_encoder(name: str, revision: str, device: str) -> SentenceTransformer:
    return SentenceTransformer(name, revision=revision, device=device)


def _encode(model: SentenceTransformer, texts: list[str], config: Config) -> np.ndarray[Any, Any]:
    dense = _dense_config(config)
    encoded = model.encode(
        texts,
        batch_size=int(cast(str | int, dense["batch_size"])),
        convert_to_numpy=True,
        normalize_embeddings=cast(bool, dense["normalize_embeddings"]),
        show_progress_bar=False,
    )
    return np.asarray(encoded, dtype=np.float32)


def _index_dir(config: Config, granularity: str) -> Path:
    root = _mapping(config.get("indexes"), "indexes").get("dense_path")
    if not isinstance(root, str):
        raise ValueError("indexes.dense_path must be a path string")
    path = Path(root)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    return path / granularity / hash_config(config)


@lru_cache(maxsize=4)
def _load_index(directory_value: str) -> tuple[list[EvidenceUnit], np.ndarray[Any, Any]]:
    directory = Path(directory_value)
    raw_evidence = json.loads((directory / "evidence.json").read_text(encoding="utf-8"))
    if not isinstance(raw_evidence, list):
        raise ValueError("Dense evidence index must be a list")
    evidence = [
        evidence_from_payload(item) for item in raw_evidence if isinstance(item, Mapping)
    ]
    if len(evidence) != len(raw_evidence):
        raise ValueError("Dense evidence index contains a non-mapping entry")
    return evidence, np.load(directory / "embeddings.npy", allow_pickle=False)


def build_dense_index(
    evidence_units: Iterable[EvidenceUnit],
    config: Config,
) -> IndexArtifact:
    """Encode and persist a homogeneous evidence corpus with pinned model weights."""

    evidence = tuple(evidence_units)
    if not evidence:
        raise ValueError("Cannot build a dense index over an empty corpus")
    granularity = evidence[0].granularity
    if any(item.granularity is not granularity for item in evidence):
        raise ValueError("A retrieval index cannot mix evidence granularities")
    name, revision = _model_profile(config)
    embeddings = _encode(
        _load_encoder(name, revision, _device(config)),
        [item.text for item in evidence],
        config,
    )
    directory = _index_dir(config, granularity.value)
    directory.mkdir(parents=True, exist_ok=True)
    evidence_path = directory / "evidence.json"
    embedding_path = directory / "embeddings.npy"
    evidence_path.write_text(
        json.dumps([asdict(item) for item in evidence], sort_keys=True, ensure_ascii=False),
        encoding="utf-8",
    )
    np.save(embedding_path, embeddings, allow_pickle=False)
    _load_index.cache_clear()
    digest = hashlib.sha256(evidence_path.read_bytes())
    digest.update(embedding_path.read_bytes())
    return IndexArtifact(
        artifact_id=digest.hexdigest(),
        route=RetrievalRoute.DENSE,
        granularity=granularity,
        location=str(directory),
        corpus_size=len(evidence),
        config_hash=hash_config(config),
        metadata={"model_name": name, "model_revision": revision},
    )


def retrieve_dense(
    question: Question,
    index: IndexArtifact,
    config: Config,
) -> list[RetrievalCandidate]:
    """Rank evidence by cosine-equivalent dot product over normalized vectors."""

    evidence, embeddings = _load_index(index.location)
    name, revision = _model_profile(config)
    query = _encode(_load_encoder(name, revision, _device(config)), [question.text], config)[0]
    scores = embeddings @ query
    top_k = int(cast(str | int, _dense_config(config)["top_k"]))
    ranked = sorted(
        ((float(score), unit) for score, unit in zip(scores, evidence, strict=True)),
        key=lambda item: (-item[0], item[1].evidence_id),
    )[:top_k]
    return [
        RetrievalCandidate(
            question_id=question.question_id,
            evidence=unit,
            route_scores={RetrievalRoute.DENSE: score},
            route_ranks={RetrievalRoute.DENSE: rank},
        )
        for rank, (score, unit) in enumerate(ranked, start=1)
    ]
