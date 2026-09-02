"""Frozen generic reranking and future trainable interfaces."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from functools import lru_cache
from typing import cast

import numpy as np
from sentence_transformers import CrossEncoder

from src.config import Config
from src.facts.evidence import render_financial_fact
from src.types import (
    FactCandidate,
    ModelArtifact,
    QueryContext,
    Question,
    RankedFact,
    TrainingPair,
)


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _settings(config: Config) -> Mapping[str, object]:
    return _mapping(config.get("reranker"), "reranker")


def _profile(config: Config) -> tuple[str, str]:
    profile_name = _settings(config).get("model_profile")
    if not isinstance(profile_name, str):
        raise ValueError("reranker.model_profile must be configured")
    profile = _mapping(_mapping(config.get("models"), "models").get(profile_name), profile_name)
    name, revision = profile.get("name"), profile.get("revision")
    if not isinstance(name, str) or not isinstance(revision, str):
        raise ValueError("Reranker model name and immutable revision must be configured")
    return name, revision


def _device(config: Config) -> str:
    value = _mapping(config.get("runtime"), "runtime").get("device")
    if not isinstance(value, str):
        raise ValueError("runtime.device must be a string")
    return value


@lru_cache(maxsize=4)
def _load_reranker(name: str, revision: str, device: str, max_length: int) -> CrossEncoder:
    return CrossEncoder(name, revision=revision, device=device, max_length=max_length)


def rerank_generic(
    question: Question,
    candidates: Sequence[FactCandidate],
    config: Config,
) -> list[RankedFact]:
    """Rerank the configured fused prefix with a pinned generic cross-encoder."""

    settings = _settings(config)
    input_top_k = int(cast(str | int, settings["input_top_k"]))
    return rerank_generic_budgets(question, candidates, (input_top_k,), config)[input_top_k]


def rerank_generic_budgets(
    question: Question,
    candidates: Sequence[FactCandidate],
    budgets: Sequence[int],
    config: Config,
) -> Mapping[int, list[RankedFact]]:
    """Score the largest prefix once and derive rankings for each smaller budget."""

    ordered_budgets = tuple(dict.fromkeys(budgets))
    if not ordered_budgets or any(budget <= 0 for budget in ordered_budgets):
        raise ValueError("Generic reranker budgets must be positive")
    settings = _settings(config)
    input_top_k = int(cast(str | int, settings["input_top_k"]))
    if max(ordered_budgets) > input_top_k:
        raise ValueError("A sensitivity budget exceeds reranker.input_top_k")
    input_candidates = candidates[: max(ordered_budgets)]
    if not input_candidates:
        return {budget: [] for budget in ordered_budgets}
    name, revision = _profile(config)
    model = _load_reranker(
        name,
        revision,
        _device(config),
        int(cast(str | int, settings["max_input_tokens"])),
    )
    pairs = [
        (question.text, render_financial_fact(candidate.fact))
        for candidate in input_candidates
    ]
    raw_scores = model.predict(
        pairs,
        batch_size=int(cast(str | int, settings["batch_size"])),
        show_progress_bar=False,
    )
    scores = np.asarray(raw_scores, dtype=np.float64).reshape(-1)
    scored_in_input_order = list(zip(scores, input_candidates, strict=True))
    output_top_k = int(cast(str | int, settings["output_top_k"]))
    rankings: dict[int, list[RankedFact]] = {}
    for budget in ordered_budgets:
        scored = sorted(
            scored_in_input_order[:budget],
            key=lambda item: (-float(item[0]), item[1].fact.source_address.cell_id),
        )[:output_top_k]
        rankings[budget] = [
            RankedFact(
                question_id=question.question_id,
                fact=candidate.fact,
                rank=rank,
                score=float(score),
                candidate=candidate,
                component_scores={
                    "cross_encoder": float(score),
                    "fusion": float(
                        cast(
                            str | int | float,
                            candidate.metadata.get("fusion_score", 0.0),
                        )
                    ),
                },
            )
            for rank, (score, candidate) in enumerate(scored, start=1)
        ]
    return rankings


def rerank_context_aware(
    question: Question,
    query_context: QueryContext,
    candidates: Sequence[FactCandidate],
    model_artifact: ModelArtifact,
    config: Config,
) -> list[RankedFact]:
    """Reserve the learned schema-flexible reranker for later M2/M3 work."""

    raise NotImplementedError


def train_context_reranker(
    training_pairs: Iterable[TrainingPair],
    seed: int,
    config: Config,
) -> ModelArtifact:
    """Reserve project training for the later M3 experiment."""

    raise NotImplementedError
