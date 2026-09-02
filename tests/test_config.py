from copy import deepcopy
from pathlib import Path
from typing import cast

import pytest

import scripts.run_condition as condition_runner
from src.config import ConfigValue, hash_config, load_config
from src.types import EvidenceGranularity, RetrievalRoute

CONDITIONS = (
    (
        "configs/b0_flattened_hybrid.yaml",
        "b0",
        "flattened_chunk",
        (True, True, False),
        "none",
        False,
        "none",
        True,
        False,
    ),
    (
        "configs/b1_fact_hybrid.yaml",
        "b1",
        "financial_fact",
        (True, True, False),
        "generic_cross_encoder",
        False,
        "none",
        True,
        True,
    ),
    (
        "configs/b2_structured_lookup.yaml",
        "b2",
        "financial_fact",
        (False, False, True),
        "structured_metric_matcher",
        False,
        "none",
        True,
        True,
    ),
    (
        "configs/m1_candidate_union.yaml",
        "m1",
        "financial_fact",
        (True, True, True),
        "generic_cross_encoder",
        False,
        "none",
        False,
        True,
    ),
    (
        "configs/m2_context_aware.yaml",
        "m2",
        "financial_fact",
        (True, True, True),
        "schema_flexible_context_aware",
        True,
        "ordinary",
        True,
        True,
    ),
    (
        "configs/m3_hard_negative.yaml",
        "m3",
        "financial_fact",
        (True, True, True),
        "schema_flexible_context_aware",
        True,
        "natural_finance_hard",
        True,
        True,
    ),
)


def _mapping(value: ConfigValue) -> dict[str, ConfigValue]:
    assert isinstance(value, dict)
    return cast(dict[str, ConfigValue], value)


def test_config_hash_is_stable_across_mapping_order() -> None:
    assert hash_config({"b": 2, "a": 1}) == hash_config({"a": 1, "b": 2})


@pytest.mark.parametrize(
    (
        "condition_path",
        "condition_id",
        "evidence_unit",
        "routes",
        "reranker_kind",
        "training_enabled",
        "negative_strategy",
        "generation_enabled",
        "exact_cell_metrics",
    ),
    CONDITIONS,
)
def test_condition_resolves_to_the_frozen_truth_table(
    condition_path: str,
    condition_id: str,
    evidence_unit: str,
    routes: tuple[bool, bool, bool],
    reranker_kind: str,
    training_enabled: bool,
    negative_strategy: str,
    generation_enabled: bool,
    exact_cell_metrics: bool,
) -> None:
    config, config_hash = load_config(Path("configs/base.yaml"), Path(condition_path))
    condition = _mapping(config["condition"])
    retrieval = _mapping(config["retrieval"])
    route_config = _mapping(retrieval["routes"])
    reranker = _mapping(config["reranker"])
    training = _mapping(config["training"])
    generation = _mapping(config["generation"])
    evaluation = _mapping(config["evaluation"])

    assert condition["id"] == condition_id
    assert condition["evidence_unit"] == evidence_unit
    assert evidence_unit in {item.value for item in EvidenceGranularity}
    assert (
        route_config["dense"],
        route_config["sparse"],
        route_config["structured"],
    ) == routes
    assert set(route_config) == {item.value for item in RetrievalRoute}
    assert reranker["kind"] == reranker_kind
    assert training["enabled"] is training_enabled
    assert training["negative_strategy"] == negative_strategy
    assert generation["llm_enabled"] is generation_enabled
    assert generation["deterministic_renderer_enabled"] is (condition_id != "b0")
    assert evaluation["exact_cell_metrics_enabled"] is exact_cell_metrics
    assert condition["produces_exact_cell_ranking"] is exact_cell_metrics
    assert config_hash == hash_config(config)

    for legacy_key in ("claims", "verification", "abstention"):
        assert legacy_key not in config


@pytest.mark.parametrize("workflow_id", ("data", "pilot", "figures"))
def test_operational_workflow_uses_the_same_base_override_loader(workflow_id: str) -> None:
    config, _ = load_config(
        Path("configs/base.yaml"),
        Path("configs/workflows") / f"{workflow_id}.yaml",
    )

    assert _mapping(config["workflow"])["id"] == workflow_id


def test_consumed_datasets_are_pinned_to_immutable_revisions() -> None:
    config, _ = load_config(
        Path("configs/base.yaml"),
        Path("configs/b1_fact_hybrid.yaml"),
    )
    datasets = _mapping(config["datasets"])
    primary = _mapping(datasets["primary"])
    external = _mapping(datasets["external"])

    assert primary["revision"] == "adf7fe1541ac37351ce1142544d8e3b43010ed92"
    assert external["revision"] == "e04404e3a97f69f79c14d42f24981a1c9c3bcd18"


def test_b2_and_proposed_conditions_share_the_same_query_context_profile() -> None:
    profiles = []
    for condition_path in (
        "configs/b2_structured_lookup.yaml",
        "configs/m1_candidate_union.yaml",
        "configs/m2_context_aware.yaml",
        "configs/m3_hard_negative.yaml",
    ):
        config, _ = load_config(Path("configs/base.yaml"), Path(condition_path))
        profiles.append(config["query_context"])

    assert profiles.count(profiles[0]) == len(profiles)


def test_b2_has_a_pinned_extractor_and_development_run_contract() -> None:
    config, _ = load_config(
        Path("configs/base.yaml"),
        Path("configs/b2_structured_lookup.yaml"),
    )
    query_context = _mapping(config["query_context"])
    extractor = _mapping(query_context["extractor"])
    retrieval_pilot = _mapping(config["retrieval_pilot"])

    assert extractor["implementation"] == "deterministic_regex_v1"
    assert extractor["revision"] == "1"
    assert retrieval_pilot["enabled"] is True
    assert retrieval_pilot["output_path"] == "results/runs/b2_development_pilot.json"
    assert _mapping(config["evaluation"])["downstream_answer_metrics_enabled"] is False


def test_m1_has_a_development_run_and_budget_matched_contract() -> None:
    config, _ = load_config(
        Path("configs/base.yaml"),
        Path("configs/m1_candidate_union.yaml"),
    )
    retrieval_pilot = _mapping(config["retrieval_pilot"])
    retrieval = _mapping(config["retrieval"])
    sensitivity = _mapping(retrieval["budget_matched_sensitivity"])

    assert retrieval_pilot["enabled"] is True
    assert retrieval_pilot["output_path"] == "results/runs/m1_development_pilot.json"
    assert sensitivity["enabled"] is True
    assert sensitivity["final_candidate_counts"] == [20, 50, 100]
    assert _mapping(config["evaluation"])["downstream_answer_metrics_enabled"] is False
    assert _mapping(config["generation"])["llm_enabled"] is False


@pytest.mark.parametrize(
    ("section", "key", "unsupported"),
    (
        ("training", "enabled", True),
        ("training", "project_training", True),
        ("training", "negative_strategy", "ordinary"),
        ("generation", "llm_enabled", True),
        ("evaluation", "downstream_answer_metrics_enabled", True),
    ),
)
def test_m1_rejects_unexecuted_training_and_generation_stages(
    section: str,
    key: str,
    unsupported: ConfigValue,
) -> None:
    config, _ = load_config(
        Path("configs/base.yaml"),
        Path("configs/m1_candidate_union.yaml"),
    )
    changed = deepcopy(config)
    _mapping(changed[section])[key] = unsupported

    with pytest.raises(NotImplementedError, match="retrieval-only no-training contract"):
        condition_runner._validate_execution_contract(changed, "m1")


@pytest.mark.parametrize(
    ("key", "unsupported"),
    (
        ("ranking_strategy", "unimplemented_strategy"),
        ("unknown_field_policy", "penalise_unknown"),
    ),
)
def test_b2_rejects_unimplemented_structured_policies(
    key: str,
    unsupported: str,
) -> None:
    config, _ = load_config(
        Path("configs/base.yaml"),
        Path("configs/b2_structured_lookup.yaml"),
    )
    changed = deepcopy(config)
    retrieval = _mapping(changed["retrieval"])
    structured = _mapping(retrieval["structured"])
    structured[key] = unsupported

    with pytest.raises(NotImplementedError, match="structured lookup policy"):
        condition_runner._validate_execution_contract(changed, "b2")


def test_m2_and_m3_change_negative_strategy_not_model_architecture() -> None:
    m2, _ = load_config(Path("configs/base.yaml"), Path("configs/m2_context_aware.yaml"))
    m3, _ = load_config(Path("configs/base.yaml"), Path("configs/m3_hard_negative.yaml"))

    for key in ("representation", "query_context", "retrieval", "reranker", "generation"):
        assert m2[key] == m3[key]

    m2_training = dict(_mapping(m2["training"]))
    m3_training = dict(_mapping(m3["training"]))
    assert m2_training.pop("negative_strategy") == "ordinary"
    assert m3_training.pop("negative_strategy") == "natural_finance_hard"
    assert m2_training == m3_training


def test_grouped_split_policy_blocks_company_and_filing_leakage() -> None:
    config, _ = load_config(
        Path("configs/base.yaml"),
        Path("configs/m3_hard_negative.yaml"),
    )
    splits = _mapping(config["splits"])

    assert splits["group_keys"] == ["company_id", "filing_id", "derived_family_id"]
    assert splits["prevent_table_leakage"] is True
