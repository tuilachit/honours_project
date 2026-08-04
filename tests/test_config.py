from pathlib import Path

from src.config import hash_config, load_config


def test_config_hash_is_stable_across_mapping_order() -> None:
    assert hash_config({"b": 2, "a": 1}) == hash_config({"a": 1, "b": 2})


def test_condition_config_is_merged_with_base() -> None:
    config, config_hash = load_config(Path("configs/base.yaml"), Path("configs/c2b.yaml"))

    assert config["condition"] == {
        "id": "c2b",
        "description": "baseline RAG plus type-routed verification",
    }
    assert config["verification"]["routing"] == "by_claim_type"  # type: ignore[index]
    assert config["project"]["seed"] == 20260804  # type: ignore[index]
    assert config_hash == hash_config(config)

