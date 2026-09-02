from importlib import import_module
from pathlib import Path

import pytest

MODULES = (
    "src.config",
    "src.results",
    "src.types",
    "src.data.financebench",
    "src.data.t2_ragbench",
    "src.data.splits",
    "src.tables.parse",
    "src.tables.normalize",
    "src.facts.build",
    "src.facts.evidence",
    "src.facts.serialize",
    "src.query.extract",
    "src.retrieval.dense",
    "src.retrieval.sparse",
    "src.retrieval.structured",
    "src.retrieval.fusion",
    "src.retrieval.reranker",
    "src.retrieval.hard_negatives",
    "src.retrieval.training",
    "src.generation.renderer",
    "src.generation.generator",
    "src.eval.retrieval",
    "src.eval.field_errors",
    "src.eval.efficiency",
    "src.eval.statistics",
    "src.eval.agreement",
    "scripts.audit_t2_ragbench",
    "scripts.download_data",
    "scripts.build_pilot_goldset",
    "scripts.run_condition",
    "scripts.make_figures",
)


@pytest.mark.parametrize("module_name", MODULES)
def test_project_module_is_importable(module_name: str) -> None:
    assert import_module(module_name) is not None


def test_verifier_era_packages_are_removed() -> None:
    root = Path(__file__).resolve().parents[1]

    for package in ("claims", "verify", "abstain"):
        assert not any((root / "src" / package).rglob("*.py"))
