import hashlib
import json
from decimal import Decimal
from pathlib import Path

from src.config import hash_config
from src.results import write_result_json
from src.types import MeasurementUnit


def test_result_writer_includes_required_provenance(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "result.json"
    config = {"condition": {"id": "m3"}}

    metadata = write_result_json(
        output,
        {"metric": 0.5},
        resolved_config=config,
        repository=Path.cwd(),
    )
    persisted = json.loads(output.read_text(encoding="utf-8"))

    assert persisted["resolved_config"] == config
    assert persisted["dataset_manifest_hashes"] == {}
    assert persisted["result"] == {"metric": 0.5}
    assert persisted["run_metadata"]["config_hash"] == hash_config(config)
    assert persisted["run_metadata"]["git_commit"] == metadata.git_commit
    assert persisted["run_metadata"]["timestamp"] == metadata.timestamp


def test_result_writer_hashes_manifests_and_serializes_domain_values(tmp_path: Path) -> None:
    manifest = tmp_path / "dataset.yaml"
    manifest.write_text("dataset:\n  revision: pinned\n", encoding="utf-8")
    output = tmp_path / "result.json"
    config = {
        "condition": {"id": "b1"},
        "datasets": {"primary": {"manifest_path": str(manifest)}},
    }

    write_result_json(
        output,
        MeasurementUnit(currency="USD", scale=Decimal("1000000")),
        resolved_config=config,
        repository=Path.cwd(),
    )
    persisted = json.loads(output.read_text(encoding="utf-8"))

    expected_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert persisted["dataset_manifest_hashes"] == {"primary": expected_hash}
    assert persisted["result"]["currency"] == "USD"
    assert persisted["result"]["scale"] == "1000000"
