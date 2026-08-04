import json
from pathlib import Path

from src.config import hash_config
from src.results import write_result_json


def test_result_writer_includes_required_provenance(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "result.json"
    config = {"condition": {"id": "c1"}}

    metadata = write_result_json(
        output,
        {"metric": 0.5},
        resolved_config=config,
        repository=Path.cwd(),
    )
    persisted = json.loads(output.read_text(encoding="utf-8"))

    assert persisted["resolved_config"] == config
    assert persisted["result"] == {"metric": 0.5}
    assert persisted["run_metadata"]["config_hash"] == hash_config(config)
    assert persisted["run_metadata"]["git_commit"] == metadata.git_commit
    assert persisted["run_metadata"]["timestamp"] == metadata.timestamp
