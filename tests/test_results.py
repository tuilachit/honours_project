import hashlib
import json
import subprocess
from decimal import Decimal
from pathlib import Path
from typing import cast

from src.config import Config, hash_config
from src.hashing import sha256_json
from src.results import create_run_metadata, write_result_json
from src.types import MeasurementUnit


def test_json_content_hash_ignores_envelope_metadata_when_payload_is_selected() -> None:
    questions = [{"question_id": "q1", "text": "Revenue in 2024?"}]
    first = {"run_metadata": {"timestamp": "one"}, "result": {"questions": questions}}
    second = {"run_metadata": {"timestamp": "two"}, "result": {"questions": questions}}

    first_result = cast(dict[str, object], first["result"])
    second_result = cast(dict[str, object], second["result"])

    assert sha256_json(first_result["questions"]) == sha256_json(
        second_result["questions"]
    )


def test_result_writer_includes_required_provenance(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "result.json"
    config: Config = {"condition": {"id": "m3"}}

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
    config: Config = {
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


def test_run_metadata_fingerprints_tracked_and_untracked_worktree_changes(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test Researcher"],
        cwd=repository,
        check=True,
    )
    tracked = repository / "tracked.txt"
    tracked.write_text("version one\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "initial"], cwd=repository, check=True)

    clean = create_run_metadata({"condition": {"id": "b2"}}, repository)
    assert clean.git_dirty is False
    assert clean.git_worktree_sha256 is None

    tracked.write_text("version two\n", encoding="utf-8")
    tracked_change = create_run_metadata({"condition": {"id": "b2"}}, repository)
    assert tracked_change.git_dirty is True
    assert tracked_change.git_worktree_sha256 is not None
    assert len(tracked_change.git_worktree_sha256) == 64

    (repository / "untracked.txt").write_text("new evidence\n", encoding="utf-8")
    untracked_change = create_run_metadata({"condition": {"id": "b2"}}, repository)
    assert untracked_change.git_worktree_sha256 != tracked_change.git_worktree_sha256
