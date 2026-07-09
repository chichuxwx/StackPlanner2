"""Tests for SP mid-term artifact adapter."""

from pathlib import Path

import pytest

from deerflow.sp.artifacts import SPArtifactAdapter, SPArtifactMetadata


def _state(tmp_path: Path) -> dict:
    return {
        "thread_data": {
            "outputs_path": str(tmp_path / "threads" / "thread-1" / "user-data" / "outputs"),
        }
    }


def test_write_text_artifact_stores_body_in_outputs_and_returns_lightweight_refs(tmp_path):
    adapter = SPArtifactAdapter()
    state = _state(tmp_path)

    result = adapter.write_text_artifact(
        "# Report\n\nLarge body",
        artifact_type="report",
        state=state,
        thread_id="thread-1",
        run_id="run-1",
        created_by="reporter",
        stage="reporting",
    )

    ref = result.state_update["sp_current_artifact_refs"]["report"]
    file_path = Path(result.metadata.workspace_path)
    assert file_path.read_text(encoding="utf-8") == "# Report\n\nLarge body"
    assert ref["virtual_path"].startswith("/mnt/user-data/outputs/sp/report/")
    assert ref["artifact_url"].startswith("/api/threads/thread-1/artifacts/")
    assert ref["summary"] == "# Report Large body"
    assert "Large body" not in result.state_update
    assert result.state_update["artifacts"] == [ref["virtual_path"]]
    assert result.state_update["sp_current_report_version"] == "1"


def test_write_text_artifact_versions_current_ref_and_keeps_history(tmp_path):
    adapter = SPArtifactAdapter()
    state = _state(tmp_path)
    first = adapter.write_text_artifact("Outline v1", artifact_type="outline", state=state, thread_id="thread-1")
    second = adapter.write_text_artifact(
        "Outline v2",
        artifact_type="outline",
        state={**state, **first.state_update},
        thread_id="thread-1",
    )

    refs = second.state_update["sp_current_artifact_refs"]
    assert refs["outline"]["version"] == 2
    assert [item["version"] for item in refs["_history"] if item["type"] == "outline"] == [1, 2]


def test_migrate_legacy_midterm_state_externalizes_large_fields(tmp_path):
    adapter = SPArtifactAdapter()
    update = adapter.migrate_legacy_midterm_state(
        {
            "report_outline": "## Outline",
            "observations": ["observation one", "observation two"],
            "data_collections": [{"source": "search", "content": "result"}],
            "final_report": "# Final report",
            "memory_stack": "must stay out of artifact migration",
        },
        state=_state(tmp_path),
        thread_id="thread-1",
        run_id="run-1",
    )

    refs = update["sp_current_artifact_refs"]
    assert set(refs) >= {"outline", "research_observation", "data_collection", "report_revision", "_history"}
    assert len(update["artifacts"]) == 4
    assert "report_outline" not in update
    assert "final_report" not in update
    assert "observations" not in update
    for virtual_path in update["artifacts"]:
        assert virtual_path.startswith("/mnt/user-data/outputs/sp/")


def test_metadata_ref_is_json_safe_and_excludes_workspace_path_from_thread_ref():
    metadata = SPArtifactMetadata(
        type="data_collection",
        workspace_path="/tmp/secret-host-path/data.json",
        virtual_path="/mnt/user-data/outputs/sp/data/data.json",
        metadata={"set": {"b", "a"}},
    )

    ref = metadata.to_ref()

    assert "workspace_path" not in ref
    assert ref["metadata"] == {"set": ["a", "b"]}


def test_write_text_artifact_requires_dr2_thread_context_when_no_outputs_path():
    with pytest.raises(ValueError, match="thread_data.outputs_path or thread_id"):
        SPArtifactAdapter().write_text_artifact("body", artifact_type="report", state={})
