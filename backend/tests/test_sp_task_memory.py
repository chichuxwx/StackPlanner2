"""Unit tests for StackPlanner task memory primitives."""

from __future__ import annotations

import json

from deerflow.sp.memory import StackMemoryEntry, TaskMemoryStack


def test_legacy_memory_stack_json_restores_feedback_as_pinned_critical():
    legacy = json.dumps(
        [
            {
                "timestamp": "2026-07-09T12:00:00",
                "action": "human_feedback",
                "agent_type": "human",
                "content": "Put the macro pipeline first",
                "result": {"priority": "HIGHEST"},
            }
        ],
    )

    stack = TaskMemoryStack.from_legacy_memory_stack(legacy, thread_id="thread-1", run_id="run-1")

    assert stack.size() == 1
    entry = stack.entries[0]
    assert entry.action == "feedback"
    assert entry.actor == "human"
    assert entry.priority == "critical"
    assert entry.status == "pinned"
    assert entry.thread_id == "thread-1"
    assert entry.run_id == "run-1"
    assert entry.metadata["legacy_action"] == "human_feedback"


def test_legacy_delegate_with_result_becomes_observation():
    stack = TaskMemoryStack.from_legacy_memory_stack(
        [
            {
                "timestamp": "2026-07-09T12:00:00",
                "action": "delegate",
                "agent_type": "researcher",
                "content": "Research DeerFlow ThreadState",
                "result": {"summary": "ThreadState is runtime-first"},
            }
        ]
    )

    entry = stack.entries[0]
    assert entry.action == "observe"
    assert entry.actor == "researcher"
    assert entry.metadata["legacy_action"] == "delegate"
    assert entry.metadata["legacy_result"] == {"summary": "ThreadState is runtime-first"}


def test_append_feedback_forces_pinned_critical_even_when_kwargs_disagree():
    stack = TaskMemoryStack()

    entry = stack.append_feedback("User confirmed the outline", priority="low", status="active")

    assert entry.priority == "critical"
    assert entry.status == "pinned"
    assert stack.get_pinned_entries() == [entry]


def test_condense_marks_source_entries_without_touching_pinned_feedback():
    stack = TaskMemoryStack()
    first = stack.append_think("Initial plan")
    feedback = stack.append_feedback("Keep this feedback")
    second = stack.append_observe("Research result", actor="researcher")

    summary = stack.condense([first.id, feedback.id, second.id], "Stage summary")

    assert first.status == "condensed"
    assert feedback.status == "pinned"
    assert second.status == "condensed"
    assert summary.action == "summarize"
    assert summary.parent_ids == [first.id, feedback.id, second.id]


def test_prune_preserves_pinned_feedback():
    stack = TaskMemoryStack()
    old = stack.append_think("Old path")
    feedback = stack.append_feedback("User feedback must remain")
    stack.append_observe("Observation", actor="researcher")

    stack.prune(max_entries=1)

    assert old.status == "pruned"
    assert feedback.status == "pinned"
    assert feedback in stack.get_active_entries()


def test_serialize_round_trip_uses_thread_state_shape():
    stack = TaskMemoryStack()
    stack.append(StackMemoryEntry(id="spmem_custom", action="think", content="Keep it structured"))

    restored = TaskMemoryStack.from_dict(stack.to_dict())

    assert restored.to_dict()["version"] == 1
    assert restored.entries[0].id == "spmem_custom"
    assert restored.entries[0].content == "Keep it structured"
