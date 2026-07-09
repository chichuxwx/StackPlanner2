"""Tests for SP prompt context rendering."""

from deerflow.sp.memory import TaskMemoryStack
from deerflow.sp.prompt import PromptContextBuilder


def test_context_prioritizes_pinned_human_feedback_before_recent_memory():
    stack = TaskMemoryStack()
    stack.append_think("Try the default lead-agent flow first", stage="planning")
    stack.append_feedback("User requires CentralAgent to route actions through handlers", stage="planning")

    context = PromptContextBuilder().build(stack, current_stage="planning")

    assert context.index("critical_feedback:") < context.index("recent_task_memory:")
    assert context.index("User requires CentralAgent") < context.index("Try the default")
    assert "priority_rules:" in context


def test_context_is_bounded_and_clips_entry_content():
    stack = TaskMemoryStack()
    stack.append_observe("x" * 1000, actor="researcher", result_ref="artifact://research-1")

    context = PromptContextBuilder(max_chars=1200, max_entry_chars=80).build(
        stack,
        artifact_refs={"research": "artifact://research-1"},
    )

    assert len(context) <= 1200
    assert "...<truncated>" in context
    assert "artifact://research-1" in context


def test_context_uses_refs_without_requiring_artifact_bodies():
    stack = TaskMemoryStack()
    stack.append_delegate("Ask reporter to draft from artifact refs only", result_ref="artifact://outline")

    context = PromptContextBuilder().build(
        stack,
        artifact_refs={"outline": "artifact://outline", "report": "artifact://report"},
        report_version="v1",
    )

    assert "artifact://outline" in context
    assert "artifact://report" in context
    assert "current_report_version: v1" in context
    assert "do not infer large artifact bodies" in context
