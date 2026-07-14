"""Tests for SP TaskMemoryMiddleware."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from deerflow.sp.memory import TaskMemoryStack
from deerflow.sp.middlewares import TaskMemoryMiddleware
from deerflow.sp.middlewares.task_memory_middleware import SP_TASK_CONTEXT_MESSAGE_NAME


def _make_request(*, messages, state):
    request = MagicMock()
    request.messages = list(messages)
    request.state = state
    request.runtime = SimpleNamespace(context={"thread_id": "thread-1", "run_id": "run-1"}, state=state)
    request.override = lambda **updates: _override_request(request, updates)
    return request


def _override_request(request, updates):
    new = MagicMock()
    new.messages = updates.get("messages", request.messages)
    new.state = request.state
    new.runtime = request.runtime
    new.override = lambda **kw: _override_request(new, kw)
    return new


def _capture_handler():
    captured = []

    def handler(req):
        captured.append(req)
        return "response"

    return captured, handler


def test_before_agent_restores_legacy_memory_stack_into_thread_state_shape():
    middleware = TaskMemoryMiddleware()
    state = {
        "memory_stack": [
            {
                "timestamp": "2026-07-09T12:00:00",
                "action": "human_feedback",
                "agent_type": "human",
                "content": "Keep HITL feedback above summaries",
                "result": {"priority": "HIGHEST"},
            }
        ]
    }

    result = middleware.before_agent(state, Runtime(context={"thread_id": "thread-1", "run_id": "run-1"}))

    assert result is not None
    payload = result["sp_task_memory"]
    assert payload["version"] == 1
    entry = payload["entries"][0]
    assert entry["action"] == "feedback"
    assert entry["priority"] == "critical"
    assert entry["status"] == "pinned"
    assert entry["thread_id"] == "thread-1"
    assert entry["run_id"] == "run-1"


def test_before_agent_leaves_empty_state_untouched():
    result = TaskMemoryMiddleware().before_agent({}, Runtime(context={"thread_id": "thread-1"}))

    assert result is None


def test_wrap_model_call_injects_context_before_latest_user_message_only_for_request():
    stack = TaskMemoryStack()
    stack.append_feedback("User says report artifacts must stay out of ThreadState", stage="planning")
    state = {
        "sp_task_memory": stack.to_dict(),
        "sp_current_stage": "planning",
        "sp_current_artifact_refs": {"report": "artifact://report"},
    }
    request = _make_request(
        messages=[AIMessage(content="Previous answer"), HumanMessage(content="Continue the migration")],
        state=state,
    )
    captured, handler = _capture_handler()

    result = TaskMemoryMiddleware().wrap_model_call(request, handler)

    assert result == "response"
    sent = captured[0]
    assert sent is not request
    assert request.messages[-1].content == "Continue the migration"
    assert sent.messages[-2].name == SP_TASK_CONTEXT_MESSAGE_NAME
    assert sent.messages[-2].additional_kwargs["hide_from_ui"] is True
    assert "User says report artifacts" in sent.messages[-2].content
    assert sent.messages[-1].content == "Continue the migration"


def test_wrap_model_call_skips_empty_context():
    request = _make_request(messages=[HumanMessage(content="Hello")], state={})
    captured, handler = _capture_handler()

    TaskMemoryMiddleware().wrap_model_call(request, handler)

    assert captured[0] is request


def test_wrap_model_call_keeps_compact_summary_on_fresh_user_turn():
    stack = TaskMemoryStack()
    stack.append_summary("Previous task completed; report artifact is ready.", stage="finished")
    state = {"sp_task_memory": stack.to_dict(), "sp_current_stage": "finished"}
    request = _make_request(
        messages=[AIMessage(content="Previous answer"), HumanMessage(content="What did we decide?")],
        state=state,
    )
    request.runtime.context["fresh_user_turn_after_terminal"] = True
    captured, handler = _capture_handler()

    TaskMemoryMiddleware().wrap_model_call(request, handler)

    sent = captured[0]
    assert len(sent.messages) == 2
    assert sent.messages[-1].content == "What did we decide?"
    assert sent.messages[-2].name == SP_TASK_CONTEXT_MESSAGE_NAME
    assert "Previous task completed" in sent.messages[-2].content


def test_after_agent_prunes_active_entries_but_preserves_pinned_feedback():
    stack = TaskMemoryStack()
    stack.append_think("Old normal entry")
    feedback = stack.append_feedback("Pinned human correction")
    stack.append_observe("Another normal entry", actor="researcher")
    state = {"sp_task_memory": stack.to_dict()}

    result = TaskMemoryMiddleware(max_active_entries=1).after_agent(state, Runtime(context={"thread_id": "thread-1"}))

    assert result is not None
    restored = TaskMemoryStack.from_dict(result["sp_task_memory"])
    pinned = restored.get_pinned_entries()
    assert [entry.content for entry in pinned] == [feedback.content]
    assert all(entry.status != "active" for entry in restored.entries if entry.id != feedback.id)
