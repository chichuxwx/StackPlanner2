"""Tests for SP Action schema, router, and state-only handlers."""

import pytest

from deerflow.sp.actions import ActionType, ActionValidationError, SPAction, build_default_action_router
from deerflow.sp.central import CENTRAL_AGENT_ACTION_PROMPT
from deerflow.sp.hitl import record_human_feedback
from deerflow.sp.memory import TaskMemoryStack
from deerflow.sp.subagents import SPSubagentResult, SPSubagentStatus, SPSubagentTask


def _action(action_type: ActionType | str, **kwargs):
    return SPAction.create(action_type, reason=kwargs.pop("reason", "unit-test reason"), **kwargs)


def test_action_schema_requires_action_id_and_normalizes_type():
    with pytest.raises(ActionValidationError, match="action_id"):
        SPAction.from_dict({"action_type": "think", "reason": "missing id"})

    action = SPAction.from_dict(
        {
            "action_id": "act-1",
            "action_type": "think",
            "reason": "Need a plan",
            "task": "Plan next step",
        }
    )

    assert action.action_type == ActionType.THINK
    assert action.idempotency_key.startswith("spidem_")


def test_delegate_schema_requires_target_agent_and_task():
    with pytest.raises(ActionValidationError, match="target_agent"):
        SPAction.create(ActionType.DELEGATE, reason="Need research", task="Research")
    with pytest.raises(ActionValidationError, match="task"):
        SPAction.create(ActionType.DELEGATE, reason="Need research", target_agent="researcher")


def test_recall_memory_schema_requires_query_or_task():
    with pytest.raises(ActionValidationError, match="memory_query"):
        SPAction.create(ActionType.RECALL_MEMORY, reason="Need prior context")

    action = SPAction.create(
        ActionType.RECALL_MEMORY,
        action_id="act-recall-schema",
        reason="Need prior context",
        metadata={"memory_query": "migration preferences"},
    )

    assert action.action_type == ActionType.RECALL_MEMORY


def test_router_executes_think_and_records_events_and_thread_state():
    router = build_default_action_router()
    action = _action(ActionType.THINK, action_id="act-think", idempotency_key="idem-think", task="Inspect state", stage="planning")

    result = router.execute(action, state={}, thread_id="thread-1", run_id="run-1")

    assert result.next_step == "continue"
    assert result.state_update["sp_current_stage"] == "planning"
    assert result.state_update["sp_last_action_id"] == "act-think"
    assert result.state_update["sp_last_idempotency_key"] == "idem-think"
    assert result.state_update["sp_loop_iteration"] == 1
    stack = TaskMemoryStack.from_dict(result.state_update["sp_task_memory"])
    assert stack.entries[-1].action == "think"
    assert stack.entries[-1].thread_id == "thread-1"
    assert [event["event_type"] for event in result.run_events[:2]] == ["sp.action.created", "sp.handler.started"]
    assert result.run_events[-1]["event_type"] == "sp.handler.completed"


def test_router_skips_duplicate_idempotency_key():
    router = build_default_action_router()
    state = {
        "sp_last_idempotency_key": "idem-1",
        "sp_last_handler_result": {"next_step": "finish"},
    }
    action = _action(ActionType.THINK, action_id="act-repeat", idempotency_key="idem-1", task="Repeat")

    result = router.execute(action, state=state)

    assert result.next_step == "finish"
    assert result.state_update == {}
    assert result.run_events[0]["event_type"] == "sp.action.duplicate_skipped"


def test_summarize_handler_condenses_sources_without_touching_pinned_feedback():
    stack = TaskMemoryStack()
    first = stack.append_think("old plan")
    feedback = stack.append_feedback("Pinned feedback")
    second = stack.append_observe("research", actor="researcher")
    state = {"sp_task_memory": stack.to_dict()}
    action = _action(
        ActionType.SUMMARIZE,
        action_id="act-summary",
        task="Summary of useful parts",
        metadata={"source_entry_ids": [first.id, feedback.id, second.id]},
    )

    result = build_default_action_router().execute(action, state=state)
    restored = TaskMemoryStack.from_dict(result.state_update["sp_task_memory"])

    by_id = {entry.id: entry for entry in restored.entries}
    assert by_id[first.id].status == "condensed"
    assert by_id[feedback.id].status == "pinned"
    assert by_id[second.id].status == "condensed"
    assert restored.entries[-1].action == "summarize"


def test_backtrack_marks_entries_after_target_and_preserves_artifact_history():
    stack = TaskMemoryStack()
    target = stack.append_think("safe checkpoint", stage="planning")
    doomed = stack.append_delegate("too broad delegation", stage="research")
    state = {
        "sp_task_memory": stack.to_dict(),
        "sp_active_delegate_id": "delegate-1",
        "sp_current_artifact_refs": {
            "report": {"artifact_id": "new", "type": "report", "version": 2},
            "_history": [
                {"artifact_id": "old", "type": "report", "version": 1},
                {"artifact_id": "new", "type": "report", "version": 2},
            ],
        },
    }
    action = _action(
        ActionType.BACKTRACK,
        action_id="act-backtrack",
        metadata={
            "backtrack_target_type": "entry",
            "backtrack_target_id": target.id,
            "rollback_scope": "full_working_state",
            "reason": "delegation went wide",
        },
        stage="planning",
    )

    result = build_default_action_router().execute(action, state=state)
    restored = TaskMemoryStack.from_dict(result.state_update["sp_task_memory"])
    by_id = {entry.id: entry for entry in restored.entries}

    assert by_id[doomed.id].status == "pruned"
    assert result.state_update["sp_active_delegate_id"] is None
    assert result.state_update["sp_current_artifact_refs"]["report"]["artifact_id"] == "new"
    assert restored.entries[-1].action == "backtrack"
    assert any(event["event_type"] == "sp.memory.backtracked" for event in result.run_events)


def test_finish_handler_rejects_pending_human_and_accepts_final_ref():
    router = build_default_action_router()
    action = _action(ActionType.FINISH, action_id="act-finish", task="Done")

    rejected = router.execute(action, state={"sp_pending_human_interaction": {"status": "pending"}})
    assert rejected.next_step == "error_recoverable"
    assert "pending" in rejected.error

    accepted = router.execute(
        _action(ActionType.FINISH, action_id="act-finish-2", task="Done"),
        state={"sp_current_artifact_refs": {"report": {"artifact_id": "report-1"}}},
    )
    assert accepted.next_step == "finish"
    assert accepted.state_update["sp_current_stage"] == "finished"
    assert TaskMemoryStack.from_dict(accepted.state_update["sp_task_memory"]).entries[-1].action == "finish"


def test_ask_human_interrupts_and_records_pending_interaction():
    state = {"sp_current_artifact_refs": {"outline": {"artifact_id": "outline-1", "type": "outline"}}}
    action = _action(
        ActionType.ASK_HUMAN,
        action_id="act-human",
        task="Please confirm the outline",
        metadata={"interaction_type": "outline_confirmation"},
        stage="planning",
    )

    result = build_default_action_router().execute(action, state=state, thread_id="thread-1", run_id="run-1")

    assert result.next_step == "interrupt"
    pending = result.state_update["sp_pending_human_interaction"]
    assert pending["status"] == "pending"
    assert pending["interaction_type"] == "outline_confirmation"
    assert pending["artifact_refs"]["outline"]["artifact_id"] == "outline-1"
    restored = TaskMemoryStack.from_dict(result.state_update["sp_task_memory"])
    assert restored.entries[-1].action == "ask_human"
    assert restored.entries[-1].content == "Please confirm the outline"
    assert any(event["event_type"] == "sp.human.requested" for event in result.run_events)


def test_record_human_feedback_clears_pending_and_pins_feedback():
    stack = TaskMemoryStack()
    state = {
        "sp_task_memory": stack.to_dict(),
        "sp_current_stage": "planning",
        "sp_pending_human_interaction": {
            "interaction_id": "hitl-1",
            "interaction_type": "outline_confirmation",
            "artifact_refs": {"outline": {"artifact_id": "outline-1", "type": "outline"}},
            "status": "pending",
        },
        "sp_current_artifact_refs": {
            "outline": {"artifact_id": "outline-1", "type": "outline", "feedback_entry_ids": []},
            "_history": [{"artifact_id": "outline-1", "type": "outline", "feedback_entry_ids": []}],
        },
    }

    result = record_human_feedback(state, "以后大纲先写结论再写证据", thread_id="thread-1", run_id="run-1")

    assert result.state_update["sp_pending_human_interaction"] is None
    restored = TaskMemoryStack.from_dict(result.state_update["sp_task_memory"])
    feedback = restored.entries[-1]
    assert feedback.action == "feedback"
    assert feedback.priority == "critical"
    assert feedback.status == "pinned"
    assert feedback.metadata["interaction_id"] == "hitl-1"
    assert result.state_update["sp_current_artifact_refs"]["outline"]["feedback_entry_ids"] == [feedback.id]
    assert result.state_update["sp_current_artifact_refs"]["_history"][0]["feedback_entry_ids"] == [feedback.id]


def test_recall_memory_requires_memory_recaller_executor():
    action = _action(
        ActionType.RECALL_MEMORY,
        action_id="act-recall-no-executor",
        metadata={"memory_query": "migration preferences"},
    )

    result = build_default_action_router().execute(action, state={})

    assert result.next_step == "error_recoverable"
    assert "memory_recaller subagent executor" in result.error
    assert any(event["event_type"] == "sp.handler.failed" for event in result.run_events)


def test_router_rejects_unregistered_subagent_actions_until_phase_3():
    action = _action(ActionType.DELEGATE, action_id="act-delegate", target_agent="researcher", task="Research")

    result = build_default_action_router().execute(action, state={})

    assert result.next_step == "error_recoverable"
    assert "No handler registered" in result.error


class FakeSubagentExecutor:
    def __init__(self, result: SPSubagentResult):
        self.result = result
        self.tasks: list[SPSubagentTask] = []

    def execute(self, task: SPSubagentTask) -> SPSubagentResult:
        self.tasks.append(task)
        return self.result


def _thread_state_with_outputs(tmp_path):
    return {
        "thread_data": {
            "outputs_path": str(tmp_path / "threads" / "thread-1" / "user-data" / "outputs"),
        }
    }


def test_delegate_handler_calls_executor_and_externalizes_large_result(tmp_path):
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result="Report artifact created",
            task_id="task-1",
            artifact_content="# Draft report\n\nLarge report body",
            artifact_type="report_revision",
        )
    )
    state = _thread_state_with_outputs(tmp_path)
    action = _action(
        ActionType.DELEGATE,
        action_id="act-delegate-ok",
        target_agent="reporter",
        task="Draft the report",
        input_refs=["artifact://outline"],
        expected_output="report artifact",
        stage="reporting",
    )

    result = build_default_action_router(delegate_executor=executor).execute(action, state=state, thread_id="thread-1", run_id="run-1")

    assert result.next_step == "continue"
    assert executor.tasks[0].subagent_type == "reporter"
    assert executor.tasks[0].input_refs == ["artifact://outline"]
    assert "task_memory" in executor.tasks[0].context_refs
    restored = TaskMemoryStack.from_dict(result.state_update["sp_task_memory"])
    assert [entry.action for entry in restored.entries[-2:]] == ["delegate", "observe"]
    assert restored.entries[-1].actor == "reporter"
    assert result.state_update["sp_active_delegate_id"] is None
    assert result.state_update["artifacts"][0].startswith("/mnt/user-data/outputs/sp/report_revision/")
    report_ref = result.state_update["sp_current_artifact_refs"]["report_revision"]
    assert report_ref["artifact_id"] == restored.entries[-1].result_ref
    assert "Large report body" not in str(report_ref)
    assert any(event["event_type"] == "sp.delegate.completed" for event in result.run_events)


def test_delegate_handler_records_failure_as_recoverable_error():
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.FAILED,
            error="research failed",
            task_id="task-failed",
        )
    )
    action = _action(ActionType.DELEGATE, action_id="act-delegate-fail", target_agent="researcher", task="Research")

    result = build_default_action_router(delegate_executor=executor).execute(action, state={})

    assert result.next_step == "error_recoverable"
    assert result.error == "research failed"
    restored = TaskMemoryStack.from_dict(result.state_update["sp_task_memory"])
    assert [entry.action for entry in restored.entries[-2:]] == ["delegate", "error"]
    assert restored.entries[-1].result_ref == "task-failed"
    assert any(event["event_type"] == "sp.delegate.failed" for event in result.run_events)


def test_recall_memory_handler_calls_memory_recaller_and_records_dry_run_result():
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.COMPLETED,
            result='{"summary":"Use phased commits and tests.","items":[{"content":"User wants a unit test after every migration unit.","source":"user_memory","score":0.94,"scope":"project","memory_id":"mem-1"}]}',
            task_id="mem-task-1",
        )
    )
    action = _action(
        ActionType.RECALL_MEMORY,
        action_id="act-recall-ok",
        metadata={"memory_query": "StackPlanner migration preferences"},
        stage="planning",
    )

    result = build_default_action_router(memory_recall_executor=executor).execute(action, state={}, thread_id="thread-1", run_id="run-1")

    assert result.next_step == "continue"
    assert executor.tasks[0].subagent_type == "memory_recaller"
    assert "Do not write, update, promote, or mutate long-term memory." in executor.tasks[0].task
    assert executor.tasks[0].metadata["dry_run_promotion"] is True
    restored = TaskMemoryStack.from_dict(result.state_update["sp_task_memory"])
    assert [entry.action for entry in restored.entries[-2:]] == ["recall_memory", "recall_memory"]
    recall_entry = restored.entries[-1]
    assert recall_entry.actor == "memory_recaller"
    assert recall_entry.result_ref == "mem-task-1"
    assert recall_entry.metadata["dry_run_promotion"] is True
    assert recall_entry.metadata["recall"]["dry_run_promotion"] is True
    assert recall_entry.metadata["recall"]["items"][0]["memory_id"] == "mem-1"
    assert "User wants a unit test" in recall_entry.content
    assert any(event["event_type"] == "sp.memory.recall.completed" for event in result.run_events)


def test_recall_memory_handler_records_failure_as_recoverable_error():
    executor = FakeSubagentExecutor(
        SPSubagentResult(
            status=SPSubagentStatus.FAILED,
            error="memory unavailable",
            task_id="mem-task-failed",
        )
    )
    action = _action(
        ActionType.RECALL_MEMORY,
        action_id="act-recall-fail",
        metadata={"memory_query": "StackPlanner migration preferences"},
    )

    result = build_default_action_router(memory_recall_executor=executor).execute(action, state={})

    assert result.next_step == "error_recoverable"
    assert result.error == "memory unavailable"
    restored = TaskMemoryStack.from_dict(result.state_update["sp_task_memory"])
    assert [entry.action for entry in restored.entries[-2:]] == ["recall_memory", "error"]
    assert restored.entries[-1].actor == "memory_recaller"
    assert restored.entries[-1].result_ref == "mem-task-failed"
    assert any(event["event_type"] == "sp.memory.recall.failed" for event in result.run_events)


def test_router_stops_when_loop_limit_is_exceeded():
    action = _action(ActionType.THINK, action_id="act-loop", task="keep thinking")

    result = build_default_action_router().execute(
        action,
        state={"sp_loop_iteration": 2, "sp_max_loop_iterations": 2},
    )

    assert result.next_step == "error_fatal"
    assert "exceeded max iterations" in result.error


def test_central_prompt_enforces_action_json_and_no_direct_tools():
    prompt = CENTRAL_AGENT_ACTION_PROMPT

    assert "Return JSON only" in prompt
    assert "do not call business tools directly" in prompt
    assert "search" in prompt
    assert "bash" in prompt
    assert "FINISH requires no pending human interaction" in prompt
    assert "handler routes to memory_recaller" in prompt
