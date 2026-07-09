"""Tests for the SP -> DR2 SubagentExecutor adapter."""

from dataclasses import dataclass

from deerflow.sp.subagents import DR2SubagentExecutorAdapter, SPSubagentStatus, SPSubagentTask, render_sp_subagent_prompt


@dataclass
class FakeDR2Result:
    status: str
    result: str | None = None
    error: str | None = None
    stop_reason: str | None = None
    task_id: str | None = None


class FakeDR2Executor:
    def __init__(self, result: FakeDR2Result):
        self.result = result
        self.prompts: list[str] = []

    def execute(self, task: str) -> FakeDR2Result:
        self.prompts.append(task)
        return self.result


def _sp_task() -> SPSubagentTask:
    return SPSubagentTask(
        action_id="act-1",
        subagent_type="researcher",
        task="Research migration risks",
        description="Need evidence",
        input_refs=["artifact://outline"],
        expected_output="short risk summary",
        context_refs={"task_memory": {"entries": [{"action": "think", "content": "plan"}]}},
        thread_id="thread-1",
        run_id="run-1",
        metadata={"stage": "research"},
    )


def test_render_sp_subagent_prompt_carries_structured_context():
    prompt = render_sp_subagent_prompt(_sp_task())

    assert "<sp-subagent-task>" in prompt
    assert '"action_id": "act-1"' in prompt
    assert '"subagent_type": "researcher"' in prompt
    assert '"artifact://outline"' in prompt
    assert '"task_memory"' in prompt


def test_dr2_subagent_executor_adapter_normalizes_success_result():
    executor = FakeDR2Executor(FakeDR2Result(status="completed", result="done", task_id="task-1"))
    adapter = DR2SubagentExecutorAdapter(lambda task: executor)

    result = adapter.execute(_sp_task())

    assert result.status == SPSubagentStatus.COMPLETED
    assert result.result == "done"
    assert result.task_id == "task-1"
    assert len(executor.prompts) == 1
    assert "Research migration risks" in executor.prompts[0]


def test_dr2_subagent_executor_adapter_maps_unknown_status_to_failed():
    executor = FakeDR2Executor(FakeDR2Result(status="unexpected", error="bad status", task_id="task-bad"))
    adapter = DR2SubagentExecutorAdapter(lambda task: executor)

    result = adapter.execute(_sp_task())

    assert result.status == SPSubagentStatus.FAILED
    assert result.error == "bad status"
    assert result.task_id == "task-bad"
