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
    token_usage_records: list[dict] | None = None


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


def test_adapter_parses_structured_artifact_contract_and_usage():
    executor = FakeDR2Executor(
        FakeDR2Result(
            status="completed",
            result="""```json
{"summary":"Verified three migration risks.","artifact_content":"# Evidence\\n\\nFull research body","artifact_type":"research_observation","artifact_metadata":{"sources":3}}
```""",
            task_id="task-structured",
            token_usage_records=[{"input_tokens": 10, "output_tokens": 5}],
        )
    )

    result = DR2SubagentExecutorAdapter(lambda task: executor).execute(_sp_task())

    assert result.result == "Verified three migration risks."
    assert result.artifact_content == "# Evidence\n\nFull research body"
    assert result.artifact_type == "research_observation"
    assert result.artifact_metadata == {"sources": 3}
    assert result.token_usage_records == [{"input_tokens": 10, "output_tokens": 5}]


def test_adapter_externalizes_unstructured_large_result_as_defensive_fallback():
    large_result = "research evidence " * 200
    executor = FakeDR2Executor(FakeDR2Result(status="completed", result=large_result, task_id="task-large"))

    result = DR2SubagentExecutorAdapter(lambda task: executor).execute(_sp_task())

    assert result.artifact_content == large_result
    assert result.artifact_type == "research_observation"
    assert result.result.endswith("...<truncated>")
    assert len(result.result) == 700


def test_adapter_preserves_timeout_status_and_observes_raw_result():
    raw = FakeDR2Result(status="timed_out", error="deadline", task_id="task-timeout")
    executor = FakeDR2Executor(raw)
    observed = []

    result = DR2SubagentExecutorAdapter(lambda task: executor, result_observer=observed.append).execute(_sp_task())

    assert result.status == SPSubagentStatus.TIMED_OUT
    assert result.error == "deadline"
    assert observed == [raw]
