"""Adapters from SP subagent tasks to DR2 SubagentExecutor instances."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from deerflow.sp.subagents.adapter import SPSubagentResult, SPSubagentStatus, SPSubagentTask


class DR2SubagentExecutorLike(Protocol):
    def execute(self, task: str) -> Any:
        """Execute a rendered subagent task using DR2's SubagentExecutor API."""


DR2ExecutorFactory = Callable[[SPSubagentTask], DR2SubagentExecutorLike]


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return sorted(_json_safe(item) for item in value)
    return repr(value)


def render_sp_subagent_prompt(task: SPSubagentTask) -> str:
    """Render a structured SP task for a DR2 subagent prompt."""
    payload = {
        "action_id": task.action_id,
        "subagent_type": task.subagent_type,
        "description": task.description,
        "task": task.task,
        "input_refs": task.input_refs,
        "expected_output": task.expected_output,
        "context_refs": task.context_refs,
        "metadata": task.metadata,
    }
    return "\n".join(
        [
            "<sp-subagent-task>",
            json.dumps(_json_safe(payload), ensure_ascii=False, sort_keys=True, indent=2),
            "</sp-subagent-task>",
            "",
            "Complete the task using the tools available to this DR2 subagent. Return a concise result summary and mention any artifact paths or ids created.",
        ]
    )


def normalize_dr2_subagent_result(result: Any) -> SPSubagentResult:
    """Normalize DR2 SubagentResult-like objects into SPSubagentResult."""
    raw_status = getattr(result, "status", None)
    status_value = getattr(raw_status, "value", raw_status)
    try:
        status = SPSubagentStatus(str(status_value or "failed"))
    except ValueError:
        status = SPSubagentStatus.FAILED
    return SPSubagentResult(
        status=status,
        result=getattr(result, "result", None),
        error=getattr(result, "error", None),
        stop_reason=getattr(result, "stop_reason", None),
        task_id=getattr(result, "task_id", None),
    )


class DR2SubagentExecutorAdapter:
    """SPSubagentExecutorProtocol implementation backed by DR2 SubagentExecutor."""

    def __init__(self, executor_factory: DR2ExecutorFactory):
        self._executor_factory = executor_factory

    def execute(self, task: SPSubagentTask) -> SPSubagentResult:
        executor = self._executor_factory(task)
        result = executor.execute(render_sp_subagent_prompt(task))
        return normalize_dr2_subagent_result(result)
