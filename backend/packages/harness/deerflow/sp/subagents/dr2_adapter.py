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
DR2ResultObserver = Callable[[Any], None]

DEFAULT_RESULT_SUMMARY_MAX_CHARS = 700
DEFAULT_LARGE_RESULT_THRESHOLD = 1200


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
            "Complete the task using the tools available to this DR2 subagent.",
            "Honor the subagent output contract: return compact JSON with summary, artifact_content, artifact_type, and artifact_metadata.",
            "Full reports, outlines, research bodies, and other large text belong in artifact_content, never in summary.",
        ]
    )


def _compact_result(value: Any, *, max_chars: int = DEFAULT_RESULT_SUMMARY_MAX_CHARS) -> str | None:
    if value is None:
        return None
    text = " ".join(str(value).split())
    if len(text) <= max_chars:
        return text
    suffix = "...<truncated>"
    return f"{text[: max_chars - len(suffix)]}{suffix}"


def _parse_result_payload(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def normalize_dr2_subagent_result(result: Any, *, task: SPSubagentTask | None = None) -> SPSubagentResult:
    """Normalize DR2 SubagentResult-like objects into SPSubagentResult."""
    raw_status = getattr(result, "status", None)
    status_value = getattr(raw_status, "value", raw_status)
    try:
        status = SPSubagentStatus(str(status_value or "failed"))
    except ValueError:
        status = SPSubagentStatus.FAILED
    raw_result = getattr(result, "result", None)
    payload = _parse_result_payload(raw_result)
    is_memory_recaller = task is not None and task.subagent_type == "memory_recaller"
    artifact_content = getattr(result, "artifact_content", None)
    artifact_type = getattr(result, "artifact_type", None)
    artifact_metadata = getattr(result, "artifact_metadata", None)
    summary = raw_result
    if not is_memory_recaller and payload is not None and any(key in payload for key in ("summary", "artifact_content", "artifact_type", "artifact_metadata")):
        summary = payload.get("summary") or payload.get("result")
        artifact_content = artifact_content if artifact_content is not None else payload.get("artifact_content")
        artifact_type = artifact_type or payload.get("artifact_type")
        payload_metadata = payload.get("artifact_metadata")
        if not isinstance(artifact_metadata, dict):
            artifact_metadata = {}
        if isinstance(payload_metadata, dict):
            artifact_metadata = {**payload_metadata, **artifact_metadata}

    if artifact_content is None and isinstance(raw_result, str) and len(raw_result) > DEFAULT_LARGE_RESULT_THRESHOLD and status == SPSubagentStatus.COMPLETED and not is_memory_recaller:
        artifact_content = raw_result
        artifact_type = artifact_type or _default_artifact_type(task.subagent_type if task is not None else None)

    if not isinstance(artifact_metadata, dict):
        artifact_metadata = {}

    normalized_result = _compact_result(summary)
    if is_memory_recaller:
        normalized_result = json.dumps(payload, ensure_ascii=False, separators=(",", ":")) if payload is not None else raw_result

    return SPSubagentResult(
        status=status,
        result=normalized_result,
        error=getattr(result, "error", None),
        stop_reason=getattr(result, "stop_reason", None),
        task_id=getattr(result, "task_id", None),
        artifact_content=artifact_content,
        artifact_type=str(artifact_type) if artifact_type else None,
        artifact_metadata=artifact_metadata,
        token_usage_records=list(getattr(result, "token_usage_records", None) or []),
    )


def _default_artifact_type(subagent_type: str | None) -> str:
    return {
        "researcher": "research_observation",
        "reporter": "report_revision",
        "outline": "outline",
        "coder": "generated_file",
        "perception": "perception_observation",
    }.get(str(subagent_type or ""), "generated_file")


class DR2SubagentExecutorAdapter:
    """SPSubagentExecutorProtocol implementation backed by DR2 SubagentExecutor."""

    def __init__(self, executor_factory: DR2ExecutorFactory, *, result_observer: DR2ResultObserver | None = None):
        self._executor_factory = executor_factory
        self._result_observer = result_observer

    def execute(self, task: SPSubagentTask) -> SPSubagentResult:
        executor = self._executor_factory(task)
        raw_result = executor.execute(render_sp_subagent_prompt(task))
        if self._result_observer is not None:
            self._result_observer(raw_result)
        return normalize_dr2_subagent_result(raw_result, task=task)
