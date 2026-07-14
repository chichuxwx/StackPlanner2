"""SP control tools layered onto DeerFlow's native agent loop.

The tools are the SP-specific vocabulary. Normal DeerFlow tools remain normal
tools and continue in the same agent loop; they are not converted into SP
actions or sent through a second CentralAgent.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, Literal, Protocol

from langchain.agents import AgentState
from langchain.agents.middleware import AgentMiddleware
from langchain.tools import tool
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import END
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from deerflow.sp.actions import ActionType, SPAction, build_default_action_router
from deerflow.sp.memory import StackMemoryEntry, TaskMemoryStack
from deerflow.sp.subagents import SPSubagentExecutorProtocol

SP_CONTROL_TOOL_NAMES = frozenset(
    {
        "sp_think",
        "sp_delegate",
        "sp_recall_memory",
        "sp_reflect",
        "sp_revise",
        "sp_backtrack",
        "sp_replan",
        "sp_summarize",
        "sp_ask_human",
        "sp_finish",
    }
)


@tool("sp_think", parse_docstring=True)
def sp_think(task: str, reason: str | None = None, stage: str | None = None) -> str:
    """Record an explicit StackPlanner thought or checkpoint before continuing.

    Args:
        task: The concise thought or checkpoint to record.
        reason: Why this checkpoint is useful.
        stage: Optional task stage.
    """
    return "StackPlanner THINK action recorded. Continue with the task."


@tool("sp_delegate", parse_docstring=True)
def sp_delegate(
    target_agent: Literal["researcher", "coder", "reporter", "outline", "perception"],
    task: str,
    reason: str | None = None,
    stage: str | None = None,
    input_refs: list[str] | None = None,
    expected_output: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    """Delegate a separable task to a StackPlanner specialist.

    Args:
        target_agent: Specialist role to invoke.
        task: Concrete delegated task.
        reason: Why delegation is needed.
        stage: Current task stage.
        input_refs: Artifact or memory references for the specialist.
        expected_output: Expected result shape.
        metadata: Optional structured delegation metadata.
    """
    return "StackPlanner DELEGATE action recorded. Inspect the specialist result before continuing."


@tool("sp_recall_memory", parse_docstring=True)
def sp_recall_memory(query: str, reason: str | None = None, stage: str | None = None) -> str:
    """Recall relevant long-term memory through the StackPlanner memory handler.

    Args:
        query: Historical memory query; inspect short-term task context first.
        reason: Why recall is needed.
        stage: Current task stage.
    """
    return "StackPlanner RECALL_MEMORY action recorded. Use the returned memory context."


@tool("sp_reflect", parse_docstring=True)
def sp_reflect(task: str, reason: str | None = None, stage: str | None = None) -> str:
    """Diagnose progress, evidence, or a failed action.

    Args:
        task: What should be inspected or diagnosed.
        reason: Why reflection is needed.
        stage: Current task stage.
    """
    return "StackPlanner REFLECT action recorded. Continue after incorporating the diagnosis."


@tool("sp_revise", parse_docstring=True)
def sp_revise(
    target_entry_ids: list[str],
    correction: str,
    reason: str,
    stage: str | None = None,
) -> str:
    """Correct erroneous central task-memory entries and continue.

    Args:
        target_entry_ids: IDs of active SP memory entries proven to be wrong.
        correction: The corrected fact, decision, or plan to retain.
        reason: Evidence explaining why the target entries are wrong.
        stage: Current task stage.
    """
    return "StackPlanner REVISE action recorded. The invalid memory was superseded; continue from the correction."


@tool("sp_backtrack", parse_docstring=True)
def sp_backtrack(
    target_type: Literal["entry", "stage", "artifact_version", "delegation"],
    target_id: str,
    reason: str,
    rollback_scope: Literal["memory_only", "artifact_refs", "delegation", "stage", "full_working_state"] = "memory_only",
) -> str:
    """Backtrack StackPlanner working state without deleting artifact history.

    Args:
        target_type: Type of checkpoint to backtrack to.
        target_id: Entry, stage, artifact version, or delegation identifier.
        reason: Why the rollback is required.
        rollback_scope: Portion of working state to restore.
    """
    return "StackPlanner BACKTRACK action recorded. Replan from the restored state."


@tool("sp_replan", parse_docstring=True)
def sp_replan(task: str, reason: str | None = None, stage: str | None = None) -> str:
    """Create a revised StackPlanner plan after new evidence or failure.

    Args:
        task: Revised plan or next plan objective.
        reason: Why replanning is needed.
        stage: Current task stage.
    """
    return "StackPlanner REPLAN action recorded. Continue with the revised plan."


@tool("sp_summarize", parse_docstring=True)
def sp_summarize(
    summary: str,
    source_entry_ids: list[str] | None = None,
    reason: str | None = None,
    stage: str | None = None,
) -> str:
    """Condense repetitive task memory at a StackPlanner stage boundary.

    Args:
        summary: Bounded summary to retain.
        source_entry_ids: Active task-memory entry IDs to pop into this summary.
        reason: Why summarization is useful.
        stage: Current task stage.
    """
    return "StackPlanner SUMMARIZE action recorded. Continue with the condensed context."


@tool("sp_ask_human", parse_docstring=True)
def sp_ask_human(
    question: str,
    reason: str | None = None,
    interaction_type: str = "clarification",
    options: list[str] | None = None,
) -> str:
    """Interrupt the run and request human feedback.

    Args:
        question: Question to present to the user.
        reason: Why human input is required.
        interaction_type: Type of interaction.
        options: Optional choices.
    """
    return "StackPlanner ASK_HUMAN action recorded. Wait for the user's response."


@tool("sp_finish", parse_docstring=True)
def sp_finish(summary: str, allow_without_artifact: bool = False) -> str:
    """Mark the StackPlanner task ready to finish.

    Args:
        summary: Concise user-facing completion summary.
        allow_without_artifact: Set true only when no artifact is expected.
    """
    return "StackPlanner FINISH action recorded. Provide the final response."


def build_sp_control_tools() -> list[Any]:
    """Return only SP control tools; ordinary DeerFlow tools stay unchanged."""
    return [
        sp_think,
        sp_delegate,
        sp_recall_memory,
        sp_reflect,
        sp_revise,
        sp_backtrack,
        sp_replan,
        sp_summarize,
        sp_ask_human,
        sp_finish,
    ]


class SPExecutorProvider(Protocol):
    def __call__(self, state: Mapping[str, Any], runtime: Any) -> SPSubagentExecutorProtocol | None:
        """Build an executor for the current DR2 agent turn."""


def _run_id(runtime: Any) -> str | None:
    context = getattr(runtime, "context", None)
    if isinstance(context, Mapping) and context.get("run_id"):
        return str(context["run_id"])
    return None


def _thread_id(runtime: Any) -> str | None:
    context = getattr(runtime, "context", None)
    if isinstance(context, Mapping) and context.get("thread_id"):
        return str(context["thread_id"])
    return None


def _action_payload(tool_name: str, args: Mapping[str, Any], *, tool_call_id: str) -> dict[str, Any]:
    mapping = {
        "sp_think": ActionType.THINK,
        "sp_delegate": ActionType.DELEGATE,
        "sp_recall_memory": ActionType.RECALL_MEMORY,
        "sp_reflect": ActionType.REFLECT,
        "sp_revise": ActionType.REVISE,
        "sp_backtrack": ActionType.BACKTRACK,
        "sp_replan": ActionType.REPLAN,
        "sp_summarize": ActionType.SUMMARIZE,
        "sp_ask_human": ActionType.ASK_HUMAN,
        "sp_finish": ActionType.FINISH,
    }
    action_type = mapping[tool_name]
    payload = dict(args)
    payload["action_id"] = str(payload.get("action_id") or f"spact_{tool_call_id}")
    payload["action_type"] = action_type.value
    payload["idempotency_key"] = str(payload.get("idempotency_key") or f"spidem_{tool_call_id}")
    payload["reason"] = str(payload.get("reason") or f"CentralAgent selected {action_type.value}")
    if tool_name == "sp_recall_memory":
        payload["metadata"] = {**dict(payload.get("metadata") or {}), "memory_query": payload.pop("query")}
    elif tool_name == "sp_revise":
        payload["task"] = payload.pop("correction")
        payload["metadata"] = {
            **dict(payload.get("metadata") or {}),
            "target_entry_ids": payload.pop("target_entry_ids"),
            "revision_reason": payload.get("reason"),
        }
    elif tool_name == "sp_backtrack":
        payload["metadata"] = {
            **dict(payload.get("metadata") or {}),
            "backtrack_target_type": payload.pop("target_type"),
            "backtrack_target_id": payload.pop("target_id"),
            "rollback_scope": payload.pop("rollback_scope", "memory_only"),
        }
    elif tool_name == "sp_ask_human":
        payload["metadata"] = {
            **dict(payload.get("metadata") or {}),
            "question": payload.pop("question"),
            "interaction_type": payload.pop("interaction_type", "clarification"),
            "options": payload.pop("options", None),
        }
    elif tool_name == "sp_finish":
        payload["task"] = payload.pop("summary")
        payload["metadata"] = {"allow_without_artifact": bool(payload.pop("allow_without_artifact", False))}
    elif tool_name == "sp_summarize":
        payload["task"] = payload.pop("summary")
        source_entry_ids = payload.pop("source_entry_ids", None)
        if source_entry_ids is not None:
            payload["metadata"] = {
                **dict(payload.get("metadata") or {}),
                "source_entry_ids": source_entry_ids,
            }
    return payload


class SPControlActionMiddleware(AgentMiddleware[AgentState]):
    """Execute SP control methods in-place inside the one DR2 agent loop."""

    state_schema = AgentState

    def __init__(self, *, executor_provider: SPExecutorProvider | None = None) -> None:
        super().__init__()
        self._executor_provider = executor_provider

    def wrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Any]) -> Any:
        tool_name = str(request.tool_call.get("name") or "")
        if tool_name not in SP_CONTROL_TOOL_NAMES:
            return handler(request)

        return self._execute_control_tool(request)

    async def awrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[Any]]) -> Any:
        tool_name = str(request.tool_call.get("name") or "")
        if tool_name not in SP_CONTROL_TOOL_NAMES:
            return await handler(request)

        return self._execute_control_tool(request)

    def _execute_control_tool(self, request: ToolCallRequest) -> Command[Any]:
        tool_name = str(request.tool_call.get("name") or "")

        payload = _action_payload(tool_name, request.tool_call.get("args") or {}, tool_call_id=str(request.tool_call.get("id") or "sp-tool"))
        runtime = request.runtime
        executor = self._executor_provider(request.state, runtime) if self._executor_provider is not None else None
        router = build_default_action_router(delegate_executor=executor, memory_recall_executor=executor)
        result = router.execute(payload, state=request.state, thread_id=_thread_id(runtime), run_id=_run_id(runtime))
        action = SPAction.from_dict(payload)
        self._record_events(runtime, result.run_events)
        content = json.dumps(
            {
                "action_type": action.action_type.value,
                "next_step": result.next_step,
                "error": result.error,
            },
            ensure_ascii=False,
        )
        tool_message = ToolMessage(
            content=content,
            tool_call_id=str(request.tool_call.get("id") or "sp-tool"),
            name=tool_name,
            additional_kwargs={
                "stackplanner": {
                    "action_type": action.action_type.value,
                    "action_id": action.action_id,
                    "action_label": f"SP {action.action_type.value}",
                }
            },
        )
        update = {**result.state_update, "messages": [tool_message]}
        if result.next_step == "interrupt":
            pending = update.get("sp_pending_human_interaction") or {}
            tool_message.artifact = {
                "human_input": {
                    "version": 1,
                    "kind": "human_input_request",
                    "source": "stackplanner",
                    "request_id": str(pending.get("interaction_id") or action.action_id),
                    "clarification_type": str(pending.get("interaction_type") or "clarification"),
                    "question": str(pending.get("question") or action.metadata.get("question") or action.task or "Please provide feedback."),
                    "input_mode": "free_text",
                }
            }
            return Command(update=update, goto=END)
        return Command(update=update)

    @staticmethod
    def _record_events(runtime: Any, events: list[Mapping[str, Any]]) -> None:
        context = getattr(runtime, "context", None)
        journal = context.get("__run_journal") if isinstance(context, Mapping) else None
        record = getattr(journal, "record_custom_event", None)
        if not callable(record):
            return
        for event in events:
            record(
                str(event.get("event_type") or "sp.event"),
                content={"action_id": event.get("action_id"), "payload": event.get("payload") or {}},
                metadata={"source": "stackplanner", "action_id": event.get("action_id"), "run_id": event.get("run_id")},
            )


class SPThinkLabelMiddleware(AgentMiddleware[AgentState]):
    """Mark ordinary turns as THINK and retain useful tool observations."""

    state_schema = AgentState

    def after_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        return self._after_model_update(state, runtime)

    async def aafter_model(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        return self._after_model_update(state, runtime)

    def _after_model_update(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        messages = state.get("messages") if isinstance(state, Mapping) else None
        latest = messages[-1] if isinstance(messages, list) and messages else None
        if not isinstance(latest, AIMessage):
            return None
        if any(str(call.get("name") or "") in SP_CONTROL_TOOL_NAMES for call in latest.tool_calls):
            return None
        update: dict[str, Any] = {
            "sp_last_handler_result": {
                "action_type": ActionType.THINK.value,
                "action_id": str(latest.id or "think"),
                "next_step": "continue",
            }
        }
        memory_update = self._collect_tool_observations(state, runtime)
        if memory_update is not None:
            update["sp_task_memory"] = memory_update
        return update

    def wrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Any]) -> Any:
        # Tool calls may be executed concurrently. Returning a state update here
        # would make each tool write its own value to the single-value
        # ``sp_task_memory``/``sp_last_handler_result`` channels, which LangGraph
        # rejects when one model turn emits multiple tool calls. The next
        # ``after_model`` callback sees all ToolMessages and commits one merged
        # memory update for the whole turn.
        return handler(request)

    async def awrap_tool_call(self, request: ToolCallRequest, handler: Callable[[ToolCallRequest], Awaitable[Any]]) -> Any:
        # The web gateway uses LangGraph's async stream path. Keep native tools
        # untouched there as well; observations are aggregated in aafter_model.
        return await handler(request)

    def _collect_tool_observations(self, state: AgentState, runtime: Any) -> dict[str, Any] | None:
        messages = state.get("messages") if isinstance(state, Mapping) else None
        if not isinstance(messages, list):
            return None

        stack = TaskMemoryStack.from_dict(
            state.get("sp_task_memory"),
            thread_id=_thread_id(runtime),
            run_id=_run_id(runtime),
        )
        observed_tool_call_ids = {
            str(entry.metadata.get("tool_call_id"))
            for entry in stack.entries
            if entry.action == "observe" and entry.metadata.get("tool_call_id")
        }
        observed_keys = {
            str(entry.metadata.get("observation_key"))
            for entry in stack.entries
            if entry.metadata.get("observation_key")
        }
        changed = False
        for message in messages:
            if not isinstance(message, ToolMessage):
                continue
            tool_name = str(message.name or "")
            if tool_name in SP_CONTROL_TOOL_NAMES or tool_name == "ask_clarification":
                continue
            tool_call_id = str(message.tool_call_id or message.id or "")
            if not tool_call_id or tool_call_id in observed_tool_call_ids:
                continue
            summary = _tool_result_summary(message)
            if not summary or not _should_record_tool_observation(tool_name, summary):
                continue
            observation_key = _observation_key(tool_name, summary)
            if observation_key in observed_keys:
                continue
            is_error = _is_tool_error(summary)
            stack.append(
                StackMemoryEntry(
                    action="error" if is_error else "observe",
                    content=summary,
                    actor="deerflow",
                    thread_id=_thread_id(runtime),
                    run_id=_run_id(runtime),
                    priority="high" if is_error else "normal",
                    metadata={
                        "action_type": ActionType.THINK.value,
                        "tool_name": tool_name,
                        "tool_call_id": tool_call_id,
                        "observation_key": observation_key,
                        "result_chars": len(summary),
                    },
                )
            )
            observed_tool_call_ids.add(tool_call_id)
            observed_keys.add(observation_key)
            changed = True
        return stack.to_dict() if changed else None


def _tool_result_summary(result: Any, *, max_chars: int = 700) -> str:
    message = result if isinstance(result, ToolMessage) else None
    if message is None:
        update = getattr(result, "update", None)
        messages = update.get("messages") if isinstance(update, Mapping) else None
        if isinstance(messages, list):
            message = next((item for item in reversed(messages) if isinstance(item, ToolMessage)), None)
    content = getattr(message, "content", "") if message is not None else ""
    if isinstance(content, list):
        content = " ".join(str(item) for item in content)
    text = " ".join(str(content or "").split())
    if not text:
        return ""
    if len(text) <= max_chars:
        return text
    suffix = "...<truncated>"
    return f"{text[: max_chars - len(suffix)]}{suffix}"


# These are execution details, not CentralAgent decisions. The model still
# receives the native tool message in the current turn; we simply do not make
# it durable task memory after the turn has completed.
_NON_MEMORY_TOOL_NAMES = frozenset(
    {
        "ls",
        "glob",
        "list_dir",
        "list_files",
        "read_file",
        "present_file",
        "present_files",
        "describe_skill",
    }
)
_NON_MEMORY_SUCCESS_MESSAGES = frozenset(
    {
        "ok",
        "success",
        "successfully presented files",
        "file presented",
    }
)
_URL_ONLY_RE = re.compile(r"^(?:https?://\S+\s*)+$", re.IGNORECASE)
_FILE_LISTING_RE = re.compile(r"(?:^|\s)[-dl][rwx-]{9}\s+\d+\s+\S+\s+\S+\s+\d+\s+", re.IGNORECASE)
_ERROR_MARKERS = (
    "error:",
    "traceback",
    "connection refused",
    "unsafe absolute path",
    "http 4",
    "http 5",
    "aborterror",
    "failed:",
)


def _is_tool_error(summary: str) -> bool:
    lowered = summary.strip().lower()
    return lowered.startswith(_ERROR_MARKERS) or any(marker in lowered for marker in _ERROR_MARKERS)


def _should_record_tool_observation(tool_name: str, summary: str) -> bool:
    normalized = " ".join(summary.split()).strip().lower()
    if tool_name.lower() in _NON_MEMORY_TOOL_NAMES:
        return False
    if normalized in _NON_MEMORY_SUCCESS_MESSAGES or _URL_ONLY_RE.fullmatch(normalized):
        return False
    if _FILE_LISTING_RE.search(normalized):
        return False
    if (
        normalized.startswith("--- name:")
        or "this skill should be used when" in normalized
        or ("/mnt/skills/" in normalized and "skill" in normalized)
    ):
        return False
    return True


def _observation_key(tool_name: str, summary: str) -> str:
    normalized = " ".join(summary.split()).lower()
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    return f"{tool_name.lower()}:{digest}"
