"""Middleware for SP short-term task memory.

This middleware restores StackPlanner task memory from DeerFlow ThreadState,
keeps the serialized stack normalized for checkpointing, and injects a bounded
request-only prompt context for the CentralAgent. It does not register with the
default Lead Agent chain; SP orchestration should opt into it explicitly.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from typing import Any, override

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelCallResult, ModelRequest, ModelResponse
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.config import get_config
from langgraph.runtime import Runtime

from deerflow.agents.thread_state import ThreadState
from deerflow.sp.memory import TaskMemoryStack
from deerflow.sp.prompt import PromptContextBuilder

SP_TASK_CONTEXT_MESSAGE_NAME = "sp_task_memory_context"
SP_TASK_CONTEXT_KWARG = "sp_task_memory_context"


class TaskMemoryMiddleware(AgentMiddleware[ThreadState]):
    """Restore and inject SP task memory without creating a new runtime store."""

    state_schema = ThreadState

    def __init__(
        self,
        *,
        context_builder: PromptContextBuilder | None = None,
        max_stack_entries: int = 50,
        max_active_entries: int = 24,
        max_active_chars: int = 12000,
        inject_context: bool = True,
    ) -> None:
        super().__init__()
        self._context_builder = context_builder or PromptContextBuilder()
        self._max_stack_entries = max_stack_entries
        self._max_active_entries = max_active_entries
        self._max_active_chars = max_active_chars
        self._inject_context = inject_context

    @override
    def before_agent(self, state: ThreadState, runtime: Runtime) -> dict[str, Any] | None:
        stack = self._restore_stack(state, runtime)
        if stack.is_empty() and not self._has_raw_stack(state):
            return None
        stack.prune(max_entries=self._max_active_entries, max_chars=self._max_active_chars)
        payload = stack.to_dict()
        if payload == state.get("sp_task_memory"):
            return None
        return {"sp_task_memory": payload}

    @override
    async def abefore_agent(self, state: ThreadState, runtime: Runtime) -> dict[str, Any] | None:
        return self.before_agent(state, runtime)

    @override
    def after_agent(self, state: ThreadState, runtime: Runtime) -> dict[str, Any] | None:
        stack = self._restore_stack(state, runtime)
        if stack.is_empty() and not self._has_raw_stack(state):
            return None
        stack.prune(max_entries=self._max_active_entries, max_chars=self._max_active_chars)
        payload = stack.to_dict()
        if payload == state.get("sp_task_memory"):
            return None
        return {"sp_task_memory": payload}

    @override
    async def aafter_agent(self, state: ThreadState, runtime: Runtime) -> dict[str, Any] | None:
        return self.after_agent(state, runtime)

    @override
    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelCallResult:
        return handler(self._augment_request(request))

    @override
    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelCallResult:
        return await handler(self._augment_request(request))

    def _augment_request(self, request: ModelRequest) -> ModelRequest:
        messages = getattr(request, "messages", None)
        if not isinstance(messages, list):
            return request
        if _is_fresh_user_turn(request):
            # Drop the abandoned tool-call transcript, but still inject the
            # compact SP task context retained for the new user turn.
            request = request.override(messages=_isolate_fresh_turn_messages(messages))
            messages = getattr(request, "messages", None)
            if not isinstance(messages, list):
                return request
        if not self._inject_context:
            return request
        if _has_task_context_message(messages):
            return request

        runtime = getattr(request, "runtime", None)
        state = _request_state(request)
        stack = self._restore_stack(state, runtime)
        if stack.is_empty() and not _has_context_refs(state):
            return request

        context = self._context_builder.build(
            stack,
            current_stage=state.get("sp_current_stage"),
            active_delegate_id=state.get("sp_active_delegate_id"),
            pending_human_interaction=_mapping_or_none(state.get("sp_pending_human_interaction")),
            artifact_refs=_mapping_or_none(state.get("sp_current_artifact_refs")),
            report_version=state.get("sp_current_report_version"),
            current_run_id=_run_id(runtime),
        )
        context_message = HumanMessage(
            content=context,
            name=SP_TASK_CONTEXT_MESSAGE_NAME,
            additional_kwargs={"hide_from_ui": True, SP_TASK_CONTEXT_KWARG: True},
        )
        return request.override(messages=_insert_before_last_human(messages, context_message))

    def _restore_stack(self, state: Mapping[str, Any] | None, runtime: Runtime | None) -> TaskMemoryStack:
        state = state or {}
        thread_id = _thread_id(runtime)
        run_id = _run_id(runtime)
        raw = self._raw_stack_payload(state)
        stack = TaskMemoryStack.from_dict(raw, thread_id=thread_id, run_id=run_id, max_size=self._max_stack_entries)
        for entry in stack.entries:
            if thread_id and entry.thread_id is None:
                entry.thread_id = thread_id
            if run_id and entry.run_id is None:
                entry.run_id = run_id
        return stack

    @staticmethod
    def _raw_stack_payload(state: Mapping[str, Any]) -> Any:
        if "sp_task_memory" in state:
            return state.get("sp_task_memory")
        return state.get("memory_stack")

    @staticmethod
    def _has_raw_stack(state: Mapping[str, Any] | None) -> bool:
        return bool(state is not None and ("sp_task_memory" in state or "memory_stack" in state))


def _request_state(request: ModelRequest) -> Mapping[str, Any]:
    state = getattr(request, "state", None)
    if isinstance(state, Mapping):
        return state
    runtime = getattr(request, "runtime", None)
    runtime_state = getattr(runtime, "state", None)
    if isinstance(runtime_state, Mapping):
        return runtime_state
    return {}


def _mapping_or_none(value: Any) -> Mapping[str, Any] | None:
    return value if isinstance(value, Mapping) else None


def _has_context_refs(state: Mapping[str, Any]) -> bool:
    return any(
        state.get(key)
        for key in (
            "sp_current_stage",
            "sp_active_delegate_id",
            "sp_pending_human_interaction",
            "sp_current_artifact_refs",
            "sp_current_report_version",
        )
    )


def _has_task_context_message(messages: list[BaseMessage]) -> bool:
    return any(getattr(message, "name", None) == SP_TASK_CONTEXT_MESSAGE_NAME for message in messages)


def _is_fresh_user_turn(request: ModelRequest) -> bool:
    runtime = getattr(request, "runtime", None)
    context = getattr(runtime, "context", None)
    return isinstance(context, Mapping) and bool(context.get("fresh_user_turn_after_terminal"))


def _isolate_fresh_turn_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Keep the current prompt while dropping an abandoned tool-call history."""
    latest_human = next(
        (
            message
            for message in reversed(messages)
            if isinstance(message, HumanMessage)
            and getattr(message, "name", None) != SP_TASK_CONTEXT_MESSAGE_NAME
        ),
        None,
    )
    if latest_human is None:
        return messages
    system_messages = [message for message in messages if isinstance(message, SystemMessage)]
    return [*system_messages, latest_human]


def _insert_before_last_human(messages: list[BaseMessage], context_message: HumanMessage) -> list[BaseMessage]:
    for idx in reversed(range(len(messages))):
        if isinstance(messages[idx], HumanMessage):
            return [*messages[:idx], context_message, *messages[idx:]]
    return [context_message, *messages]


def _thread_id(runtime: Runtime | None) -> str | None:
    context = getattr(runtime, "context", None) if runtime is not None else None
    if isinstance(context, Mapping) and context.get("thread_id"):
        return str(context["thread_id"])
    try:
        config = get_config()
    except RuntimeError:
        return None
    thread_id = config.get("configurable", {}).get("thread_id")
    return str(thread_id) if thread_id else None


def _run_id(runtime: Runtime | None) -> str | None:
    context = getattr(runtime, "context", None) if runtime is not None else None
    if isinstance(context, Mapping) and context.get("run_id"):
        return str(context["run_id"])
    try:
        config = get_config()
    except RuntimeError:
        return None
    run_id = config.get("metadata", {}).get("run_id") or config.get("configurable", {}).get("run_id")
    return str(run_id) if run_id else None
