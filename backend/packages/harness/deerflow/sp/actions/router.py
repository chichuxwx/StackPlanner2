"""Router for validated SP CentralAgent actions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from deerflow.sp.actions.events import make_sp_event
from deerflow.sp.actions.handlers import AskHumanHandler, BacktrackHandler, DelegateHandler, FinishHandler, HandlerContext, MemoryRecallHandler, ReflectHandler, ReplanHandler, SummarizeHandler, ThinkHandler
from deerflow.sp.actions.handlers.base import BaseActionHandler
from deerflow.sp.actions.schema import ActionType, ActionValidationError, HandlerResult, SPAction
from deerflow.sp.artifacts import SPArtifactAdapter
from deerflow.sp.memory import TaskMemoryStack
from deerflow.sp.subagents import SPSubagentExecutorProtocol

DEFAULT_MAX_LOOP_ITERATIONS = 20


class ActionRouter:
    """Validate, route, and checkpoint SP actions without running business tools."""

    def __init__(self, handlers: Mapping[ActionType, BaseActionHandler] | None = None):
        self._handlers = dict(handlers or {})

    def execute(
        self,
        action_input: SPAction | dict[str, Any],
        *,
        state: Mapping[str, Any] | None = None,
        thread_id: str | None = None,
        run_id: str | None = None,
    ) -> HandlerResult:
        state = state or {}
        try:
            action = action_input if isinstance(action_input, SPAction) else SPAction.from_dict(action_input)
        except ActionValidationError as exc:
            return HandlerResult(
                next_step="error_recoverable",
                error=str(exc),
                run_events=[make_sp_event("sp.action.validation_failed", run_id=run_id, error=str(exc))],
            )

        duplicate = self._duplicate_result(action, state, run_id=run_id)
        if duplicate is not None:
            return duplicate

        limit_result = self._loop_limit_result(action, state, run_id=run_id)
        if limit_result is not None:
            return limit_result

        handler = self._handlers.get(action.action_type)
        if handler is None:
            return self._unsupported_result(action, state, run_id=run_id)

        stack = TaskMemoryStack.from_dict(state.get("sp_task_memory"), thread_id=thread_id, run_id=run_id)
        context = HandlerContext(state=state, stack=stack, thread_id=thread_id, run_id=run_id)
        events = [
            make_sp_event(
                "sp.action.created",
                action_id=action.action_id,
                run_id=run_id,
                action_type=action.action_type.value,
                idempotency_key=action.idempotency_key,
            ),
            make_sp_event("sp.handler.started", action_id=action.action_id, run_id=run_id, action_type=action.action_type.value),
        ]
        try:
            result = handler.handle(action, context)
        except Exception as exc:  # pragma: no cover - defensive runtime boundary
            result = HandlerResult(next_step="error_recoverable", idempotency_key=action.idempotency_key, error=str(exc))

        state_update = self._build_state_update(action, state, stack, result)
        result.state_update = {**result.state_update, **state_update}
        result.run_events = [
            *events,
            *result.run_events,
            make_sp_event(
                "sp.handler.failed" if result.next_step.startswith("error") else "sp.handler.completed",
                action_id=action.action_id,
                run_id=run_id,
                action_type=action.action_type.value,
                next_step=result.next_step,
                error=result.error,
            ),
        ]
        result.idempotency_key = action.idempotency_key
        return result

    def _duplicate_result(self, action: SPAction, state: Mapping[str, Any], *, run_id: str | None) -> HandlerResult | None:
        if state.get("sp_last_idempotency_key") != action.idempotency_key:
            return None
        last_result = state.get("sp_last_handler_result") if isinstance(state.get("sp_last_handler_result"), dict) else {}
        next_step = str(last_result.get("next_step") or "continue")
        if next_step not in {"continue", "interrupt", "finish", "error_recoverable", "error_fatal"}:
            next_step = "continue"
        return HandlerResult(
            next_step=next_step,  # type: ignore[arg-type]
            idempotency_key=action.idempotency_key,
            run_events=[make_sp_event("sp.action.duplicate_skipped", action_id=action.action_id, run_id=run_id, idempotency_key=action.idempotency_key)],
        )

    def _loop_limit_result(self, action: SPAction, state: Mapping[str, Any], *, run_id: str | None) -> HandlerResult | None:
        loop_iteration = int(state.get("sp_loop_iteration") or 0)
        max_loop_iterations = int(state.get("sp_max_loop_iterations") or DEFAULT_MAX_LOOP_ITERATIONS)
        if loop_iteration < max_loop_iterations:
            return None
        error = f"SP action loop exceeded max iterations: {loop_iteration}/{max_loop_iterations}"
        result_summary = {"next_step": "error_fatal", "error": error, "action_id": action.action_id}
        return HandlerResult(
            next_step="error_fatal",
            state_update={"sp_last_handler_result": result_summary},
            idempotency_key=action.idempotency_key,
            error=error,
            run_events=[make_sp_event("sp.handler.failed", action_id=action.action_id, run_id=run_id, error=error)],
        )

    def _unsupported_result(self, action: SPAction, state: Mapping[str, Any], *, run_id: str | None) -> HandlerResult:
        error = f"No handler registered for {action.action_type.value}"
        result = HandlerResult(
            next_step="error_recoverable",
            idempotency_key=action.idempotency_key,
            error=error,
            run_events=[
                make_sp_event("sp.action.created", action_id=action.action_id, run_id=run_id, action_type=action.action_type.value),
                make_sp_event("sp.handler.failed", action_id=action.action_id, run_id=run_id, error=error),
            ],
        )
        result.state_update = self._build_state_update(action, state, TaskMemoryStack.from_dict(state.get("sp_task_memory")), result)
        return result

    def _build_state_update(self, action: SPAction, state: Mapping[str, Any], stack: TaskMemoryStack, result: HandlerResult) -> dict[str, Any]:
        loop_iteration = int(state.get("sp_loop_iteration") or 0) + 1
        state_update: dict[str, Any] = {
            "sp_task_memory": stack.to_dict(),
            "sp_current_action_id": None,
            "sp_last_action_id": action.action_id,
            "sp_last_idempotency_key": action.idempotency_key,
            "sp_loop_iteration": loop_iteration,
            "sp_last_handler_result": {
                "next_step": result.next_step,
                "error": result.error,
                "action_id": action.action_id,
                "action_type": action.action_type.value,
                "idempotency_key": action.idempotency_key,
            },
        }
        if result.artifact_refs:
            existing_refs = state.get("sp_current_artifact_refs") if isinstance(state.get("sp_current_artifact_refs"), dict) else {}
            state_update["sp_current_artifact_refs"] = {**existing_refs, **result.artifact_refs}
        return state_update


def build_default_action_router(
    *,
    delegate_executor: SPSubagentExecutorProtocol | None = None,
    memory_recall_executor: SPSubagentExecutorProtocol | None = None,
    artifact_adapter: SPArtifactAdapter | None = None,
) -> ActionRouter:
    handlers: dict[ActionType, BaseActionHandler] = {
        ActionType.THINK: ThinkHandler(),
        ActionType.REFLECT: ReflectHandler(),
        ActionType.BACKTRACK: BacktrackHandler(),
        ActionType.REPLAN: ReplanHandler(),
        ActionType.SUMMARIZE: SummarizeHandler(),
        ActionType.ASK_HUMAN: AskHumanHandler(),
        ActionType.RECALL_MEMORY: MemoryRecallHandler(executor=memory_recall_executor),
        ActionType.FINISH: FinishHandler(),
    }
    if delegate_executor is not None:
        handlers[ActionType.DELEGATE] = DelegateHandler(executor=delegate_executor, artifact_adapter=artifact_adapter)
    return ActionRouter(handlers)
