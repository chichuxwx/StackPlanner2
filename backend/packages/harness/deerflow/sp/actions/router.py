"""Router for validated SP CentralAgent actions."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from deerflow.sp.actions.events import make_sp_event
from deerflow.sp.actions.handlers import (
    AskHumanHandler,
    BacktrackHandler,
    DelegateHandler,
    FinishHandler,
    HandlerContext,
    MemoryRecallHandler,
    ReflectHandler,
    ReplanHandler,
    ReviseHandler,
    SummarizeHandler,
    ThinkHandler,
)
from deerflow.sp.actions.handlers.base import BaseActionHandler
from deerflow.sp.actions.schema import ActionType, ActionValidationError, HandlerResult, SPAction
from deerflow.sp.artifacts import SPArtifactAdapter
from deerflow.sp.memory import StackMemoryEntry, TaskMemoryStack
from deerflow.sp.subagents import SPSubagentExecutorProtocol

DEFAULT_MAX_LOOP_ITERATIONS = 20
MAX_IDEMPOTENCY_LEDGER_ENTRIES = 100


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
            stack = TaskMemoryStack.from_dict(state.get("sp_task_memory"), thread_id=thread_id, run_id=run_id)
            duplicate.state_update = self._build_state_update(
                action,
                state,
                stack,
                duplicate,
                run_id=run_id,
            )
            return duplicate

        limit_result = self._loop_limit_result(action, state, run_id=run_id)
        if limit_result is not None:
            return limit_result

        delegation_policy = self._delegation_policy_result(action, state, run_id=run_id)
        if delegation_policy is not None:
            stack = TaskMemoryStack.from_dict(state.get("sp_task_memory"), run_id=run_id)
            for entry in delegation_policy.memory_entries:
                stack.append(entry)
            delegation_policy.state_update = self._build_state_update(
                action,
                state,
                stack,
                delegation_policy,
                run_id=run_id,
            )
            return delegation_policy

        forced_reflection = self._forced_recovery_reflection(action, state, run_id=run_id)
        if forced_reflection is not None:
            return forced_reflection

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

        state_update = self._build_state_update(action, state, stack, result, run_id=run_id)
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
        ledger = state.get("sp_idempotency_ledger")
        previous = ledger.get(action.idempotency_key) if isinstance(ledger, Mapping) else None
        if not isinstance(previous, Mapping) and state.get("sp_last_idempotency_key") == action.idempotency_key:
            previous = state.get("sp_last_handler_result")
        if not isinstance(previous, Mapping):
            return None
        next_step = str(previous.get("next_step") or "continue")
        previous_action_type = previous.get("action_type")
        if previous_action_type and previous_action_type != action.action_type.value:
            error = f"idempotency_key {action.idempotency_key!r} was already used by {previous_action_type}, not {action.action_type.value}"
            return HandlerResult(
                next_step="error_recoverable",
                idempotency_key=action.idempotency_key,
                error=error,
                run_events=[
                    make_sp_event(
                        "sp.action.idempotency_collision",
                        action_id=action.action_id,
                        run_id=run_id,
                        idempotency_key=action.idempotency_key,
                        previous_action_type=previous_action_type,
                        action_type=action.action_type.value,
                        error=error,
                    )
                ],
            )

        # Recoverable failures are deliberately retryable with the same stable
        # key. Successful side effects remain protected by duplicate skipping.
        if next_step == "error_recoverable":
            return None

        # Once feedback has resolved an ASK_HUMAN action, replaying its stable
        # key must continue rather than recreating an already-answered request.
        if action.action_type == ActionType.ASK_HUMAN and not state.get("sp_pending_human_interaction"):
            next_step = "continue"
        if next_step not in {"continue", "interrupt", "finish", "error_recoverable", "error_fatal"}:
            next_step = "continue"
        return HandlerResult(
            next_step=next_step,  # type: ignore[arg-type]
            idempotency_key=action.idempotency_key,
            run_events=[make_sp_event("sp.action.duplicate_skipped", action_id=action.action_id, run_id=run_id, idempotency_key=action.idempotency_key)],
        )

    def _delegation_policy_result(self, action: SPAction, state: Mapping[str, Any], *, run_id: str | None) -> HandlerResult | None:
        """Require a new control decision before repeating the same delegation target."""
        if action.action_type != ActionType.DELEGATE:
            return None
        previous = state.get("sp_last_handler_result")
        if not isinstance(previous, Mapping):
            return None
        if previous.get("next_step") != "continue" or previous.get("action_type") != ActionType.DELEGATE.value:
            return None
        if action.target_agent == "reporter":
            # DelegateHandler has a stricter report-version guard that returns
            # FINISH when no explicit revision intent exists.
            return None
        if previous.get("target_agent") != action.target_agent:
            return None

        content = (
            f"Blocked a consecutive delegation to {action.target_agent}. "
            "CentralAgent must inspect the previous result with THINK, REFLECT, REPLAN, or SUMMARIZE "
            "before delegating to the same specialist again."
        )
        entry = StackMemoryEntry(
            thread_id=None,
            run_id=run_id,
            actor="policy",
            action="delegate_skipped",
            content=content,
            stage=action.stage,
            priority="high",
            metadata={"action_id": action.action_id, "target_agent": action.target_agent},
        )
        return HandlerResult(
            next_step="continue",
            memory_entries=[entry],
            idempotency_key=action.idempotency_key,
            run_events=[
                make_sp_event(
                    "sp.delegate.policy_blocked",
                    action_id=action.action_id,
                    run_id=run_id,
                    target_agent=action.target_agent,
                    reason="same_target_requires_intermediate_control_action",
                )
            ],
        )

    def _loop_limit_result(self, action: SPAction, state: Mapping[str, Any], *, run_id: str | None) -> HandlerResult | None:
        loop_iteration = self._loop_iteration_for_run(state, run_id)
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

    def _forced_recovery_reflection(
        self,
        action: SPAction,
        state: Mapping[str, Any],
        *,
        run_id: str | None,
    ) -> HandlerResult | None:
        """Force one REFLECT before a new action follows a recoverable failure."""
        previous = state.get("sp_last_handler_result")
        if not isinstance(previous, Mapping) or previous.get("next_step") != "error_recoverable":
            return None
        if not previous.get("action_type"):
            return None
        if action.action_type in {ActionType.REFLECT, ActionType.ASK_HUMAN}:
            return None
        previous_key = previous.get("idempotency_key") or state.get("sp_last_idempotency_key")
        if action.idempotency_key == previous_key:
            # Retrying the same logical operation is explicitly allowed.
            return None

        reflect_handler = self._handlers.get(ActionType.REFLECT)
        if reflect_handler is None:
            return None
        failure = str(previous.get("error") or "The previous action failed and needs diagnosis.")
        reflect_action = SPAction.create(
            ActionType.REFLECT,
            action_id=f"sp-recovery-reflect-{action.action_id}",
            idempotency_key=f"sp-recovery-reflect-{previous_key or action.idempotency_key}",
            reason="Diagnose the previous recoverable action failure before continuing.",
            task=failure,
            stage=str(state.get("sp_current_stage") or action.stage or "verification"),
            priority="high",
            metadata={"failure_note": failure, "triggered_by_action_id": action.action_id},
        )
        result = self.execute(reflect_action, state=state, run_id=run_id)
        result.run_events.insert(
            0,
            make_sp_event(
                "sp.action.policy_reflection_forced",
                action_id=action.action_id,
                run_id=run_id,
                failed_action_id=previous.get("action_id"),
                requested_action_type=action.action_type.value,
            ),
        )
        return result

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
        result.state_update = self._build_state_update(
            action,
            state,
            TaskMemoryStack.from_dict(state.get("sp_task_memory")),
            result,
            run_id=run_id,
        )
        return result

    @staticmethod
    def _loop_iteration_for_run(state: Mapping[str, Any], run_id: str | None) -> int:
        state_run_id = state.get("sp_loop_run_id")
        if run_id and state_run_id and str(state_run_id) != run_id:
            return 0
        return int(state.get("sp_loop_iteration") or 0)

    def _build_state_update(
        self,
        action: SPAction,
        state: Mapping[str, Any],
        stack: TaskMemoryStack,
        result: HandlerResult,
        *,
        run_id: str | None,
    ) -> dict[str, Any]:
        loop_iteration = self._loop_iteration_for_run(state, run_id) + 1
        state_update: dict[str, Any] = {
            "sp_task_memory": stack.to_dict(),
            "sp_current_action_id": None,
            "sp_current_action": None,
            "sp_last_action_id": action.action_id,
            "sp_last_idempotency_key": action.idempotency_key,
            "sp_loop_iteration": loop_iteration,
            "sp_loop_run_id": run_id or state.get("sp_loop_run_id"),
            "sp_last_handler_result": {
                "next_step": result.next_step,
                "error": result.error,
                "action_id": action.action_id,
                "action_type": action.action_type.value,
                "idempotency_key": action.idempotency_key,
                "target_agent": action.target_agent,
            },
        }
        ledger = dict(state.get("sp_idempotency_ledger")) if isinstance(state.get("sp_idempotency_ledger"), Mapping) else {}
        ledger.pop(action.idempotency_key, None)
        ledger[action.idempotency_key] = dict(state_update["sp_last_handler_result"])
        state_update["sp_idempotency_ledger"] = dict(list(ledger.items())[-MAX_IDEMPOTENCY_LEDGER_ENTRIES:])
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
        ActionType.REVISE: ReviseHandler(),
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
