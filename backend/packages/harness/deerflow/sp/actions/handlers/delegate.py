"""DELEGATE action handler."""

from __future__ import annotations

from typing import Any

from deerflow.sp.actions.events import make_sp_event
from deerflow.sp.actions.handlers.base import HandlerContext
from deerflow.sp.actions.handlers.context import build_handler_context_refs
from deerflow.sp.actions.schema import HandlerResult, SPAction
from deerflow.sp.artifacts import SPArtifactAdapter
from deerflow.sp.memory import StackMemoryEntry
from deerflow.sp.subagents import SPSubagentExecutorProtocol, SPSubagentResult, SPSubagentTask

OBSERVE_SUMMARY_MAX_CHARS = 700
LARGE_RESULT_ARTIFACT_THRESHOLD = 1200


def _compact_result(value: str | None, *, fallback: str) -> str:
    text = " ".join((value or fallback).split())
    if len(text) <= OBSERVE_SUMMARY_MAX_CHARS:
        return text
    suffix = "...<truncated>"
    return f"{text[: OBSERVE_SUMMARY_MAX_CHARS - len(suffix)]}{suffix}"


class DelegateHandler:
    """Delegate business work to a subagent executor, never directly to tools."""

    def __init__(
        self,
        *,
        executor: SPSubagentExecutorProtocol | None = None,
        artifact_adapter: SPArtifactAdapter | None = None,
    ) -> None:
        self._executor = executor
        self._artifact_adapter = artifact_adapter or SPArtifactAdapter()

    def handle(self, action: SPAction, context: HandlerContext) -> HandlerResult:
        if self._executor is None:
            return HandlerResult(
                next_step="error_recoverable",
                idempotency_key=action.idempotency_key,
                error="DELEGATE requires a SP subagent executor",
            )

        delegate_entry = context.stack.append_delegate(
            f"Delegate to {action.target_agent}: {action.task}",
            thread_id=context.thread_id,
            run_id=context.run_id,
            stage=action.stage,
            priority=action.priority,
            metadata={"action_id": action.action_id, "target_agent": action.target_agent, **action.metadata},
        )
        task = SPSubagentTask(
            action_id=action.action_id,
            subagent_type=str(action.target_agent),
            task=str(action.task),
            description=action.reason,
            input_refs=list(action.input_refs),
            expected_output=action.expected_output,
            context_refs=build_handler_context_refs(context),
            thread_id=context.thread_id,
            run_id=context.run_id,
            metadata=action.metadata,
        )

        result = self._executor.execute(task)
        if result.is_success:
            return self._handle_success(action, context, delegate_entry, result)
        return self._handle_failure(action, context, delegate_entry, result)

    def _handle_success(
        self,
        action: SPAction,
        context: HandlerContext,
        delegate_entry: StackMemoryEntry,
        result: SPSubagentResult,
    ) -> HandlerResult:
        state_update: dict[str, Any] = {"sp_active_delegate_id": None}
        artifact_refs: dict[str, Any] = {}
        artifact_events: list[dict[str, Any]] = []
        result_ref = result.task_id
        artifact_content = result.artifact_content
        if artifact_content is None and isinstance(result.result, str) and len(result.result) > LARGE_RESULT_ARTIFACT_THRESHOLD:
            artifact_content = result.result
        if artifact_content is not None:
            artifact_type = result.artifact_type or _default_artifact_type(str(action.target_agent))
            artifact = self._artifact_adapter.write_text_artifact(
                artifact_content,
                artifact_type=artifact_type,
                state=context.state,
                thread_id=context.thread_id,
                run_id=context.run_id,
                created_by=str(action.target_agent),
                stage=action.stage,
                source_entry_id=delegate_entry.id,
                summary=_compact_result(result.result, fallback="Subagent artifact created"),
                metadata={"delegate_action_id": action.action_id, **result.artifact_metadata},
            )
            state_update.update(artifact.state_update)
            artifact_refs = artifact.state_update.get("sp_current_artifact_refs", {})
            result_ref = artifact.metadata.artifact_id
            artifact_events.append(
                make_sp_event(
                    "sp.artifact.created",
                    action_id=action.action_id,
                    run_id=context.run_id,
                    artifact_id=artifact.metadata.artifact_id,
                    artifact_type=artifact.metadata.type,
                    virtual_path=artifact.metadata.virtual_path,
                )
            )
        else:
            created_paths = result.artifact_metadata.get("created_paths")
            if isinstance(created_paths, list):
                artifact_state = dict(context.state)
                registered_paths: list[str] = []
                for created_path in created_paths:
                    try:
                        artifact = self._artifact_adapter.register_existing_artifact(
                            str(created_path),
                            artifact_type=result.artifact_type or _default_artifact_type(str(action.target_agent)),
                            state={**artifact_state, **state_update},
                            thread_id=context.thread_id,
                            run_id=context.run_id,
                            created_by=str(action.target_agent),
                            stage=action.stage,
                            source_entry_id=delegate_entry.id,
                            summary=_compact_result(result.result, fallback="Subagent artifact created"),
                            metadata={"delegate_action_id": action.action_id},
                        )
                    except ValueError:
                        continue
                    state_update = _merge_artifact_state_updates(state_update, artifact.state_update)
                    artifact_refs = state_update.get("sp_current_artifact_refs", {})
                    result_ref = artifact.metadata.artifact_id
                    registered_paths.append(artifact.metadata.virtual_path)
                    artifact_events.append(
                        make_sp_event(
                            "sp.artifact.registered",
                            action_id=action.action_id,
                            run_id=context.run_id,
                            artifact_id=artifact.metadata.artifact_id,
                            artifact_type=artifact.metadata.type,
                            virtual_path=artifact.metadata.virtual_path,
                        )
                    )
                if registered_paths:
                    result.artifact_metadata["created_paths"] = registered_paths

        observe_content = _compact_result(result.result, fallback="Subagent completed without textual result")
        observe_entry = context.stack.append_observe(
            observe_content,
            actor=str(action.target_agent),
            thread_id=context.thread_id,
            run_id=context.run_id,
            stage=action.stage,
            priority=action.priority,
            result_ref=result_ref,
            metadata={
                "action_id": action.action_id,
                "task_id": result.task_id,
                "stop_reason": result.stop_reason,
                "target_agent": action.target_agent,
            },
        )
        return HandlerResult(
            next_step="continue",
            state_update=state_update,
            memory_entries=[delegate_entry, observe_entry],
            artifact_refs=artifact_refs,
            idempotency_key=action.idempotency_key,
            run_events=[
                make_sp_event("sp.delegate.started", action_id=action.action_id, run_id=context.run_id, target_agent=action.target_agent),
                make_sp_event("sp.delegate.completed", action_id=action.action_id, run_id=context.run_id, target_agent=action.target_agent, task_id=result.task_id),
                *artifact_events,
            ],
        )

    def _handle_failure(
        self,
        action: SPAction,
        context: HandlerContext,
        delegate_entry: StackMemoryEntry,
        result: SPSubagentResult,
    ) -> HandlerResult:
        error = result.error or f"Subagent {action.target_agent} failed with status {result.status.value}"
        error_entry = context.stack.append(
            StackMemoryEntry(
                thread_id=context.thread_id,
                run_id=context.run_id,
                actor=str(action.target_agent),
                action="error",
                content=error,
                result_ref=result.task_id,
                priority="high",
                stage=action.stage,
                failure_note=error,
                metadata={"action_id": action.action_id, "status": result.status.value},
            )
        )
        return HandlerResult(
            next_step="error_recoverable",
            state_update={"sp_active_delegate_id": None},
            memory_entries=[delegate_entry, error_entry],
            idempotency_key=action.idempotency_key,
            error=error,
            run_events=[
                make_sp_event("sp.delegate.started", action_id=action.action_id, run_id=context.run_id, target_agent=action.target_agent),
                make_sp_event("sp.delegate.failed", action_id=action.action_id, run_id=context.run_id, target_agent=action.target_agent, task_id=result.task_id, error=error),
            ],
        )


def _default_artifact_type(target_agent: str) -> str:
    return {
        "outline": "outline",
        "researcher": "research_observation",
        "reporter": "report_revision",
        "coder": "generated_file",
        "perception": "generated_file",
    }.get(target_agent, "generated_file")


def _merge_artifact_state_updates(existing: dict[str, Any], new: dict[str, Any]) -> dict[str, Any]:
    merged = {**existing, **new}
    if "artifacts" in existing or "artifacts" in new:
        merged["artifacts"] = list(dict.fromkeys([*existing.get("artifacts", []), *new.get("artifacts", [])]))
    return merged
