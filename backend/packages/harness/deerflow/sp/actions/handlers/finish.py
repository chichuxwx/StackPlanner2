"""FINISH action handler."""

from __future__ import annotations

from deerflow.sp.actions.events import make_sp_event
from deerflow.sp.actions.handlers.base import HandlerContext
from deerflow.sp.actions.schema import HandlerResult, SPAction


def _artifact_identifiers(ref: object) -> set[str]:
    if not isinstance(ref, dict):
        return set()
    return {
        str(value)
        for key in ("artifact_id", "artifact_url", "virtual_path")
        if (value := ref.get(key))
    }


def _selected_artifact_ref(context: HandlerContext, action: SPAction) -> str | None:
    explicit = action.metadata.get("final_artifact_ref")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    requested = {str(value) for value in action.input_refs}
    if not requested:
        return None
    refs = context.state.get("sp_current_artifact_refs")
    if not isinstance(refs, dict):
        return None
    for ref in refs.values():
        identifiers = _artifact_identifiers(ref)
        match = next((identifier for identifier in identifiers if identifier in requested), None)
        if match:
            return match
    return None


def _has_current_human_feedback(context: HandlerContext) -> bool:
    raw_memory = context.state.get("sp_task_memory")
    entries = raw_memory.get("entries") if isinstance(raw_memory, dict) else None
    return bool(
        context.run_id
        and isinstance(entries, list)
        and any(
            isinstance(entry, dict)
            and entry.get("action") == "feedback"
            and entry.get("run_id") == context.run_id
            for entry in entries
        )
    )


def _has_final_artifact(context: HandlerContext, action: SPAction) -> bool:
    if action.metadata.get("allow_without_artifact"):
        return True
    if action.metadata.get("final_artifact_ref"):
        return True
    refs = context.state.get("sp_current_artifact_refs") or {}
    if not isinstance(refs, dict):
        return False
    explicit = _selected_artifact_ref(context, action)
    run_id = context.run_id
    state_run_id = context.state.get("sp_loop_run_id")
    for key in ("report", "report_revision", "final_report"):
        ref = refs.get(key)
        if not isinstance(ref, dict):
            continue
        if not run_id:
            return True
        if not ref.get("run_id") and state_run_id == run_id:
            return True
        if not state_run_id or _has_current_human_feedback(context):
            return True
        if run_id and ref.get("run_id") == run_id:
            return True
        if explicit and explicit in _artifact_identifiers(ref):
            return True
    return False


class FinishHandler:
    def handle(self, action: SPAction, context: HandlerContext) -> HandlerResult:
        if context.state.get("sp_pending_human_interaction"):
            return HandlerResult(
                next_step="error_recoverable",
                idempotency_key=action.idempotency_key,
                error="FINISH is blocked while sp_pending_human_interaction exists",
                run_events=[
                    make_sp_event(
                        "sp.finish.rejected",
                        action_id=action.action_id,
                        run_id=context.run_id,
                        reason="pending_human_interaction",
                    )
                ],
            )
        if not _has_final_artifact(context, action):
            return HandlerResult(
                next_step="error_recoverable",
                idempotency_key=action.idempotency_key,
                error="FINISH requires a final artifact ref or allow_without_artifact=true",
                run_events=[
                    make_sp_event(
                        "sp.finish.rejected",
                        action_id=action.action_id,
                        run_id=context.run_id,
                        reason="missing_final_artifact",
                    )
                ],
            )

        entry = context.stack.append_finish(
            action.task or action.reason,
            thread_id=context.thread_id,
            run_id=context.run_id,
            priority=action.priority,
            metadata={"action_id": action.action_id, **action.metadata},
        )
        return HandlerResult(
            next_step="finish",
            state_update={
                "sp_current_stage": "finished",
                "sp_active_delegate_id": None,
                "sp_last_run_summary": action.task or action.reason,
                **(
                    {"sp_last_final_artifact_ref": _selected_artifact_ref(context, action)}
                    if _selected_artifact_ref(context, action)
                    else {}
                ),
            },
            memory_entries=[entry],
            idempotency_key=action.idempotency_key,
            run_events=[make_sp_event("sp.finish.accepted", action_id=action.action_id, run_id=context.run_id)],
        )
