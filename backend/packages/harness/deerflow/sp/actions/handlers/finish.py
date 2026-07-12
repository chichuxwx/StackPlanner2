"""FINISH action handler."""

from __future__ import annotations

from deerflow.sp.actions.events import make_sp_event
from deerflow.sp.actions.handlers.base import HandlerContext
from deerflow.sp.actions.schema import HandlerResult, SPAction


def _has_final_artifact(context: HandlerContext, action: SPAction) -> bool:
    if action.metadata.get("allow_without_artifact"):
        return True
    if action.metadata.get("final_artifact_ref"):
        return True
    refs = context.state.get("sp_current_artifact_refs") or {}
    return isinstance(refs, dict) and any(key in refs for key in ("report", "report_revision", "final_report"))


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
            },
            memory_entries=[entry],
            idempotency_key=action.idempotency_key,
            run_events=[make_sp_event("sp.finish.accepted", action_id=action.action_id, run_id=context.run_id)],
        )
