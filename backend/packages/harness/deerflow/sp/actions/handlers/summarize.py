"""SUMMARIZE action handler."""

from __future__ import annotations

from deerflow.sp.actions.handlers.base import HandlerContext
from deerflow.sp.actions.schema import HandlerResult, SPAction


class SummarizeHandler:
    def handle(self, action: SPAction, context: HandlerContext) -> HandlerResult:
        source_entry_ids = [str(entry_id) for entry_id in action.metadata.get("source_entry_ids", [])]
        summary = str(action.metadata.get("summary") or action.task or action.reason)
        if source_entry_ids:
            entry = context.stack.condense(
                source_entry_ids,
                summary,
                thread_id=context.thread_id,
                run_id=context.run_id,
                stage=action.stage,
                priority=action.priority,
                metadata={"action_id": action.action_id, **action.metadata},
            )
        else:
            entry = context.stack.append_summary(
                summary,
                thread_id=context.thread_id,
                run_id=context.run_id,
                stage=action.stage,
                priority=action.priority,
                metadata={"action_id": action.action_id, **action.metadata},
            )
        state_update = {"sp_last_run_summary": summary}
        if action.stage:
            state_update["sp_current_stage"] = action.stage
        return HandlerResult(next_step="continue", state_update=state_update, memory_entries=[entry], idempotency_key=action.idempotency_key)
