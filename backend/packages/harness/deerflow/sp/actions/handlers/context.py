"""Bounded context assembly shared by SP action handlers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from deerflow.sp.actions.handlers.base import HandlerContext
from deerflow.sp.memory import StackMemoryEntry
from deerflow.utils.messages import message_to_text

MAX_HANDLER_MEMORY_ENTRIES = 24
MAX_HANDLER_PINNED_ENTRIES = 12
MAX_HANDLER_USER_INPUT_CHARS = 2400
MAX_HANDLER_ARTIFACT_HISTORY = 12


def _entry_context(entry: StackMemoryEntry) -> dict[str, Any]:
    """Keep handler context useful without forwarding unbounded entry metadata."""
    return {
        "id": entry.id,
        "actor": entry.actor,
        "action": entry.action,
        "content": entry.content,
        "result_ref": entry.result_ref,
        "priority": entry.priority,
        "stage": entry.stage,
        "status": entry.status,
        "failure_note": entry.failure_note,
    }


def _bounded_task_memory(context: HandlerContext) -> dict[str, Any]:
    pinned = context.stack.get_pinned_entries()[-MAX_HANDLER_PINNED_ENTRIES:]
    pinned_ids = {entry.id for entry in pinned}
    recent = [entry for entry in context.stack.get_active_entries() if entry.id not in pinned_ids]
    recent = recent[-(MAX_HANDLER_MEMORY_ENTRIES - len(pinned)) :]
    selected = [*pinned, *recent]
    return {
        "version": 1,
        "entry_count": len(context.stack.entries),
        "selected_entry_count": len(selected),
        "entries": [_entry_context(entry) for entry in selected],
    }


def _bounded_artifact_refs(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    refs = {str(key): item for key, item in value.items() if key != "_history"}
    history = value.get("_history")
    if isinstance(history, list):
        refs["_history"] = history[-MAX_HANDLER_ARTIFACT_HISTORY:]
    return refs


def _visible_user_inputs(state: Mapping[str, Any]) -> list[str]:
    messages = state.get("messages")
    if not isinstance(messages, list):
        return []
    values: list[str] = []
    for message in messages:
        message_type = getattr(message, "type", None)
        if message_type is None and isinstance(message, Mapping):
            message_type = message.get("type") or message.get("role")
        if message_type not in {"human", "user"}:
            continue
        additional_kwargs = getattr(message, "additional_kwargs", None)
        if additional_kwargs is None and isinstance(message, Mapping):
            additional_kwargs = message.get("additional_kwargs")
        if isinstance(additional_kwargs, Mapping) and additional_kwargs.get("hide_from_ui") is True:
            continue
        text = " ".join(message_to_text(message).split())
        if text:
            values.append(text[:MAX_HANDLER_USER_INPUT_CHARS])
    return values


def build_handler_context_refs(context: HandlerContext) -> dict[str, Any]:
    """Build the precise, bounded context passed to DR2 subagents."""
    refs: dict[str, Any] = {"task_memory": _bounded_task_memory(context)}
    artifact_refs = _bounded_artifact_refs(context.state.get("sp_current_artifact_refs"))
    if artifact_refs is not None:
        refs["artifact_refs"] = artifact_refs
    pending_human = context.state.get("sp_pending_human_interaction")
    if isinstance(pending_human, Mapping):
        refs["pending_human_interaction"] = dict(pending_human)
    current_stage = context.state.get("sp_current_stage")
    if current_stage:
        refs["current_stage"] = str(current_stage)
    user_inputs = _visible_user_inputs(context.state)
    if user_inputs:
        refs["original_query"] = user_inputs[0]
        refs["latest_user_input"] = user_inputs[-1]
    return refs
