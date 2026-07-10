"""Bounded prompt context for StackPlanner task memory.

The builder is intentionally pure: it reads already-restored SP task memory and
lightweight ThreadState refs, then renders a compact control context for the
CentralAgent. It never reads files, artifacts, tools, or long-term memory.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from deerflow.sp.memory import StackMemoryEntry, TaskMemoryStack

DEFAULT_CONTEXT_MAX_CHARS = 6000
DEFAULT_ENTRY_CONTENT_MAX_CHARS = 700
DEFAULT_RECENT_ENTRY_LIMIT = 12


def _compact(value: Any, *, max_chars: int) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True)
        except TypeError:
            text = repr(value)
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 15]}...<truncated>"


def _json_ref(value: Any, *, max_chars: int) -> str:
    if not value:
        return "{}"
    return _compact(value, max_chars=max_chars)


def _entry_priority_rank(entry: StackMemoryEntry) -> int:
    priority_rank = {"critical": 0, "high": 1, "normal": 2, "low": 3}.get(entry.priority, 2)
    return priority_rank


@dataclass(slots=True)
class PromptContextBuilder:
    """Render SP short-term memory into a bounded CentralAgent context block."""

    max_chars: int = DEFAULT_CONTEXT_MAX_CHARS
    max_entry_chars: int = DEFAULT_ENTRY_CONTENT_MAX_CHARS
    recent_entry_limit: int = DEFAULT_RECENT_ENTRY_LIMIT

    def build(
        self,
        stack: TaskMemoryStack,
        *,
        current_stage: str | None = None,
        active_delegate_id: str | None = None,
        pending_human_interaction: Mapping[str, Any] | None = None,
        artifact_refs: Mapping[str, Any] | None = None,
        report_version: str | None = None,
    ) -> str:
        """Build the bounded SP context used by CentralAgent decisions."""
        lines = [
            "<sp-task-context>",
            "CentralAgent control context. Use this as task memory, not as tool output.",
            f"current_stage: {current_stage or 'unknown'}",
            f"pending_human_interaction: {_json_ref(pending_human_interaction, max_chars=900)}",
            "",
            "priority_rules:",
            "- Critical or pinned human feedback outranks summaries, observations, and model plans.",
            "- Artifact refs point to Workspace/Artifact content; do not infer large artifact bodies from this block.",
        ]

        pinned = self._select_pinned(stack)
        if pinned:
            lines.extend(["", "critical_feedback:"])
            lines.extend(self._format_entries(pinned))

        lines.extend(
            [
                "",
                f"active_delegate_id: {active_delegate_id or 'none'}",
                f"current_artifact_refs: {_json_ref(artifact_refs, max_chars=1200)}",
                f"current_report_version: {report_version or 'none'}",
            ]
        )

        recent = self._select_recent(stack, exclude_ids={entry.id for entry in pinned})
        if recent:
            lines.extend(["", "recent_task_memory:"])
            lines.extend(self._format_entries(recent))

        lines.append("</sp-task-context>")
        return self._clip_lines(lines)

    def _select_pinned(self, stack: TaskMemoryStack) -> list[StackMemoryEntry]:
        # Keep the newest item inside each priority band. This matters when a
        # long task accumulates more pinned feedback than the prompt window can
        # render: the latest correction must not be displaced by an old one.
        pinned = sorted(stack.get_pinned_entries(), key=lambda entry: entry.ts, reverse=True)
        pinned.sort(key=_entry_priority_rank)
        return pinned[: self.recent_entry_limit]

    def _select_recent(self, stack: TaskMemoryStack, *, exclude_ids: set[str]) -> list[StackMemoryEntry]:
        active = [entry for entry in stack.get_active_entries() if entry.id not in exclude_ids and entry.status != "pinned" and entry.priority != "critical"]
        return active[-self.recent_entry_limit :]

    def _format_entries(self, entries: Iterable[StackMemoryEntry]) -> list[str]:
        return [self._format_entry(entry) for entry in entries]

    def _format_entry(self, entry: StackMemoryEntry) -> str:
        parts = [entry.action]
        if entry.actor:
            parts.append(f"actor={entry.actor}")
        if entry.stage:
            parts.append(f"stage={entry.stage}")
        if entry.priority:
            parts.append(f"priority={entry.priority}")
        if entry.result_ref:
            parts.append(f"result_ref={entry.result_ref}")
        if entry.failure_note:
            parts.append(f"failure={_compact(entry.failure_note, max_chars=180)}")
        content = _compact(entry.content, max_chars=self.max_entry_chars)
        return f"- id={entry.id} ({', '.join(parts)}): {content}"

    def _clip_lines(self, lines: list[str]) -> str:
        full_context = "\n".join(lines)
        if len(full_context) <= self.max_chars:
            return full_context

        closing = "</sp-task-context>"
        marker = "...<sp-task-context-truncated>"
        body_lines = lines[:-1] if lines and lines[-1] == closing else lines
        reserved = len(marker) + len(closing) + 2
        output: list[str] = []
        total = 0
        for line in body_lines:
            additional = len(line) + 1
            if total + additional + reserved > self.max_chars:
                break
            output.append(line)
            total += additional
        output.extend([marker, closing])
        return "\n".join(output)[: self.max_chars]
