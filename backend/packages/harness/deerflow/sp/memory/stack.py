"""Task memory stack for StackPlanner-on-DeerFlow."""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from deerflow.sp.memory.entry import StackMemoryEntry, utc_now_iso


SP_TASK_MEMORY_VERSION = 1


class TaskMemoryStack:
    """Explicit SP short-term memory stored as serializable thread state."""

    def __init__(self, entries: Iterable[StackMemoryEntry] | None = None, *, max_size: int = 50):
        self.max_size = max_size
        self.entries: list[StackMemoryEntry] = list(entries or [])
        self._maintain_size()

    def append(self, entry: StackMemoryEntry) -> StackMemoryEntry:
        self.entries.append(entry)
        self._maintain_size()
        return entry

    def append_think(self, content: str, **kwargs: Any) -> StackMemoryEntry:
        return self.append(StackMemoryEntry(action="think", content=content, **kwargs))

    def append_delegate(self, content: str, *, actor: str = "central", **kwargs: Any) -> StackMemoryEntry:
        return self.append(StackMemoryEntry(action="delegate", actor=actor, content=content, **kwargs))

    def append_observe(self, content: str, *, actor: str, **kwargs: Any) -> StackMemoryEntry:
        return self.append(StackMemoryEntry(action="observe", actor=actor, content=content, **kwargs))

    def append_memory_recall(self, content: str, **kwargs: Any) -> StackMemoryEntry:
        return self.append(StackMemoryEntry(action="recall_memory", content=content, **kwargs))

    def append_reflect(self, content: str, **kwargs: Any) -> StackMemoryEntry:
        return self.append(StackMemoryEntry(action="reflect", content=content, **kwargs))

    def append_backtrack(self, content: str, **kwargs: Any) -> StackMemoryEntry:
        return self.append(StackMemoryEntry(action="backtrack", content=content, **kwargs))

    def append_replan(self, content: str, **kwargs: Any) -> StackMemoryEntry:
        return self.append(StackMemoryEntry(action="replan", content=content, **kwargs))

    def append_summary(self, content: str, *, parent_ids: list[str] | None = None, **kwargs: Any) -> StackMemoryEntry:
        return self.append(StackMemoryEntry(action="summarize", content=content, parent_ids=parent_ids or [], **kwargs))

    def append_feedback(self, content: str, *, actor: str = "human", **kwargs: Any) -> StackMemoryEntry:
        kwargs.pop("priority", None)
        kwargs.pop("status", None)
        return self.append(
            StackMemoryEntry(
                action="feedback",
                actor=actor,
                content=content,
                priority="critical",
                status="pinned",
                **kwargs,
            )
        )

    def append_finish(self, content: str, **kwargs: Any) -> StackMemoryEntry:
        kwargs.pop("stage", None)
        return self.append(StackMemoryEntry(action="finish", content=content, stage="finished", **kwargs))

    def get_active_entries(self) -> list[StackMemoryEntry]:
        return [entry for entry in self.entries if entry.status in {"active", "pinned"}]

    def get_pinned_entries(self) -> list[StackMemoryEntry]:
        return [entry for entry in self.entries if entry.status == "pinned" or entry.priority == "critical"]

    def get_recent_observations(self, count: int = 5) -> list[StackMemoryEntry]:
        observations = [entry for entry in self.get_active_entries() if entry.action == "observe"]
        return observations[-count:]

    def get_active_delegations(self) -> list[StackMemoryEntry]:
        return [entry for entry in self.get_active_entries() if entry.action == "delegate"]

    def get_checkpoint(self, entry_id_or_stage: str) -> StackMemoryEntry | None:
        for entry in reversed(self.entries):
            if entry.id == entry_id_or_stage or entry.stage == entry_id_or_stage:
                return entry
        return None

    def condense(self, source_entry_ids: list[str], summary: str, **kwargs: Any) -> StackMemoryEntry:
        source_ids = set(source_entry_ids)
        for entry in self.entries:
            if entry.id in source_ids and entry.status != "pinned":
                entry.status = "condensed"
        return self.append_summary(summary, parent_ids=list(source_entry_ids), **kwargs)

    def mark_backtracked(self, source_entry_ids: list[str], reason: str, **kwargs: Any) -> StackMemoryEntry:
        source_ids = set(source_entry_ids)
        for entry in self.entries:
            if entry.id in source_ids and entry.status != "pinned":
                entry.status = "pruned"
                entry.failure_note = reason
        return self.append_backtrack(reason, parent_ids=list(source_entry_ids), failure_note=reason, **kwargs)

    def prune(self, *, max_entries: int | None = None, max_chars: int | None = None) -> None:
        max_entries = max_entries if max_entries is not None else self.max_size

        def active_size() -> int:
            return len([entry for entry in self.entries if entry.status in {"active", "pinned"}])

        def active_chars() -> int:
            return sum(len(entry.content) for entry in self.entries if entry.status in {"active", "pinned"})

        for entry in self.entries:
            if entry.status == "pinned":
                continue
            if active_size() <= max_entries and (max_chars is None or active_chars() <= max_chars):
                break
            entry.status = "pruned"
            entry.failure_note = entry.failure_note or "Pruned from active SP working memory"

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": SP_TASK_MEMORY_VERSION,
            "max_size": self.max_size,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, data: Any, *, thread_id: str | None = None, run_id: str | None = None, max_size: int = 50) -> "TaskMemoryStack":
        if not data:
            return cls(max_size=max_size)
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                return cls(max_size=max_size)
        if isinstance(data, list):
            return cls.from_legacy_memory_stack(data, thread_id=thread_id, run_id=run_id, max_size=max_size)
        if not isinstance(data, dict):
            return cls(max_size=max_size)

        entries_data = data.get("entries")
        if entries_data is None and "stack" in data:
            entries_data = data.get("stack")
        if entries_data is None and ("timestamp" in data or "agent_type" in data):
            entries_data = [data]
        if not isinstance(entries_data, list):
            return cls(max_size=int(data.get("max_size") or max_size))

        entries = []
        for raw_entry in entries_data:
            if isinstance(raw_entry, dict):
                entry = StackMemoryEntry.from_dict(raw_entry)
                if thread_id and entry.thread_id is None:
                    entry.thread_id = thread_id
                if run_id and entry.run_id is None:
                    entry.run_id = run_id
                entries.append(entry)
        return cls(entries, max_size=int(data.get("max_size") or max_size))

    @classmethod
    def from_legacy_memory_stack(
        cls,
        data: Any,
        *,
        thread_id: str | None = None,
        run_id: str | None = None,
        max_size: int = 50,
    ) -> "TaskMemoryStack":
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                return cls(max_size=max_size)
        if not isinstance(data, list):
            return cls(max_size=max_size)

        entries = [
            StackMemoryEntry.from_legacy_dict(entry, thread_id=thread_id, run_id=run_id)
            for entry in data
            if isinstance(entry, dict)
        ]
        return cls(entries, max_size=max_size)

    def size(self) -> int:
        return len(self.entries)

    def is_empty(self) -> bool:
        return not self.entries

    def _maintain_size(self) -> None:
        while len([entry for entry in self.entries if entry.status in {"active", "pinned"}]) > self.max_size:
            removable = next((entry for entry in self.entries if entry.status == "active"), None)
            if removable is None:
                break
            removable.status = "pruned"
            removable.failure_note = removable.failure_note or f"Pruned at {utc_now_iso()} because stack exceeded max_size"
