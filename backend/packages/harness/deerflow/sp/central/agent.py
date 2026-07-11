"""SP CentralAgent decision adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage, SystemMessage

from deerflow.utils.messages import message_content_to_text, message_to_text

if TYPE_CHECKING:
    from deerflow.sp.loop import CentralDecisionRequest

DEFAULT_LATEST_USER_INPUT_MAX_CHARS = 2400


def _compact_text(value: str, *, max_chars: int) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 15]}...<truncated>"


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        candidates: list[dict[str, Any]] = []
        for start, character in enumerate(stripped):
            if character != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(stripped[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                candidates.append(candidate)
        if not candidates:
            raise ValueError("CentralAgent output did not contain a JSON object") from None
        action_candidates = [candidate for candidate in candidates if "action_type" in candidate]
        payload = (action_candidates or candidates)[-1]
    if not isinstance(payload, dict):
        raise ValueError("CentralAgent output JSON must be an object")
    return payload


def _latest_user_input(state: Any) -> str:
    if not isinstance(state, dict):
        return ""
    messages = state.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in reversed(messages):
        message_type = getattr(message, "type", None)
        if message_type == "human" or message.__class__.__name__ == "HumanMessage":
            return message_to_text(message)
    return ""


@dataclass(slots=True)
class CentralAgentDecider:
    """LLM-backed CentralAgent that only emits structured SP Action JSON."""

    model: Any

    def decide(self, request: CentralDecisionRequest) -> dict[str, Any]:
        latest_user_input = _compact_text(_latest_user_input(request.state), max_chars=DEFAULT_LATEST_USER_INPUT_MAX_CHARS)
        user_parts = [
            "<sp-decision-input>",
            request.task_context,
        ]
        if latest_user_input:
            user_parts.extend(["", "latest_user_input:", latest_user_input])
        user_parts.extend(["", "Return the next SPAction JSON object only.", "</sp-decision-input>"])
        response = self.model.invoke(
            [
                SystemMessage(content=request.system_prompt),
                HumanMessage(content="\n".join(user_parts)),
            ]
        )
        decision_text = message_content_to_text(response.content)
        if not decision_text.strip():
            additional_kwargs = getattr(response, "additional_kwargs", {})
            reasoning_content = additional_kwargs.get("reasoning_content") if isinstance(additional_kwargs, dict) else None
            if isinstance(reasoning_content, str):
                decision_text = reasoning_content
        return _extract_json_object(decision_text)
