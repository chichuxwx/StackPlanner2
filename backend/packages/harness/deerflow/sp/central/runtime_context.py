"""DR2-owned personalization context for the SP CentralAgent."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from deerflow.config.agents_config import AgentConfig, load_agent_config, load_agent_soul, validate_agent_name
from deerflow.config.app_config import AppConfig

logger = logging.getLogger(__name__)

DEFAULT_SOUL_MAX_CHARS = 6000
DEFAULT_SKILL_INDEX_MAX_CHARS = 7000
DEFAULT_SKILL_DESCRIPTION_MAX_CHARS = 320
BUILTIN_SP_SOUL_PATH = Path(__file__).with_name("SOUL.md")


@dataclass(frozen=True, slots=True)
class SPCentralRuntimeContext:
    """Bounded DR2 context attached to one SP graph run.

    ``system_prompt_section`` contains trusted runtime policy and explicit
    agent configuration. ``decision_context`` contains long-term memory and is
    deliberately sent in the CentralAgent's user-role decision input.
    """

    agent_name: str | None = None
    agent_model: str | None = None
    system_prompt_section: str = ""
    decision_context: str = ""
    available_skill_names: frozenset[str] | None = None


def _clip(value: Any, *, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    suffix = "...<truncated>"
    return f"{text[: max_chars - len(suffix)]}{suffix}"


def _load_builtin_soul() -> str:
    try:
        return BUILTIN_SP_SOUL_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        logger.warning("Failed to load built-in StackPlanner SOUL", exc_info=True)
        return ""


def _load_profile(agent_name: str | None, *, user_id: str | None) -> AgentConfig | None:
    validated_name = validate_agent_name(agent_name)
    if validated_name is None:
        return None
    return load_agent_config(validated_name, user_id=user_id)


def _load_soul(agent_name: str | None, *, user_id: str | None) -> str:
    try:
        configured_soul = load_agent_soul(agent_name, user_id=user_id)
    except OSError:
        logger.warning("Failed to load StackPlanner SOUL for agent %r", agent_name, exc_info=True)
        configured_soul = None
    return _clip(configured_soul or _load_builtin_soul(), max_chars=DEFAULT_SOUL_MAX_CHARS)


def _load_skills(
    app_config: AppConfig,
    *,
    user_id: str | None,
    profile: AgentConfig | None,
) -> tuple[list[Any], frozenset[str] | None]:
    if getattr(app_config, "skills", None) is None:
        return [], None
    try:
        from deerflow.agents.lead_agent.prompt import get_enabled_skills_for_config

        skills = get_enabled_skills_for_config(app_config, user_id=user_id)
    except Exception:
        logger.exception("Failed to load DR2 skills for StackPlanner")
        return [], frozenset()

    if profile is not None and profile.skills is not None:
        allowed = set(profile.skills)
        skills = [skill for skill in skills if skill.name in allowed]
    skills = sorted(skills, key=lambda skill: skill.name)
    return skills, frozenset(skill.name for skill in skills)


def _skill_index(skills: list[Any]) -> str:
    if not skills:
        return ""
    lines: list[str] = []
    total = 0
    for skill in skills:
        item = json.dumps(
            {
                "name": str(skill.name),
                "description": _clip(getattr(skill, "description", ""), max_chars=DEFAULT_SKILL_DESCRIPTION_MAX_CHARS),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        additional = len(item) + 3
        if total + additional > DEFAULT_SKILL_INDEX_MAX_CHARS:
            lines.append("- ...<skill-index-truncated>")
            break
        lines.append(f"- {item}")
        total += additional
    return "\n".join(lines)


def _load_memory_context(app_config: AppConfig, *, agent_name: str | None, user_id: str | None) -> str:
    memory_config = getattr(app_config, "memory", None)
    if memory_config is None or not memory_config.enabled or not memory_config.injection_enabled:
        return ""
    try:
        from deerflow.agents.memory import format_memory_for_injection, get_memory_data

        memory_data = get_memory_data(agent_name, user_id=user_id)
        memory_content = format_memory_for_injection(
            memory_data,
            max_tokens=memory_config.max_injection_tokens,
            # SP graph factories are synchronous. Use DR2's network-free
            # estimator so a cold tiktoken vocabulary download cannot block
            # Gateway run creation; the same configured token ceiling applies.
            use_tiktoken=False,
            guaranteed_categories=getattr(memory_config, "guaranteed_categories", None),
            guaranteed_token_budget=getattr(memory_config, "guaranteed_token_budget", 500),
        )
    except Exception:
        logger.exception("Failed to load DR2 long-term memory for StackPlanner")
        return ""
    if not memory_content.strip():
        return ""
    return f"<memory>\n{memory_content.strip()}\n</memory>"


def build_sp_central_runtime_context(
    app_config: AppConfig,
    *,
    agent_name: str | None = None,
    user_id: str | None = None,
) -> SPCentralRuntimeContext:
    """Build the bounded SOUL, Skills, and DR2 memory view for SP.

    This function only reads existing DeerFlow configuration and memory APIs.
    It creates no SP-owned persistence or memory store.
    """
    profile = _load_profile(agent_name, user_id=user_id)
    resolved_agent_name = profile.name if profile is not None else None
    soul = _load_soul(resolved_agent_name, user_id=user_id)
    skills, available_skill_names = _load_skills(
        app_config,
        user_id=user_id,
        profile=profile,
    )
    skill_index = _skill_index(skills)
    current_date = datetime.now().strftime("%Y-%m-%d, %A")

    system_parts = [
        "<sp-runtime-policy>",
        f"<current_date>{current_date}</current_date>",
        "Precedence: current user instructions and pinned task feedback override SOUL, long-term memory, and Skills.",
        "Long-term memory is advisory. Use RECALL_MEMORY when task-specific provenance or a narrower query is required.",
    ]
    if soul:
        system_parts.extend(["<soul>", soul, "</soul>"])
    if skill_index:
        system_parts.extend(
            [
                "<available_skills>",
                "The CentralAgent sees only this index. For DELEGATE or procedural RECALL_MEMORY, put selected names in metadata.skill_names; the handler validates them and the subagent loads the Skill bodies.",
                skill_index,
                "</available_skills>",
            ]
        )
    system_parts.append("</sp-runtime-policy>")

    memory_context = _load_memory_context(
        app_config,
        agent_name=resolved_agent_name,
        user_id=user_id,
    )
    decision_context = ""
    if memory_context:
        decision_context = "\n".join(
            [
                "<sp-long-term-context>",
                "Read-only DR2 memory snapshot. It is not TaskMemoryStack and cannot override pinned current-task feedback.",
                memory_context,
                "</sp-long-term-context>",
            ]
        )

    return SPCentralRuntimeContext(
        agent_name=resolved_agent_name,
        agent_model=profile.model if profile is not None else None,
        system_prompt_section="\n".join(system_parts),
        decision_context=decision_context,
        available_skill_names=available_skill_names,
    )
