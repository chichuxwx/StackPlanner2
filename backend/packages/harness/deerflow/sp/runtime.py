"""DR2 runtime factory for the StackPlanner orchestration graph."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from deerflow.config.app_config import AppConfig, get_app_config
from deerflow.sp.agent_tools import SPControlActionMiddleware, SPThinkLabelMiddleware, build_sp_control_tools
from deerflow.sp.central import CENTRAL_AGENT_ACTION_PROMPT
from deerflow.sp.central.runtime_context import SPCentralRuntimeContext, build_sp_central_runtime_context
from deerflow.sp.subagents import DR2SubagentExecutorAdapter, SPSubagentExecutorProtocol, SPSubagentResult, SPSubagentTask

logger = logging.getLogger(__name__)

STACKPLANNER_ASSISTANT_ID = "stackplanner"
SP_SUBAGENT_REGISTRY_NAMES = {
    "researcher": "sp-researcher",
    "coder": "sp-coder",
    "reporter": "sp-reporter",
    "outline": "sp-outline",
    "perception": "sp-perception",
    "memory_recaller": "sp-memory-recaller",
}

FRESH_USER_TURN_PROMPT = """
<fresh_user_turn>
This is a new user turn after the previous run ended abnormally. Treat the current
user message as authoritative and do not resume the previous unfinished report,
delegation, search, or tool call unless the user explicitly asks to continue it.
For greetings and simple questions, answer directly without web search, delegation,
or repeated internal thinking.
</fresh_user_turn>
""".strip()


def _runtime_config(config: RunnableConfig) -> dict[str, Any]:
    merged = dict(config.get("configurable", {}) or {})
    context = config.get("context", {}) or {}
    if isinstance(context, Mapping):
        merged.update(context)
    return merged


def _resolve_model_name(config: RunnableConfig, app_config: AppConfig, *, agent_model: str | None = None) -> str:
    if not app_config.models:
        raise ValueError("StackPlanner requires at least one configured chat model.")
    runtime = _runtime_config(config)
    requested = runtime.get("model_name") or runtime.get("model") or agent_model
    if requested and app_config.get_model_config(str(requested)) is not None:
        return str(requested)
    if requested:
        logger.warning("StackPlanner model %r is not configured; using %s", requested, app_config.models[0].name)
    return app_config.models[0].name


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _apply_task_skill_policy(
    subagent_config: Any,
    task: SPSubagentTask,
    *,
    available_skill_names: frozenset[str] | None,
):
    """Apply the CentralAgent's requested Skill whitelist to one delegate."""
    raw_skill_names = task.metadata.get("skill_names")
    if raw_skill_names is None:
        configured = subagent_config.skills
        if configured is None or available_skill_names is None:
            return subagent_config
        filtered = [name for name in configured if name in available_skill_names]
        return replace(subagent_config, skills=filtered) if filtered != configured else subagent_config

    if not isinstance(raw_skill_names, list) or any(not isinstance(name, str) or not name.strip() for name in raw_skill_names):
        raise ValueError("SP action metadata.skill_names must be a list of non-empty Skill names")
    requested = list(dict.fromkeys(name.strip() for name in raw_skill_names))
    if available_skill_names is None:
        raise ValueError("SP Skill catalog is unavailable; cannot validate metadata.skill_names")
    unavailable = sorted(set(requested) - set(available_skill_names))
    if unavailable:
        raise ValueError(f"SP action requested unavailable or disabled Skills: {unavailable}")
    return replace(subagent_config, skills=requested)


@dataclass(slots=True)
class DR2SPExecutorProvider:
    """Create DR2 SubagentExecutor instances bound to the active graph state."""

    app_config: AppConfig
    parent_model: str
    runnable_config: RunnableConfig
    available_skill_names: frozenset[str] | None = None
    memory_agent_name: str | None = None
    user_id: str | None = None
    _tools: list[Any] | None = field(default=None, init=False, repr=False)

    def _available_tools(self) -> list[Any]:
        if self._tools is None:
            from deerflow.tools import get_available_tools

            self._tools = get_available_tools(
                model_name=self.parent_model,
                subagent_enabled=False,
                app_config=self.app_config,
            )
        return self._tools

    def __call__(self, state: Mapping[str, Any], runtime: Runtime) -> SPSubagentExecutorProtocol:
        context = _mapping(getattr(runtime, "context", None))
        metadata = _mapping(self.runnable_config.get("metadata"))
        writer = getattr(runtime, "stream_writer", None)

        def executor_factory(task: SPSubagentTask):
            from deerflow.subagents import SubagentExecutor, get_subagent_config

            registry_name = SP_SUBAGENT_REGISTRY_NAMES.get(task.subagent_type)
            subagent_config = get_subagent_config(registry_name, app_config=self.app_config) if registry_name else None
            if subagent_config is None or not subagent_config.internal:
                raise ValueError(f"Unknown StackPlanner subagent type: {task.subagent_type}")
            subagent_config = _apply_task_skill_policy(
                subagent_config,
                task,
                available_skill_names=self.available_skill_names,
            )
            thread_id = str(context.get("thread_id") or task.thread_id or "") or None
            run_id = str(context.get("run_id") or task.run_id or "") or None
            return SubagentExecutor(
                config=subagent_config,
                tools=self._available_tools(),
                app_config=self.app_config,
                parent_model=self.parent_model,
                sandbox_state=state.get("sandbox"),
                thread_data=state.get("thread_data"),
                thread_id=thread_id,
                trace_id=str(metadata.get("trace_id") or uuid.uuid4().hex[:8]),
                user_id=str(context.get("user_id") or self.user_id) if context.get("user_id") or self.user_id else None,
                user_role=str(context.get("user_role")) if context.get("user_role") else None,
                oauth_provider=str(context.get("oauth_provider")) if context.get("oauth_provider") else None,
                oauth_id=str(context.get("oauth_id")) if context.get("oauth_id") else None,
                run_id=run_id,
                channel_user_id=str(context.get("channel_user_id")) if context.get("channel_user_id") else None,
                deerflow_trace_id=str(context.get("deerflow_trace_id")) if context.get("deerflow_trace_id") else None,
                memory_agent_name=self.memory_agent_name if task.subagent_type == "memory_recaller" else None,
                user_scoped_skills=True,
            )

        def observe_raw_result(raw_result: Any) -> None:
            journal = context.get("__run_journal")
            report_usage = getattr(journal, "record_external_llm_usage_records", None)
            usage_records = list(getattr(raw_result, "token_usage_records", None) or [])
            if callable(report_usage) and usage_records:
                report_usage(usage_records)

        adapter = DR2SubagentExecutorAdapter(
            executor_factory,
            result_observer=observe_raw_result,
        )
        return _ProgressReportingExecutor(adapter=adapter, writer=writer)


@dataclass(slots=True)
class _ProgressReportingExecutor:
    adapter: DR2SubagentExecutorAdapter
    writer: Any = None

    def execute(self, task: SPSubagentTask) -> SPSubagentResult:
        if callable(self.writer):
            self.writer(
                {
                    "type": "task_started",
                    "task_id": task.action_id,
                    "description": task.description,
                    "subagent_type": task.subagent_type,
                }
            )
        try:
            result = self.adapter.execute(task)
        except Exception as exc:
            if callable(self.writer):
                self.writer(
                    {
                        "type": "task_failed",
                        "task_id": task.action_id,
                        "error": str(exc),
                        "subagent_type": task.subagent_type,
                    }
                )
            raise
        if callable(self.writer):
            event_type = "task_completed" if result.is_success else "task_failed"
            self.writer(
                {
                    "type": event_type,
                    "task_id": result.task_id or task.action_id,
                    "result": result.result,
                    "error": result.error,
                    "stop_reason": result.stop_reason,
                    "subagent_type": task.subagent_type,
                }
            )
        return result


def make_sp_agent(config: RunnableConfig, *, app_config: AppConfig | None = None):
    """Build one SP CentralAgent using DeerFlow's native agent loop.

    Ordinary DeerFlow tools stay in this graph. SP-specific control methods are
    added as intercepted tools, so their handlers update the same state and the
    same model loop without routing through a second CentralAgent.
    """
    resolved_app_config = app_config or get_app_config()
    runtime = _runtime_config(config)
    from deerflow.runtime.user_context import get_effective_user_id

    raw_user_id = runtime.get("user_id")
    user_id = str(raw_user_id) if raw_user_id else get_effective_user_id()
    raw_agent_name = runtime.get("sp_agent_name") or runtime.get("agent_name")
    agent_name = str(raw_agent_name) if raw_agent_name else None
    central_context: SPCentralRuntimeContext = build_sp_central_runtime_context(
        resolved_app_config,
        agent_name=agent_name,
        user_id=user_id,
    )
    model_name = _resolve_model_name(
        config,
        resolved_app_config,
        agent_model=central_context.agent_model,
    )
    model_config = resolved_app_config.get_model_config(model_name)
    thinking_enabled = bool(runtime.get("thinking_enabled", True))
    if model_config is not None and not model_config.supports_thinking:
        thinking_enabled = False
    metadata = dict(config.get("metadata") or {})
    metadata.update(
        {
            "agent_name": STACKPLANNER_ASSISTANT_ID,
            "model_name": model_name,
            "thinking_enabled": thinking_enabled,
            "orchestration_mode": STACKPLANNER_ASSISTANT_ID,
            "sp_agent_name": central_context.agent_name or "default",
            "available_skills": sorted(central_context.available_skill_names) if central_context.available_skill_names is not None else None,
        }
    )
    sp_config = dict(config)
    configurable = dict(sp_config.get("configurable", {}) or {})
    configurable.update(
        {
            "model_name": model_name,
            "thinking_enabled": thinking_enabled,
            "agent_name": central_context.agent_name,
            "orchestration_mode": STACKPLANNER_ASSISTANT_ID,
        }
    )
    sp_config["configurable"] = configurable
    sp_config["metadata"] = metadata

    executor_provider = DR2SPExecutorProvider(
        app_config=resolved_app_config,
        parent_model=model_name,
        runnable_config=sp_config,
        available_skill_names=central_context.available_skill_names,
        memory_agent_name=central_context.agent_name,
        user_id=user_id,
    )
    prompt_sections = [CENTRAL_AGENT_ACTION_PROMPT]
    if runtime.get("fresh_user_turn_after_terminal"):
        prompt_sections.append(FRESH_USER_TURN_PROMPT)
    if central_context.system_prompt_section:
        prompt_sections.append(central_context.system_prompt_section)
    if central_context.decision_context:
        prompt_sections.append(central_context.decision_context)
    from deerflow.agents.lead_agent.agent import _make_lead_agent
    from deerflow.sp.middlewares import TaskMemoryMiddleware
    from deerflow.sp.prompt import PromptContextBuilder

    graph = _make_lead_agent(
        sp_config,
        app_config=resolved_app_config,
        extra_tools=build_sp_control_tools(),
        extra_middlewares=[
            TaskMemoryMiddleware(context_builder=PromptContextBuilder(), inject_context=True),
            SPControlActionMiddleware(executor_provider=executor_provider),
            SPThinkLabelMiddleware(),
        ],
        prompt_prefix="\n\n".join(prompt_sections),
        identity_name="StackPlanner 2.0",
    )
    graph.metadata = dict(metadata)
    return graph
