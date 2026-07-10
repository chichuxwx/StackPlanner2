"""DR2 runtime factory for the StackPlanner orchestration graph."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.runtime import Runtime

from deerflow.agents.middlewares.title_middleware import TitleMiddleware
from deerflow.config.app_config import AppConfig, get_app_config
from deerflow.models import create_chat_model
from deerflow.sp.central import CENTRAL_AGENT_ACTION_PROMPT, create_sp_central_decider
from deerflow.sp.graph import DEFAULT_SP_MAX_ITERATIONS, create_sp_agent_graph
from deerflow.sp.subagents import DR2SubagentExecutorAdapter, SPSubagentExecutorProtocol, SPSubagentResult, SPSubagentTask
from deerflow.tracing import build_tracing_callbacks

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


def _runtime_config(config: RunnableConfig) -> dict[str, Any]:
    merged = dict(config.get("configurable", {}) or {})
    context = config.get("context", {}) or {}
    if isinstance(context, Mapping):
        merged.update(context)
    return merged


def _resolve_model_name(config: RunnableConfig, app_config: AppConfig) -> str:
    if not app_config.models:
        raise ValueError("StackPlanner requires at least one configured chat model.")
    runtime = _runtime_config(config)
    requested = runtime.get("model_name") or runtime.get("model")
    if requested and app_config.get_model_config(str(requested)) is not None:
        return str(requested)
    if requested:
        logger.warning("StackPlanner model %r is not configured; using %s", requested, app_config.models[0].name)
    return app_config.models[0].name


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


@dataclass(slots=True)
class DR2SPExecutorProvider:
    """Create DR2 SubagentExecutor instances bound to the active graph state."""

    app_config: AppConfig
    parent_model: str
    runnable_config: RunnableConfig
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
                user_id=str(context.get("user_id")) if context.get("user_id") else None,
                user_role=str(context.get("user_role")) if context.get("user_role") else None,
                oauth_provider=str(context.get("oauth_provider")) if context.get("oauth_provider") else None,
                oauth_id=str(context.get("oauth_id")) if context.get("oauth_id") else None,
                run_id=run_id,
                channel_user_id=str(context.get("channel_user_id")) if context.get("channel_user_id") else None,
                deerflow_trace_id=str(context.get("deerflow_trace_id")) if context.get("deerflow_trace_id") else None,
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
    """LangGraph-compatible factory used by the existing DR2 Gateway/RunWorker."""
    resolved_app_config = app_config or get_app_config()
    model_name = _resolve_model_name(config, resolved_app_config)
    runtime = _runtime_config(config)
    model_config = resolved_app_config.get_model_config(model_name)
    thinking_enabled = bool(runtime.get("thinking_enabled", True))
    if model_config is not None and not model_config.supports_thinking:
        thinking_enabled = False

    metadata = config.setdefault("metadata", {})
    metadata.update(
        {
            "agent_name": STACKPLANNER_ASSISTANT_ID,
            "model_name": model_name,
            "thinking_enabled": thinking_enabled,
            "orchestration_mode": STACKPLANNER_ASSISTANT_ID,
        }
    )
    tracing_callbacks = build_tracing_callbacks()
    if tracing_callbacks:
        callbacks = list(config.get("callbacks") or [])
        config["callbacks"] = [*callbacks, *tracing_callbacks]

    model = create_chat_model(
        name=model_name,
        thinking_enabled=thinking_enabled,
        app_config=resolved_app_config,
        attach_tracing=False,
    )
    max_iterations = int(runtime.get("sp_max_loop_iterations") or DEFAULT_SP_MAX_ITERATIONS)
    max_iterations = max(1, min(max_iterations, DEFAULT_SP_MAX_ITERATIONS))
    graph = create_sp_agent_graph(
        decider=create_sp_central_decider(model=model),
        system_prompt=CENTRAL_AGENT_ACTION_PROMPT,
        executor_provider=DR2SPExecutorProvider(
            app_config=resolved_app_config,
            parent_model=model_name,
            runnable_config=config,
        ),
        title_middleware=TitleMiddleware(app_config=resolved_app_config),
        max_iterations=max_iterations,
    )
    graph.metadata = dict(metadata)
    return graph
