"""Tests for DR2 SOUL, long-term memory, and Skill context in SP."""

from types import SimpleNamespace

import pytest

from deerflow.sp.central.runtime_context import build_sp_central_runtime_context
from deerflow.sp.runtime import DR2SPExecutorProvider, _apply_task_skill_policy
from deerflow.sp.subagents import SPSubagentTask
from deerflow.subagents.config import SubagentConfig


def _app_config():
    return SimpleNamespace(
        skills=SimpleNamespace(),
        memory=SimpleNamespace(
            enabled=True,
            injection_enabled=True,
            max_injection_tokens=600,
            token_counting="char",
            guaranteed_categories=["correction"],
            guaranteed_token_budget=100,
        ),
    )


def test_runtime_context_separates_soul_skills_and_long_term_memory(monkeypatch):
    profile = SimpleNamespace(
        name="sp-professor",
        model="custom-model",
        skills=["stackplanner-long-task", "deep-research"],
    )
    skills = [
        SimpleNamespace(name="deep-research", description="Verify evidence from multiple sources."),
        SimpleNamespace(name="disabled-by-profile", description="Must not be visible."),
        SimpleNamespace(name="stackplanner-long-task", description="Run the SP stage and HITL workflow."),
    ]
    monkeypatch.setattr("deerflow.sp.central.runtime_context.load_agent_config", lambda name, user_id=None: profile)
    monkeypatch.setattr("deerflow.sp.central.runtime_context.load_agent_soul", lambda name, user_id=None: "Use concise group-meeting language.")
    monkeypatch.setattr("deerflow.agents.lead_agent.prompt.get_enabled_skills_for_config", lambda app_config, user_id=None: skills)
    monkeypatch.setattr("deerflow.agents.memory.get_memory_data", lambda agent_name, user_id=None: {"facts": [{"content": "Prefer Chinese reports."}]})
    monkeypatch.setattr("deerflow.agents.memory.format_memory_for_injection", lambda memory_data, **kwargs: "- Prefer Chinese reports.")

    context = build_sp_central_runtime_context(
        _app_config(),
        agent_name="sp-professor",
        user_id="user-1",
    )

    assert context.agent_name == "sp-professor"
    assert context.agent_model == "custom-model"
    assert context.available_skill_names == frozenset({"deep-research", "stackplanner-long-task"})
    assert "Use concise group-meeting language." in context.system_prompt_section
    assert "stackplanner-long-task" in context.system_prompt_section
    assert "disabled-by-profile" not in context.system_prompt_section
    assert "Prefer Chinese reports." not in context.system_prompt_section
    assert "Prefer Chinese reports." in context.decision_context
    assert "not TaskMemoryStack" in context.decision_context


def test_runtime_context_honors_explicit_empty_agent_skill_whitelist(monkeypatch):
    profile = SimpleNamespace(name="no-skills", model=None, skills=[])
    monkeypatch.setattr("deerflow.sp.central.runtime_context.load_agent_config", lambda name, user_id=None: profile)
    monkeypatch.setattr("deerflow.sp.central.runtime_context.load_agent_soul", lambda name, user_id=None: "No tools in CentralAgent.")
    monkeypatch.setattr(
        "deerflow.agents.lead_agent.prompt.get_enabled_skills_for_config",
        lambda app_config, user_id=None: [SimpleNamespace(name="deep-research", description="Research")],
    )
    monkeypatch.setattr("deerflow.agents.memory.get_memory_data", lambda agent_name, user_id=None: {})
    monkeypatch.setattr("deerflow.agents.memory.format_memory_for_injection", lambda memory_data, **kwargs: "")

    context = build_sp_central_runtime_context(_app_config(), agent_name="no-skills", user_id="user-1")

    assert context.available_skill_names == frozenset()
    assert "<available_skills>" not in context.system_prompt_section


def test_delegate_skill_policy_loads_only_validated_requested_skills():
    config = SubagentConfig(name="sp-researcher", description="research", skills=[])
    task = SPSubagentTask(
        action_id="delegate-1",
        subagent_type="researcher",
        task="Research the evidence",
        description="Need verified facts",
        metadata={"skill_names": ["deep-research", "deep-research", "stackplanner-long-task"]},
    )

    selected = _apply_task_skill_policy(
        config,
        task,
        available_skill_names=frozenset({"deep-research", "stackplanner-long-task"}),
    )

    assert selected.skills == ["deep-research", "stackplanner-long-task"]
    assert config.skills == []


def test_delegate_skill_policy_rejects_disabled_or_invented_skill():
    config = SubagentConfig(name="sp-researcher", description="research", skills=[])
    task = SPSubagentTask(
        action_id="delegate-1",
        subagent_type="researcher",
        task="Research the evidence",
        description="Need verified facts",
        metadata={"skill_names": ["invented-skill"]},
    )

    with pytest.raises(ValueError, match="unavailable or disabled"):
        _apply_task_skill_policy(
            config,
            task,
            available_skill_names=frozenset({"deep-research"}),
        )


def test_delegate_without_skill_selection_preserves_specialist_default_empty_set():
    config = SubagentConfig(name="sp-reporter", description="report", skills=[])
    task = SPSubagentTask(
        action_id="delegate-1",
        subagent_type="reporter",
        task="Write the report",
        description="Synthesize artifacts",
    )

    selected = _apply_task_skill_policy(
        config,
        task,
        available_skill_names=frozenset({"stackplanner-long-task"}),
    )

    assert selected is config
    assert selected.skills == []


def test_sp_executor_provider_falls_back_to_factory_user_and_memory_scope(monkeypatch):
    captured = {}
    config = SubagentConfig(
        name="sp-memory-recaller",
        description="recall",
        skills=[],
        internal=True,
    )

    class FakeExecutor:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def execute(self, prompt):
            return SimpleNamespace(
                status="completed",
                result='{"summary":"No matching memory.","items":[]}',
                error=None,
                stop_reason=None,
                task_id="memory-task",
                token_usage_records=[],
            )

    monkeypatch.setattr("deerflow.subagents.get_subagent_config", lambda name, app_config=None: config)
    monkeypatch.setattr("deerflow.subagents.SubagentExecutor", FakeExecutor)
    provider = DR2SPExecutorProvider(
        app_config=SimpleNamespace(),
        parent_model="test-model",
        runnable_config={},
        available_skill_names=frozenset(),
        memory_agent_name="sp-professor",
        user_id="factory-user",
    )
    provider._tools = []
    executor = provider(
        {},
        SimpleNamespace(context={"thread_id": "thread-1", "run_id": "run-1"}, stream_writer=None),
    )

    result = executor.execute(
        SPSubagentTask(
            action_id="recall-1",
            subagent_type="memory_recaller",
            task="Recall preferences",
            description="Need prior preferences",
        )
    )

    assert result.is_success
    assert captured["user_id"] == "factory-user"
    assert captured["memory_agent_name"] == "sp-professor"
    assert captured["user_scoped_skills"] is True
