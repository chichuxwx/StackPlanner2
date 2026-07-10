"""Registration and isolation tests for SP-only DR2 subagents."""

from deerflow.subagents.builtins import BUILTIN_SUBAGENTS, SP_SPECIALIST_CONFIGS, SP_SPECIALIST_REGISTRY_NAMES
from deerflow.subagents.registry import get_available_subagent_names, get_subagent_config, get_subagent_names


def test_all_designed_sp_specialists_are_registered():
    expected = {"researcher", "coder", "reporter", "outline", "perception"}
    expected_registry_names = {f"sp-{name}" for name in expected}

    assert set(SP_SPECIALIST_CONFIGS) == expected
    assert set(SP_SPECIALIST_REGISTRY_NAMES.values()) == expected_registry_names
    assert expected_registry_names <= set(BUILTIN_SUBAGENTS)
    assert expected_registry_names <= set(get_subagent_names())

    for role in expected:
        config = get_subagent_config(SP_SPECIALIST_REGISTRY_NAMES[role])
        assert config is not None
        assert config.name == f"sp-{role}"
        assert config.internal is True
        assert "Return one JSON object only" in config.system_prompt
        assert "Never place a full report or research dump in the summary" in config.system_prompt
        assert "task" in config.disallowed_tools
        assert "ask_clarification" in config.disallowed_tools


def test_sp_specialists_stay_hidden_from_default_lead_agent_task_tool(monkeypatch):
    from deerflow.subagents import registry as registry_module

    monkeypatch.setattr(registry_module, "is_host_bash_allowed", lambda: True)

    assert get_available_subagent_names() == ["general-purpose", "bash"]


def test_sp_specialists_have_role_specific_artifact_contracts():
    expected_types = {
        "researcher": "research_observation",
        "coder": "generated_file",
        "reporter": "report_revision",
        "outline": "outline",
        "perception": "perception_observation",
    }

    for name, artifact_type in expected_types.items():
        assert f'Use artifact_type "{artifact_type}"' in SP_SPECIALIST_CONFIGS[name].system_prompt
