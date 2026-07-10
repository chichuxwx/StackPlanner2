"""Tests for the built-in StackPlanner memory_recaller subagent."""

from deerflow.subagents.builtins import BUILTIN_SUBAGENTS, MEMORY_RECALLER_CONFIG
from deerflow.subagents.registry import get_subagent_config, get_subagent_names


def test_memory_recaller_is_registered_as_builtin_subagent():
    assert BUILTIN_SUBAGENTS["memory_recaller"] is MEMORY_RECALLER_CONFIG
    assert "memory_recaller" in get_subagent_names()

    config = get_subagent_config("memory_recaller")

    assert config is not None
    assert config.name == "memory_recaller"
    assert config.max_turns == 20
    assert config.internal is True


def test_memory_recaller_is_read_only_and_cannot_delegate_or_write_files():
    config = get_subagent_config("memory_recaller")
    assert config is not None

    assert "task" in config.disallowed_tools
    assert "bash" in config.disallowed_tools
    assert "write_file" in config.disallowed_tools
    assert "Do not write, update, promote, delete, or mutate long-term memory." in config.system_prompt
    assert "Return compact JSON only" in config.system_prompt
