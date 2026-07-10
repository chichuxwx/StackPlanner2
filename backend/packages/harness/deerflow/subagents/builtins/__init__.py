"""Built-in subagent configurations."""

from .bash_agent import BASH_AGENT_CONFIG
from .general_purpose import GENERAL_PURPOSE_CONFIG
from .memory_recaller import MEMORY_RECALLER_CONFIG
from .sp_specialists import CODER_CONFIG, OUTLINE_CONFIG, PERCEPTION_CONFIG, REPORTER_CONFIG, RESEARCHER_CONFIG, SP_SPECIALIST_CONFIGS

__all__ = [
    "GENERAL_PURPOSE_CONFIG",
    "BASH_AGENT_CONFIG",
    "MEMORY_RECALLER_CONFIG",
    "RESEARCHER_CONFIG",
    "CODER_CONFIG",
    "REPORTER_CONFIG",
    "OUTLINE_CONFIG",
    "PERCEPTION_CONFIG",
    "SP_SPECIALIST_CONFIGS",
]

# Registry of built-in subagents
BUILTIN_SUBAGENTS = {
    "general-purpose": GENERAL_PURPOSE_CONFIG,
    "bash": BASH_AGENT_CONFIG,
    "memory_recaller": MEMORY_RECALLER_CONFIG,
    **SP_SPECIALIST_CONFIGS,
}
