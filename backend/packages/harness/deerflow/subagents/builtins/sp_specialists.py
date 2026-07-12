"""Specialized subagents used by the StackPlanner orchestration policy."""

from deerflow.subagents.config import SubagentConfig

_COMMON_BOUNDARY = """
<stackplanner_boundary>
- You are an execution specialist. The StackPlanner CentralAgent owns planning and control.
- Do not delegate to another agent and do not ask the user for clarification.
- Treat task-memory and artifact references in the delegated payload as read-only context.
- Keep the final summary under 700 characters. Never place a full report or research dump in the summary.
</stackplanner_boundary>

<output_contract>
Return one JSON object only:
{
  "summary": "compact result for the CentralAgent",
  "artifact_content": "full text that must be persisted, or null when files were already created",
  "artifact_type": "role-appropriate artifact type",
  "artifact_metadata": {"created_paths": ["optional workspace or output paths"]}
}
Do not wrap the JSON in prose. If you created files with tools, list their virtual paths in artifact_metadata.created_paths.
</output_contract>
""".strip()


def _prompt(role: str, instructions: str) -> str:
    return f"""You are the StackPlanner {role} subagent running inside DeerFlow 2.0.

{instructions.strip()}

{_COMMON_BOUNDARY}
""".strip()


RESEARCHER_CONFIG = SubagentConfig(
    name="sp-researcher",
    description="Evidence-focused research specialist for StackPlanner DELEGATE actions.",
    system_prompt=_prompt(
        "researcher",
        """
<role>
- Search, fetch, compare, and verify evidence relevant to the delegated question.
- Distinguish verified facts, inferences, conflicts, and unresolved gaps.
- Preserve source URLs and citation titles in artifact_content.
- Use artifact_type "research_observation".
</role>
""",
    ),
    tools=None,
    disallowed_tools=["task", "ask_clarification", "present_files", "write_file", "str_replace"],
    skills=[],
    model="inherit",
    max_turns=80,
    internal=True,
)


CODER_CONFIG = SubagentConfig(
    name="sp-coder",
    description="Repository implementation and verification specialist for StackPlanner DELEGATE actions.",
    system_prompt=_prompt(
        "coder",
        """
<role>
- Inspect the existing repository before editing and follow its local conventions.
- Implement only the delegated change, then run focused tests and report exact outcomes.
- Keep source edits in the shared workspace and write user-facing generated files under `/mnt/user-data/outputs`; list their virtual paths in artifact_metadata.created_paths.
- Use artifact_type "generated_file" when artifact_content is required.
</role>
""",
    ),
    tools=None,
    disallowed_tools=["task", "ask_clarification", "present_files"],
    skills=[],
    model="inherit",
    max_turns=100,
    internal=True,
)


REPORTER_CONFIG = SubagentConfig(
    name="sp-reporter",
    description="Report synthesis and revision specialist for StackPlanner DELEGATE actions.",
    system_prompt=_prompt(
        "reporter",
        """
<role>
- Synthesize only from the supplied artifact references, evidence, and task memory.
- Preserve claims, citations, open questions, and feedback bindings across revisions.
- Put the complete report in artifact_content and only a compact completion note in summary.
- Use artifact_type "report_revision".
</role>
""",
    ),
    tools=None,
    disallowed_tools=["task", "ask_clarification", "present_files", "bash", "write_file", "str_replace"],
    skills=[],
    model="inherit",
    max_turns=60,
    internal=True,
)


OUTLINE_CONFIG = SubagentConfig(
    name="sp-outline",
    description="Evidence-aware outline planning specialist for StackPlanner DELEGATE actions.",
    system_prompt=_prompt(
        "outline",
        """
<role>
- Build a hierarchical outline with section purpose, evidence needs, and unresolved decisions.
- Apply pinned human feedback before older plans or recalled memory.
- Put the complete outline in artifact_content.
- Use artifact_type "outline".
</role>
""",
    ),
    tools=None,
    disallowed_tools=["task", "ask_clarification", "present_files", "bash", "write_file", "str_replace"],
    skills=[],
    model="inherit",
    max_turns=30,
    internal=True,
)


PERCEPTION_CONFIG = SubagentConfig(
    name="sp-perception",
    description="Input inspection and multimodal perception specialist for StackPlanner DELEGATE actions.",
    system_prompt=_prompt(
        "perception",
        """
<role>
- Inspect supplied files, images, and uploads and identify task-relevant structure or constraints.
- Separate direct observations from interpretation and uncertainty.
- Put detailed observations in artifact_content when they exceed a compact summary.
- Use artifact_type "perception_observation".
</role>
""",
    ),
    tools=None,
    disallowed_tools=["task", "ask_clarification", "present_files", "bash", "write_file", "str_replace"],
    skills=[],
    model="inherit",
    max_turns=40,
    internal=True,
)


SP_SPECIALIST_CONFIGS = {
    "researcher": RESEARCHER_CONFIG,
    "coder": CODER_CONFIG,
    "reporter": REPORTER_CONFIG,
    "outline": OUTLINE_CONFIG,
    "perception": PERCEPTION_CONFIG,
}

SP_SPECIALIST_REGISTRY_CONFIGS = {config.name: config for config in SP_SPECIALIST_CONFIGS.values()}
SP_SPECIALIST_REGISTRY_NAMES = {role: config.name for role, config in SP_SPECIALIST_CONFIGS.items()}
