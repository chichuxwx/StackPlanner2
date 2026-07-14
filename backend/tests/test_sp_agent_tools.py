"""Focused tests for the single-loop SP control-tool layer."""

import asyncio
from types import SimpleNamespace

from langchain_core.messages import AIMessage, ToolMessage

from deerflow.sp.agent_tools import SPControlActionMiddleware, SPThinkLabelMiddleware, _action_payload, build_sp_control_tools
from deerflow.sp.memory import TaskMemoryStack


def test_sp_control_tools_are_prefixed_and_do_not_replace_native_tools():
    names = [tool.name for tool in build_sp_control_tools()]

    assert names == [
        "sp_think",
        "sp_delegate",
        "sp_recall_memory",
        "sp_reflect",
        "sp_revise",
        "sp_backtrack",
        "sp_replan",
        "sp_summarize",
        "sp_ask_human",
        "sp_finish",
    ]


def test_action_payload_adds_sp_type_without_forcing_a_json_response():
    payload = _action_payload(
        "sp_reflect",
        {"task": "Check the evidence", "reason": "The result is incomplete"},
        tool_call_id="tool-1",
    )

    assert payload["action_type"] == "REFLECT"
    assert payload["action_id"] == "spact_tool-1"
    assert payload["task"] == "Check the evidence"


def test_revise_action_payload_targets_memory_entries_and_keeps_correction_as_task():
    payload = _action_payload(
        "sp_revise",
        {
            "target_entry_ids": ["spmem-wrong"],
            "correction": "The source is from the official website.",
            "reason": "The previous observation used an unofficial mirror.",
        },
        tool_call_id="tool-revise",
    )

    assert payload["action_type"] == "REVISE"
    assert payload["task"] == "The source is from the official website."
    assert payload["metadata"]["target_entry_ids"] == ["spmem-wrong"]
    assert payload["metadata"]["revision_reason"] == payload["reason"]


def test_summarize_action_payload_carries_source_entry_ids():
    payload = _action_payload(
        "sp_summarize",
        {
            "summary": "Keep only the verified implementation decision.",
            "source_entry_ids": ["spmem-old-1", "spmem-old-2"],
        },
        tool_call_id="tool-summary",
    )

    assert payload["action_type"] == "SUMMARIZE"
    assert payload["task"] == "Keep only the verified implementation decision."
    assert payload["metadata"]["source_entry_ids"] == ["spmem-old-1", "spmem-old-2"]


def test_sp_control_middleware_executes_handler_in_place():
    middleware = SPControlActionMiddleware()
    request = SimpleNamespace(
        tool_call={
            "name": "sp_reflect",
            "id": "tool-reflect",
            "args": {"task": "Inspect the failed result"},
        },
        state={},
        runtime=SimpleNamespace(context={"thread_id": "thread-1", "run_id": "run-1"}),
    )

    command = middleware.wrap_tool_call(request, lambda _: (_ for _ in ()).throw(AssertionError("native tool handler must not run")))

    assert command.update["sp_last_action_id"] == "spact_tool-reflect"
    assert command.update["sp_last_handler_result"]["action_type"] == "REFLECT"
    assert command.update["messages"][0].additional_kwargs["stackplanner"]["action_type"] == "REFLECT"


def test_sp_control_middleware_executes_handler_in_async_stream():
    middleware = SPControlActionMiddleware()
    request = SimpleNamespace(
        tool_call={"name": "sp_think", "id": "tool-think", "args": {"task": "Make a checkpoint"}},
        state={},
        runtime=SimpleNamespace(context={"thread_id": "thread-1", "run_id": "run-1"}),
    )

    async def run():
        async def unexpected_handler(_):
            raise AssertionError("SP control tool must not call the native handler")

        return await middleware.awrap_tool_call(request, unexpected_handler)

    command = asyncio.run(run())

    assert command.update["sp_last_handler_result"]["action_type"] == "THINK"


def test_sp_control_middleware_runs_central_memory_revision_in_place():
    stack = TaskMemoryStack()
    wrong_entry = stack.append_observe("The unofficial result is authoritative.", actor="deerflow")
    middleware = SPControlActionMiddleware()
    request = SimpleNamespace(
        tool_call={
            "name": "sp_revise",
            "id": "tool-revise",
            "args": {
                "target_entry_ids": [wrong_entry.id],
                "correction": "The result requires verification against an official source.",
                "reason": "The source was unofficial.",
            },
        },
        state={"sp_task_memory": stack.to_dict()},
        runtime=SimpleNamespace(context={"thread_id": "thread-1", "run_id": "run-1"}),
    )

    command = middleware.wrap_tool_call(request, lambda _: (_ for _ in ()).throw(AssertionError("native tool handler must not run")))

    entries = command.update["sp_task_memory"]["entries"]
    assert entries[0]["status"] == "superseded"
    assert entries[-1]["action"] == "revise"
    assert entries[-1]["parent_ids"] == [wrong_entry.id]
    assert command.update["sp_last_handler_result"]["action_type"] == "REVISE"


def test_think_label_middleware_marks_normal_model_turns():
    middleware = SPThinkLabelMiddleware()
    update = middleware.after_model(
        {"messages": [AIMessage(content="I will answer directly", id="ai-1")]},
        SimpleNamespace(context={"run_id": "run-1"}),
    )

    assert update == {
        "sp_last_handler_result": {
            "action_type": "THINK",
            "action_id": "ai-1",
            "next_step": "continue",
        }
    }


def test_think_label_middleware_persists_useful_native_tool_observation():
    middleware = SPThinkLabelMiddleware()
    tool_result = ToolMessage(
        content='{"results":[{"title":"StackPlanner","url":"https://example.com"}]}',
        tool_call_id="search-1",
        name="web_search",
    )
    request = SimpleNamespace(tool_call={"name": "web_search", "id": "search-1", "args": {"query": "StackPlanner"}})

    assert middleware.wrap_tool_call(request, lambda _: tool_result) is tool_result

    update = middleware.after_model(
        {
            "messages": [
                tool_result,
                AIMessage(content="根据搜索结果继续处理", id="ai-2"),
            ]
        },
        SimpleNamespace(context={"thread_id": "thread-1", "run_id": "run-1"}),
    )

    entry = update["sp_task_memory"]["entries"][-1]
    assert entry["action"] == "observe"
    assert entry["actor"] == "deerflow"
    assert entry["metadata"]["tool_name"] == "web_search"
    assert update["sp_last_handler_result"]["action_type"] == "THINK"


def test_think_label_middleware_merges_multiple_native_tool_observations():
    middleware = SPThinkLabelMiddleware()
    messages = [
        ToolMessage(content="first result", tool_call_id="search-1", name="web_search"),
        ToolMessage(content="second result", tool_call_id="search-2", name="web_search"),
        AIMessage(content="已完成搜索", id="ai-2"),
    ]

    update = middleware.after_model(
        {"messages": messages},
        SimpleNamespace(context={"thread_id": "thread-1", "run_id": "run-1"}),
    )

    entries = update["sp_task_memory"]["entries"]
    assert [entry["metadata"]["tool_call_id"] for entry in entries] == ["search-1", "search-2"]


def test_think_label_middleware_does_not_persist_file_listing_or_success_noise():
    middleware = SPThinkLabelMiddleware()
    update = middleware.after_model(
        {
            "messages": [
                ToolMessage(content="-rw-r--r-- report.md", tool_call_id="ls-1", name="ls"),
                ToolMessage(content="Successfully presented files", tool_call_id="present-1", name="present_files"),
                ToolMessage(content="# Skill documentation\nUse the chart generator.", tool_call_id="read-1", name="read_file"),
                ToolMessage(content="-rw-rw-r-- 1 jxk jxk 78993 /mnt/user-data/outputs/chart.png", tool_call_id="bash-1", name="bash"),
                AIMessage(content="已完成", id="ai-3"),
            ]
        },
        SimpleNamespace(context={"thread_id": "thread-1", "run_id": "run-1"}),
    )

    assert "sp_task_memory" not in update


def test_think_label_middleware_deduplicates_repeated_errors():
    middleware = SPThinkLabelMiddleware()
    error = "Error: Unsafe absolute paths in command: /scripts/generate.js"
    update = middleware.after_model(
        {
            "messages": [
                ToolMessage(content=error, tool_call_id="bash-1", name="bash"),
                ToolMessage(content=error, tool_call_id="bash-2", name="bash"),
                AIMessage(content="停止重复尝试", id="ai-4"),
            ]
        },
        SimpleNamespace(context={"thread_id": "thread-1", "run_id": "run-1"}),
    )

    entries = update["sp_task_memory"]["entries"]
    assert len(entries) == 1
    assert entries[0]["action"] == "error"
    assert entries[0]["priority"] == "high"


def test_think_label_middleware_supports_async_native_tools():
    middleware = SPThinkLabelMiddleware()
    request = SimpleNamespace(tool_call={"name": "web_search", "id": "search-async", "args": {}})
    tool_result = ToolMessage(content="async result", tool_call_id="search-async", name="web_search")

    async def run():
        async def handler(_):
            return tool_result

        return await middleware.awrap_tool_call(request, handler)

    assert asyncio.run(run()) is tool_result
