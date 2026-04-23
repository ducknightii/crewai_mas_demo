"""端到端集成测试：真实 Crew 执行，验证 7 种事件类型 × 两层 hook 全部触发。

需要 LLM API（OPENAI_API_KEY + OPENAI_API_BASE）。
标记 @pytest.mark.integration，默认跳过，用 -m integration 运行。
"""

import json
import os
import sys
import textwrap
from pathlib import Path

import pytest

_M5L30_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_M5L30_DIR))

from hook_framework import (
    CrewObservabilityAdapter,
    EventType,
    HookLoader,
    HookRegistry,
)

pytestmark = pytest.mark.integration

# ── 共用的 counter hook 脚本（写 JSON 行到文件） ──────────────────

_COUNTER_HANDLER_CODE = textwrap.dedent('''\
    """Counter hook：每触发一次，往 {log_file} 追加一行 JSON。"""
    import json
    from pathlib import Path

    LOG_FILE = Path(r"{log_file}")

    def _write(ctx, layer):
        entry = {{"event": ctx.event_type.value, "layer": layer, "turn": ctx.turn_number}}
        if ctx.tool_name:
            entry["tool"] = ctx.tool_name
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(entry) + "\\n")

    def before_turn_handler(ctx):   _write(ctx, "{layer}")
    def before_llm_handler(ctx):    _write(ctx, "{layer}")
    def before_tool_handler(ctx):   _write(ctx, "{layer}")
    def after_tool_handler(ctx):    _write(ctx, "{layer}")
    def after_turn_handler(ctx):    _write(ctx, "{layer}")
    def task_complete_handler(ctx):  _write(ctx, "{layer}")
    def session_end_handler(ctx):    _write(ctx, "{layer}")
''')

_HOOKS_YAML_ALL_EVENTS = textwrap.dedent('''\
    hooks:
      BEFORE_TURN:
        - handler: counter.before_turn_handler
      BEFORE_LLM:
        - handler: counter.before_llm_handler
      BEFORE_TOOL_CALL:
        - handler: counter.before_tool_handler
      AFTER_TOOL_CALL:
        - handler: counter.after_tool_handler
      AFTER_TURN:
        - handler: counter.after_turn_handler
      TASK_COMPLETE:
        - handler: counter.task_complete_handler
      SESSION_END:
        - handler: counter.session_end_handler
''')


def _setup_hook_dirs(tmp_path):
    """创建两层 hook 目录，每层覆盖全部 7 种事件。"""
    log_file = tmp_path / "hook_events.jsonl"

    # 全局层
    global_dir = tmp_path / "shared_hooks"
    global_dir.mkdir()
    (global_dir / "hooks.yaml").write_text(_HOOKS_YAML_ALL_EVENTS)
    (global_dir / "counter.py").write_text(
        _COUNTER_HANDLER_CODE.format(log_file=str(log_file), layer="global")
    )

    # Workspace 层
    ws_dir = tmp_path / "workspace"
    ws_hooks = ws_dir / "hooks"
    ws_hooks.mkdir(parents=True)
    (ws_hooks / "hooks.yaml").write_text(_HOOKS_YAML_ALL_EVENTS)
    (ws_hooks / "counter.py").write_text(
        _COUNTER_HANDLER_CODE.format(log_file=str(log_file), layer="workspace")
    )

    return global_dir, ws_dir, log_file


def _parse_log(log_file: Path) -> list[dict]:
    if not log_file.exists():
        return []
    lines = log_file.read_text().strip().split("\n")
    return [json.loads(line) for line in lines if line.strip()]


# ── T15: 全链路事件覆盖（真实 Crew） ────────────────────────────

def test_all_event_types_fired_both_layers(tmp_path):
    """真实 Crew kickoff → 断言 7 种事件类型 × 2 层 hook 全部至少触发 1 次。"""
    from crewai import Agent, Crew, LLM, Task
    from crewai.hooks import clear_all_global_hooks
    from crewai.tools import BaseTool
    from pydantic import BaseModel, Field

    clear_all_global_hooks()

    # 自定义搜索工具（确保触发 tool call 事件）
    class QInput(BaseModel):
        query: str = Field(description="q")

    class FakeTool(BaseTool):
        name: str = "fake_search"
        description: str = "搜索信息"
        args_schema: type[BaseModel] = QInput

        def _run(self, query: str) -> str:
            return f"关于{query}的结果：这是测试数据。"

    # 设置两层 hook 目录
    global_dir, ws_dir, log_file = _setup_hook_dirs(tmp_path)

    # 初始化 Hook 框架
    registry = HookRegistry()
    loader = HookLoader(registry)
    loader.load_two_layers(global_dir, ws_dir)

    # 验证加载：每种事件 2 个 handler（global + workspace）
    for et in EventType:
        count = registry.handler_count(et)
        assert count == 2, f"{et.value} should have 2 handlers (global + workspace), got {count}"

    # 适配层
    adapter = CrewObservabilityAdapter(registry, session_id="e2e-test")
    adapter.install_global_hooks()

    # 构建 Crew
    base_url = os.environ.get(
        "OPENAI_API_BASE",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    model_name = os.environ.get("AGENT_MODEL", "qwen-plus")
    llm = LLM(model=model_name, base_url=base_url)

    agent = Agent(
        role="Tester",
        goal="测试 hook 事件触发",
        backstory="你是测试助手，用 fake_search 搜索后总结一句话。",
        llm=llm,
        verbose=False,
        tools=[FakeTool()],
    )

    task = Task(
        description="使用 fake_search 搜索'hook测试'，然后用一句话总结结果。",
        expected_output="一句话总结",
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=False,
        step_callback=adapter.make_step_callback(),
        task_callback=adapter.make_task_callback(),
    )

    # 执行
    result = crew.kickoff()
    adapter.cleanup()

    # 解析日志
    entries = _parse_log(log_file)
    assert len(entries) > 0, "No hook events were logged"

    # 按 (event, layer) 分组
    fired = {(e["event"], e["layer"]) for e in entries}

    # 断言：7 种事件类型 × 2 层 = 14 种组合全部出现
    for et in EventType:
        for layer in ("global", "workspace"):
            assert (et.value, layer) in fired, (
                f"Missing: event={et.value} layer={layer}\n"
                f"Fired: {sorted(fired)}\n"
                f"All entries: {entries}"
            )

    # 额外断言：BEFORE_TURN 至少 1 次
    before_turns = [e for e in entries if e["event"] == "before_turn"]
    assert len(before_turns) >= 2, f"Expected ≥2 BEFORE_TURN (2 layers), got {len(before_turns)}"

    # 额外断言：tool 事件有 tool name
    tool_events = [e for e in entries if e["event"] in ("before_tool_call", "after_tool_call")]
    assert len(tool_events) >= 2, f"Expected tool events, got {len(tool_events)}"
    for te in tool_events:
        assert te.get("tool") == "fake_search", f"Tool event missing tool name: {te}"

    print(f"\n✅ E2E: {len(entries)} hook events, all 7×2 combinations fired")
    print(f"   Result: {str(result)[:100]}")


# ── T16: Langfuse trace 创建验证 ──────────────────────────────

def test_langfuse_trace_created(tmp_path):
    """用真实 shared_hooks（含 langfuse_trace）跑 Crew → 验证 Langfuse trace 有 observations。"""
    from crewai import Agent, Crew, LLM, Task
    from crewai.hooks import clear_all_global_hooks
    from crewai.tools import BaseTool
    from pydantic import BaseModel, Field

    langfuse_host = os.environ.get("LANGFUSE_HOST", "http://localhost:3000")
    langfuse_pk = os.environ.get("LANGFUSE_PUBLIC_KEY")
    langfuse_sk = os.environ.get("LANGFUSE_SECRET_KEY")
    if not langfuse_pk or not langfuse_sk:
        pytest.skip("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not set")

    clear_all_global_hooks()

    class QInput(BaseModel):
        query: str = Field(description="q")

    class FakeTool(BaseTool):
        name: str = "fake_search"
        description: str = "搜索信息"
        args_schema: type[BaseModel] = QInput

        def _run(self, query: str) -> str:
            return f"关于{query}的结果：Langfuse测试数据。"

    # 用真实 shared_hooks（含 langfuse_trace）+ 临时 workspace
    global_dir = _M5L30_DIR / "shared_hooks"
    ws_dir = tmp_path / "workspace"
    ws_hooks = ws_dir / "hooks"
    ws_hooks.mkdir(parents=True)
    (ws_hooks / "hooks.yaml").write_text(textwrap.dedent("""\
        hooks:
          TASK_COMPLETE:
            - handler: counter.task_complete_handler
    """))
    log_file = tmp_path / "ws_log.jsonl"
    (ws_hooks / "counter.py").write_text(
        _COUNTER_HANDLER_CODE.format(log_file=str(log_file), layer="workspace")
    )

    registry = HookRegistry()
    loader = HookLoader(registry)
    loader.load_two_layers(global_dir, ws_dir)

    session_id = "e2e-langfuse-test"
    adapter = CrewObservabilityAdapter(registry, session_id=session_id)
    adapter.install_global_hooks()

    base_url = os.environ.get(
        "OPENAI_API_BASE",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    model_name = os.environ.get("AGENT_MODEL", "qwen-plus")
    llm = LLM(model=model_name, base_url=base_url)

    agent = Agent(
        role="Tester",
        goal="测试 Langfuse trace",
        backstory="你是测试助手，搜索后总结。",
        llm=llm,
        verbose=False,
        tools=[FakeTool()],
    )
    task = Task(
        description="用 fake_search 搜索'langfuse测试'，用一句话总结。",
        expected_output="一句话",
        agent=agent,
    )
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=False,
        step_callback=adapter.make_step_callback(),
        task_callback=adapter.make_task_callback(),
    )

    result = crew.kickoff()
    adapter.cleanup()

    # 验证 Langfuse trace
    import time
    time.sleep(5)

    from langfuse import Langfuse

    client = Langfuse()
    trace_id = client.create_trace_id(seed=session_id)
    trace = client.api.trace.get(trace_id)

    assert len(trace.observations) >= 2, (
        f"Expected ≥2 observations (tool + generation), got {len(trace.observations)}"
    )

    obs_types = {obs.type for obs in trace.observations}
    assert "TOOL" in obs_types, f"Missing TOOL observation. Types: {obs_types}"
    assert "GENERATION" in obs_types, f"Missing GENERATION observation. Types: {obs_types}"

    print(f"\n✅ Langfuse: trace {trace_id[:16]}... has {len(trace.observations)} observations")
    for obs in trace.observations:
        print(f"   {obs.type}: {obs.name}")
