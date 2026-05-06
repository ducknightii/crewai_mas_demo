"""31课 Demo：观测 + 可靠性三策略（在 30 课完整 Agent 基础上增量升级）。

演示架构（复用 25 课框架 + 30 课 Hook 骨架 + 31 课可靠性策略）：
  - build_bootstrap_prompt()：加载 workspace 四件套（soul / user / agent / memory）
  - SkillLoaderTool：渐进式披露，task 型 Skill 启动 Sub-Crew 在沙盒中执行
  - Hook 框架（两层加载）+ Langfuse 全链路追踪
  - 31课新增：hooks.yaml strategies 段自动加载可靠性策略
    - RetryTracker：重试追踪（纯观测）
    - CostGuard：成本围栏（超预算 deny）
    - LoopDetector：循环检测（重复状态 deny）

前置条件：
    docker compose -f sandbox-docker-compose.yaml up -d

运行方式：
    python3 demo.py
    python3 demo.py "为一个短链接服务产出技术设计文档"

    # 低预算模式（触发成本围栏）
    COST_GUARD_BUDGET=0.001 python3 demo.py
"""

from __future__ import annotations

import atexit
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_M5L31_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _M5L31_DIR.parent
for _p in [str(_M5L31_DIR), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv
load_dotenv(_M5L31_DIR / ".env", override=True)

from crewai import Agent, Crew, LLM, Task

from m3l20.m3l20_file_memory import build_bootstrap_prompt
from tools.skill_loader_tool import SkillLoaderTool

from hook_framework import (
    CrewObservabilityAdapter,
    GuardrailDeny,
    HookLoader,
    HookRegistry,
)

WORKSPACE_DIR = _M5L31_DIR / "workspace" / "demo_agent"
SKILLS_DIR = WORKSPACE_DIR / "skills"
OUTPUT_DIR = WORKSPACE_DIR / "output"

SANDBOX_MCP_URL = "http://localhost:8030/mcp"
SANDBOX_MOUNT_DESC = (
    "1. 所有操作必须在沙盒中执行，不得操作本地文件系统。\n"
    "   已挂载目录：\n"
    "   - skills → /mnt/skills:ro（Skill 资源，只读）\n"
    "   - output → /workspace/output:rw（产出物，可读写）\n\n"
    "2. 产出文件必须写到 /workspace/output/ 目录下\n"
    "3. 如遇依赖缺失，先在沙盒中安装再继续"
)

DEFAULT_TASK = "为一个用户注册功能产出技术设计文档"


def main():
    task_desc = " ".join(sys.argv[1:]).strip() or DEFAULT_TASK
    session_id = f"sess_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    # ── 1. Hook 框架初始化（观测 + 可靠性策略，全部 YAML 声明式加载）─────
    registry = HookRegistry()
    loader = HookLoader(registry)
    loader.load_two_layers(
        global_dir=_M5L31_DIR / "shared_hooks",
        workspace_dir=WORKSPACE_DIR,
    )

    summary = registry.summary()
    total = sum(len(v) for v in summary.values())
    print(f"Session: {session_id}")
    print(f"HookRegistry: {total} handlers loaded")
    for event, handlers in summary.items():
        for h in handlers:
            print(f"   {h} -> {event}")
    if loader.strategies:
        print(f"Strategies: {list(loader.strategies.keys())}")
    print()

    # ── 2. CrewAI 适配层 ───────────────────────────────────────────────────
    adapter = CrewObservabilityAdapter(registry, session_id=session_id)
    adapter.install_global_hooks()
    atexit.register(adapter.cleanup)

    # ── 3. Bootstrap + SkillLoader ─────────────────────────────────────────
    backstory = build_bootstrap_prompt(WORKSPACE_DIR)

    skill_tool = SkillLoaderTool(
        skills_dir=str(SKILLS_DIR),
        sandbox_mcp_url=SANDBOX_MCP_URL,
        sandbox_mount_desc=SANDBOX_MOUNT_DESC,
        step_callback=adapter.make_sub_crew_step_callback(),   # 💡 Sub-Crew 可观测性
        task_callback=adapter.make_task_callback(),            # 💡 Sub-Crew task 完成追踪
    )

    model_name = os.environ.get("AGENT_MODEL", "qwen-plus")
    base_url = os.environ.get(
        "OPENAI_API_BASE",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    llm = LLM(model=model_name, base_url=base_url)

    agent = Agent(
        role="数字员工",
        goal="根据用户请求，调用合适的 Skill 高效完成任务",
        backstory=backstory,
        llm=llm,
        verbose=True,
        tools=[skill_tool],
    )

    task = Task(
        description=(
            f"用户请求：{task_desc}\n\n"
            "请先调用 skill_loader 工具加载合适的 Skill 获取工作指引，"
            "然后严格按照指引完成任务。"
        ),
        expected_output="按照 Skill 指引产出的完整交付物",
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True,
        step_callback=adapter.make_step_callback(),
        task_callback=adapter.make_task_callback(),
    )

    # ── 4. 执行 ────────────────────────────────────────────────────────────
    print(f"Task: {task_desc}\n")
    print(f"Bootstrap: {len(backstory)} chars from workspace/demo_agent/")
    print(f"Skills: {list(skill_tool._skill_registry.keys())}\n")

    try:
        result = crew.kickoff()
        print(f"\n{'='*60}")
        print(f"Result:\n{result}")
    except GuardrailDeny as e:
        print(f"\n{'='*60}")
        print(f"Guardrail triggered: {e}")

    # ── 5. 清理 + 度量 ────────────────────────────────────────────────────
    adapter.cleanup()

    print(f"\n{'='*60}")
    print(f"Langfuse: http://localhost:3000")
    design_doc = OUTPUT_DIR / "design_doc.md"
    if design_doc.exists():
        print(f"Design doc: {design_doc}")
    audit_file = WORKSPACE_DIR / "audit.log"
    if audit_file.exists():
        print(f"Audit log: {audit_file}")

    if loader.strategies:
        print(f"\n{'='*60}")
        print("Guardrail Metrics:")
        for name, strategy in loader.strategies.items():
            if hasattr(strategy, "get_metrics"):
                metrics = strategy.get_metrics()
                print(f"\n  [{name}]")
                for k, v in metrics.items():
                    print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
