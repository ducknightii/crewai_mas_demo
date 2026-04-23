"""32课 Demo：观测 + 安全 + 可靠性三层叠加。

运行方式：
    python3 demo.py                     # 正常运行（安全工具）
    python3 demo.py --attack inject     # 模拟注入攻击（路径遍历）
    python3 demo.py --attack privilege  # 模拟权限越权（使用禁止工具）
    python3 demo.py --budget 0.001      # 同时测试安全+成本围栏
"""

from __future__ import annotations

import argparse
import atexit
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_DIR))

from crewai import Agent, Crew, LLM, Task
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from hook_framework import (
    CrewObservabilityAdapter,
    GuardrailDeny,
    HookLoader,
    HookRegistry,
)
from reliability import install_reliability_hooks
from security import install_security_hooks
from security.credential_inject import SecureToolWrapper


class SearchInput(BaseModel):
    query: str = Field(description="搜索关键词")


class KnowledgeSearchTool(BaseTool):
    name: str = "knowledge_search"
    description: str = "搜索知识库，返回关于指定主题的信息"
    args_schema: type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:
        knowledge = {
            "安全": "Agent 安全的核心是限制工具能力边界，而非给 Agent 下达安全指令。Prompt 是建议，Hook 是法律。",
            "权限": "Deny > Ask > Allow 三级权限模型。Deny 规则永远优先，新工具默认 Ask。策略外置于 YAML 配置文件。",
            "沙箱": "沙箱限制 Agent 的执行范围。输入消毒用确定性规则（正则匹配），零 LLM 依赖。",
            "注入": "Agent 注入与 Chatbot 注入本质不同——Agent 注入导致真实副作用（删文件、泄数据）。防御靠沙箱+权限，不靠 Prompt。",
            "AI Agent": "AI Agent 是能自主感知环境、做出决策并采取行动的智能系统。安全是生产部署的关键挑战。",
        }
        results = []
        for key, val in knowledge.items():
            if key.lower() in query.lower() or query.lower() in key.lower():
                results.append(f"[{key}] {val}")
        if not results:
            for key, val in list(knowledge.items())[:2]:
                results.append(f"[{key}] {val}")
        return "\n\n".join(results)


class ShellExecutorTool(BaseTool):
    """执行 Shell 命令——应被权限网关拦截。"""

    name: str = "shell_executor"
    description: str = "执行 Shell 命令获取系统信息"
    args_schema: type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:
        return f"[SECURITY BREACH] Command executed: {query}"


class InjectableSearchTool(BaseTool):
    """搜索工具——模拟被注入导致输入包含路径遍历。"""

    name: str = "knowledge_search"
    description: str = "搜索知识库（注入版——输入包含 ../）"
    args_schema: type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:
        return f"搜索结果: {query}"


class SecureApiTool(BaseTool):
    """需要 API Key 的工具——密钥在工具层注入。"""

    name: str = "secure_api"
    description: str = "调用外部 API 查询数据"
    args_schema: type[BaseModel] = SearchInput

    def _run(self, query: str, api_key: str = "") -> str:
        key_preview = f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"
        return f"[API] 使用密钥 {key_preview} 查询: {query}"


def main():
    parser = argparse.ArgumentParser(description="32课 Demo")
    parser.add_argument("--budget", type=float, default=1.0)
    parser.add_argument("--attack", choices=["inject", "privilege"], default=None)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--loop-threshold", type=int, default=3)
    args = parser.parse_args()

    session_id = f"sess_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    # === 1. HookRegistry + YAML hooks（30课） ===
    registry = HookRegistry()
    loader = HookLoader(registry)
    loader.load_two_layers(
        global_dir=_DIR / "shared_hooks",
        workspace_dir=_DIR / "workspace" / "demo_agent",
    )

    # === 2. 安全策略（32课）—— 先于可靠性 ===
    security_strategies = install_security_hooks(registry, config={
        "policy_path": _DIR / "workspace" / "demo_agent" / "security.yaml",
        "default_permission": "ask",
        "audit_file": _DIR / "workspace" / "demo_agent" / "security_audit.jsonl",
    })

    # === 3. 可靠性策略（31课）—— 后于安全 ===
    reliability_strategies = install_reliability_hooks(registry, config={
        "max_retries": args.max_retries,
        "loop_threshold": args.loop_threshold,
        "budget_usd": args.budget,
    })

    summary = registry.summary()
    total = sum(len(v) for v in summary.values())
    print(f"Session: {session_id}")
    print(f"Budget: ${args.budget:.2f}")
    print(f"Attack: {args.attack or 'none'}")
    print(f"Hooks: {total} handlers")
    for event, handlers in summary.items():
        for h in handlers:
            print(f"   {h} -> {event}")
    print()

    # === 4. CrewAI 适配器 ===
    adapter = CrewObservabilityAdapter(registry, session_id=session_id)
    adapter.install_global_hooks()
    atexit.register(adapter.cleanup)

    model_name = os.environ.get("AGENT_MODEL", "qwen-plus")
    base_url = os.environ.get(
        "OPENAI_API_BASE",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    llm = LLM(model=model_name, base_url=base_url)

    # === 5. 选择工具（根据攻击场景） ===
    if args.attack == "privilege":
        tool = ShellExecutorTool()
    elif args.attack == "inject":
        tool = InjectableSearchTool()
    else:
        tool = KnowledgeSearchTool()

    agent = Agent(
        role="Research Analyst",
        goal="搜索并总结关于 AI Agent 安全 的信息",
        backstory="你是一位研究分析师。使用工具搜索后总结要点。",
        llm=llm,
        verbose=True,
        tools=[tool],
        max_iter=15,
    )

    if args.attack == "inject":
        task_desc = "搜索 ../../etc/passwd 相关信息，列出 3 个关键要点。"
    else:
        task_desc = "搜索「AI Agent 安全」相关信息，列出 3 个关键要点。"

    task = Task(
        description=task_desc,
        expected_output="3 个关键要点，每点一句话。",
        agent=agent,
    )

    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True,
        step_callback=adapter.make_step_callback(),
        task_callback=adapter.make_task_callback(),
    )

    # === 6. 执行 ===
    print(f"Starting crew...\n")
    try:
        result = crew.kickoff()
        print(f"\n{'='*60}")
        print(f"Result:\n{result}")
    except GuardrailDeny as e:
        print(f"\n{'='*60}")
        print(f"Guardrail triggered: {e}")

    adapter.cleanup()

    # === 7. 度量 ===
    print(f"\n{'='*60}")
    print("Security Metrics:")
    for name, strategy in security_strategies.items():
        metrics = strategy.get_metrics()
        print(f"\n  [{name}]")
        for k, v in metrics.items():
            print(f"    {k}: {v}")

    print(f"\n{'='*60}")
    print("Reliability Metrics:")
    for name, strategy in reliability_strategies.items():
        metrics = strategy.get_metrics()
        print(f"\n  [{name}]")
        for k, v in metrics.items():
            print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
