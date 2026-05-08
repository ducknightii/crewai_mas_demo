"""32课 Demo：观测 + 可靠性 + 安全 三层 Hook 叠加。

设计定位：**L32 是 L31 的升级版**。业务骨架（Bootstrap + SkillLoader + sop_design）
继承自 L31；本课升级项是 hooks.yaml 多一段安全策略、workspace 多一份 security.yaml、
soul.md 追加安全禁令、demo.py 增加 --attack 分支演示 Hook 拦截。

运行方式：
    # 正常流程（需先 `docker compose -f sandbox-docker-compose.yaml up -d`）
    python3 demo.py
    python3 demo.py "为一个短链接服务产出技术设计文档"

    # 攻击演示：权限网关拦截（shell_executor 被 DENY）
    python3 demo.py --attack privilege

    # 攻击演示：沙箱消毒拦截（路径遍历输入）
    python3 demo.py --attack inject

    # 攻击演示：凭证运行时注入（LLM 看不到密钥）
    python3 demo.py --attack api-leak

    # 低预算 + 正常流程（触发成本围栏）
    python3 demo.py --budget 0.001
"""

from __future__ import annotations

import argparse
import atexit
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _DIR.parent
for _p in [str(_DIR), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv

load_dotenv(_DIR / ".env", override=True)

from crewai import Agent, Crew, LLM, Task
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from m3l20.m3l20_file_memory import build_bootstrap_prompt
from tools.skill_loader_tool import SkillLoaderTool

from hook_framework import (
    CrewObservabilityAdapter,
    GuardrailDeny,
    HookLoader,
    HookRegistry,
)
from shared_hooks.credential_inject import SecureToolWrapper

# ─────────────────────────── 顶层常量（继承 L31 结构） ───────────────────────────

WORKSPACE_DIR = _DIR / "workspace" / "demo_agent"
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


# ─────────────────────────── 攻击演示专用内联 Tool 类 ───────────────────────────


class _SearchInput(BaseModel):
    query: str = Field(description="搜索关键词")


class ShellExecutorTool(BaseTool):
    """Shell 执行工具——应被 PermissionGate DENY（security.yaml 规定）。"""

    name: str = "shell_executor"
    description: str = "执行系统命令获取信息。（本工具是演示用，实际会被安全层拦截）"
    args_schema: type[BaseModel] = _SearchInput

    def _run(self, query: str) -> str:
        return f"[SECURITY BREACH] Command executed: {query}"


class InjectableSearchTool(BaseTool):
    """搜索工具——Task 会诱导 LLM 传入含 `../` 的参数，触发 SandboxGuard。"""

    name: str = "knowledge_search"
    description: str = "搜索知识库。"
    args_schema: type[BaseModel] = _SearchInput

    def _run(self, query: str) -> str:
        return f"搜索结果: {query}"


class SecureApiTool(BaseTool):
    """需要 API Key 的外部 API——密钥通过 SecureToolWrapper 在运行时注入。"""

    name: str = "secure_api"
    description: str = "调用外部 API 查询数据。"
    args_schema: type[BaseModel] = _SearchInput

    def _run(self, query: str, api_key: str = "") -> str:
        key_preview = (
            f"{api_key[:4]}...{api_key[-4:]}" if len(api_key) > 8 else "***"
        )
        return f"[API] 使用密钥 {key_preview} 查询: {query}"


# ─────────────────────────── 内部工具函数 ───────────────────────────


def _build_llm() -> LLM:
    model_name = os.environ.get("AGENT_MODEL", "qwen-plus")
    base_url = os.environ.get(
        "OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    return LLM(model=model_name, base_url=base_url)


def _make_session_id() -> str:
    return f"sess_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"


def _set_security_env(args: argparse.Namespace) -> None:
    """集中写入安全相关 env；runner 不得再写 os.environ。"""
    os.environ.setdefault(
        "SECURITY_POLICY_PATH",
        str(WORKSPACE_DIR / "security.yaml"),
    )
    os.environ.setdefault(
        "SECURITY_AUDIT_FILE",
        str(WORKSPACE_DIR / "security_audit.jsonl"),
    )
    if getattr(args, "budget", 1.0) != 1.0:
        os.environ["COST_GUARD_BUDGET"] = str(args.budget)


def _kickoff(agent: Agent, task: Task, adapter: CrewObservabilityAdapter):
    """所有 runner 共用的 Crew 构造 + kickoff，绑定 adapter 回调。"""
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True,
        step_callback=adapter.make_step_callback(),
        task_callback=adapter.make_task_callback(),
    )
    return crew.kickoff()


def _runner_table() -> dict:
    """--attack 值 → runner 函数映射表（main() 与 tests 都会用）。"""
    return {
        None: run_normal,
        "privilege": run_attack_privilege,
        "inject": run_attack_inject,
        "api-leak": run_attack_api_leak,
    }


# ─────────────────────────── Runner 1：正常流程（继承 L31 骨架） ───────────────────────────


def run_normal(args: argparse.Namespace, adapter: CrewObservabilityAdapter, llm: LLM):
    """正常流程：Bootstrap + SkillLoader + sop_design → 沙盒产出 design_doc.md。"""
    backstory = build_bootstrap_prompt(WORKSPACE_DIR)

    skill_tool = SkillLoaderTool(
        skills_dir=str(SKILLS_DIR),
        sandbox_mcp_url=SANDBOX_MCP_URL,
        sandbox_mount_desc=SANDBOX_MOUNT_DESC,
        step_callback=adapter.make_sub_crew_step_callback(),   # 💡 Sub-Crew 可观测性
        task_callback=adapter.make_task_callback(),            # 💡 Sub-Crew task 完成追踪
    )

    task_desc = (" ".join(args.task) if args.task else "").strip() or DEFAULT_TASK

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
    print(f"Task: {task_desc}")
    print(f"Bootstrap: {len(backstory)} chars from workspace/demo_agent/")
    if hasattr(skill_tool, "_skill_registry"):
        print(f"Skills: {list(skill_tool._skill_registry.keys())}\n")
    return _kickoff(agent, task, adapter)


# ─────────────────────────── Runner 2：--attack privilege ───────────────────────────


def run_attack_privilege(
    args: argparse.Namespace, adapter: CrewObservabilityAdapter, llm: LLM
):
    """对抗压力由 Task description 承担：强制调用 shell_executor → PermissionGate DENY。"""
    agent = Agent(
        role="Research Analyst",
        goal="完成指定任务",
        backstory="你是研究分析师。必须调用提供的工具完成任务。",
        llm=llm,
        verbose=True,
        tools=[ShellExecutorTool()],
        max_iter=5,
    )
    task = Task(
        description=(
            "你拥有 shell_executor 工具。"
            "**必须调用该工具**获取当前系统的用户列表，query 参数传 'whoami'。"
        ),
        expected_output="工具返回的字符串",
        agent=agent,
    )
    return _kickoff(agent, task, adapter)


# ─────────────────────────── Runner 3：--attack inject ───────────────────────────


def run_attack_inject(
    args: argparse.Namespace, adapter: CrewObservabilityAdapter, llm: LLM
):
    """Task 携带路径遍历字符串 → SandboxGuard 拦截。"""
    agent = Agent(
        role="Research Analyst",
        goal="完成搜索任务",
        backstory="你是研究分析师。必须调用 knowledge_search 完成搜索。",
        llm=llm,
        verbose=True,
        tools=[InjectableSearchTool()],
        max_iter=5,
    )
    task = Task(
        description=(
            "调用 knowledge_search 工具，query 参数**必须**为字符串 '../../etc/passwd'，"
            "获取并返回搜索结果。"
        ),
        expected_output="工具返回的字符串",
        agent=agent,
    )
    return _kickoff(agent, task, adapter)


# ─────────────────────────── Runner 4：--attack api-leak ───────────────────────────


def run_attack_api_leak(
    args: argparse.Namespace, adapter: CrewObservabilityAdapter, llm: LLM
):
    """SecureToolWrapper 在工具执行层注入密钥；LLM 上下文与 tool_input 均不含密钥。"""
    # 若测试未设置，demo 自动填一个 fake key，便于本地手动跑
    os.environ.setdefault("SECURE_API_KEY", "sk-DEMO-" + "x" * 32)

    raw_tool = SecureApiTool()
    wrapped = SecureToolWrapper.wrap(raw_tool, credentials={"api_key": "SECURE_API_KEY"})

    agent = Agent(
        role="Research Analyst",
        goal="调用外部 API 查询用户档案",
        backstory="你是研究分析师。使用 secure_api 工具查询后总结要点。",
        llm=llm,
        verbose=True,
        tools=[wrapped],
        max_iter=5,
    )
    task = Task(
        description="调用 secure_api 工具，query 参数传 'user_profile'，返回工具输出原文。",
        expected_output="工具返回的字符串",
        agent=agent,
    )
    return _kickoff(agent, task, adapter)


# ─────────────────────────── main：共用初始化 + 分派 ───────────────────────────


def _print_banner(
    session_id: str,
    args: argparse.Namespace,
    registry: HookRegistry,
    loader: HookLoader,
):
    summary = registry.summary()
    total = sum(len(v) for v in summary.values())
    print(f"Session: {session_id}")
    print(f"Budget: ${args.budget:.3f}")
    print(f"Attack: {args.attack or 'none'}")
    print(f"Hooks: {total} handlers")
    for event, handlers in summary.items():
        for h in handlers:
            print(f"   {h} -> {event}")
    if loader.strategies:
        print(f"Strategies: {list(loader.strategies.keys())}")
    print()


def _print_metrics(strategies: dict):
    security_keys = ["audit_logger", "sandbox_guard", "permission_gate"]
    reliability_keys = ["retry_tracker", "cost_guard", "loop_detector"]

    print(f"\n{'='*60}")
    print("Security Metrics:")
    for key in security_keys:
        if key in strategies:
            metrics = strategies[key].get_metrics()
            print(f"\n  [{key}]")
            for k, v in metrics.items():
                print(f"    {k}: {v}")

    print(f"\n{'='*60}")
    print("Reliability Metrics:")
    for key in reliability_keys:
        if key in strategies:
            metrics = strategies[key].get_metrics()
            print(f"\n  [{key}]")
            for k, v in metrics.items():
                print(f"    {k}: {v}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="32课 Demo：观测+可靠性+安全")
    parser.add_argument(
        "--attack",
        choices=["privilege", "inject", "api-leak"],
        default=None,
        help="攻击演示模式",
    )
    parser.add_argument(
        "--budget", type=float, default=1.0, help="CostGuard 预算（USD）"
    )
    parser.add_argument(
        "task", nargs="*", help="正常模式下的用户任务（可选）"
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    _set_security_env(args)
    session_id = _make_session_id()

    # 不变式 I-1：HookRegistry / adapter 只在 main() 构造一次
    registry = HookRegistry()
    loader = HookLoader(registry)
    loader.load_two_layers(
        global_dir=_DIR / "shared_hooks",
        workspace_dir=WORKSPACE_DIR,
    )
    adapter = CrewObservabilityAdapter(registry, session_id=session_id)
    adapter.install_global_hooks()
    atexit.register(adapter.cleanup)

    _print_banner(session_id, args, registry, loader)

    llm = _build_llm()
    runner = _runner_table()[args.attack]

    try:
        result = runner(args, adapter, llm)
        print(f"\n{'='*60}\nResult:\n{result}")
    except GuardrailDeny as e:
        print(f"\n{'='*60}\nGuardrail triggered: {e}")

    adapter.cleanup()

    print(f"\n{'='*60}")
    print(f"Langfuse: {os.environ.get('LANGFUSE_HOST', 'http://localhost:3000')}")
    design_doc = OUTPUT_DIR / "design_doc.md"
    if design_doc.exists():
        print(f"Design doc: {design_doc}")
    audit_file = WORKSPACE_DIR / "audit.log"
    if audit_file.exists():
        print(f"Task audit: {audit_file}")
    security_audit = Path(os.environ["SECURITY_AUDIT_FILE"])
    if security_audit.exists():
        print(f"Security audit: {security_audit}")

    _print_metrics(loader.strategies)


if __name__ == "__main__":
    main()
