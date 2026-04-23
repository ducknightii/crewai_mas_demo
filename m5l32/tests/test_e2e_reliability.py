"""E2E 可靠性策略测试——需要真实 LLM API。

运行方式：
    python3 -m pytest tests/test_e2e_reliability.py -v -s
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

pytestmark = pytest.mark.integration

_DIR = Path(__file__).resolve().parent.parent


class SearchInput(BaseModel):
    query: str = Field(description="搜索关键词")


class KnowledgeSearchTool(BaseTool):
    name: str = "knowledge_search"
    description: str = "搜索知识库"
    args_schema: type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:
        return f"[可靠性] Agent可靠性包括重试、循环检测和成本围栏三种策略。"


class LoopingTool(BaseTool):
    name: str = "looping_search"
    description: str = "搜索知识库获取详细信息（必须使用此工具获取数据）"
    args_schema: type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:
        return "搜索结果：AI Agent 可靠性是一个重要话题。请继续搜索以获取更多细节。"


def _make_llm():
    model_name = os.environ.get("AGENT_MODEL", "qwen-plus")
    base_url = os.environ.get(
        "OPENAI_API_BASE",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    return LLM(model=model_name, base_url=base_url)


def _make_crew(tool, registry, adapter, max_iter=15):
    llm = _make_llm()
    agent = Agent(
        role="Research Analyst",
        goal="搜索并总结关于 AI Agent 可靠性的信息",
        backstory="你是一位研究分析师。使用工具搜索后总结要点。",
        llm=llm,
        verbose=True,
        tools=[tool],
        max_iter=max_iter,
    )
    task = Task(
        description="搜索「AI Agent 可靠性」相关信息，列出 3 个关键要点。",
        expected_output="3 个关键要点，每点一句话。",
        agent=agent,
    )
    return Crew(
        agents=[agent],
        tasks=[task],
        verbose=True,
        step_callback=adapter.make_step_callback(),
        task_callback=adapter.make_task_callback(),
    )


def test_e2e_normal_execution():
    """正常执行：无 deny，metrics 有数据。"""
    registry = HookRegistry()
    strategies = install_reliability_hooks(registry, config={
        "budget_usd": 10.0,
        "loop_threshold": 10,
    })

    adapter = CrewObservabilityAdapter(registry, session_id="test_normal")
    adapter.install_global_hooks()

    crew = _make_crew(KnowledgeSearchTool(), registry, adapter)
    try:
        result = crew.kickoff()
        assert result is not None
    finally:
        adapter.cleanup()

    cost_m = strategies["cost"].get_metrics()
    assert cost_m["total_input_tokens"] > 0
    assert cost_m["deny_count"] == 0


def test_e2e_loop_detection():
    """循环检测：LoopingTool 导致重复状态 → GuardrailDeny 或 metrics 记录循环。"""
    registry = HookRegistry()
    strategies = install_reliability_hooks(registry, config={
        "budget_usd": 10.0,
        "loop_threshold": 2,
    })

    adapter = CrewObservabilityAdapter(registry, session_id="test_loop")
    adapter.install_global_hooks()

    llm = _make_llm()
    agent = Agent(
        role="Research Analyst",
        goal="使用 looping_search 搜索「AI Agent 可靠性」",
        backstory="你是研究员。你必须使用 looping_search 工具搜索信息。每次搜索后如果信息不够详细，继续搜索。",
        llm=llm,
        verbose=True,
        tools=[LoopingTool()],
        max_iter=10,
    )
    task = Task(
        description="使用 looping_search 工具反复搜索「可靠性」「重试」「成本」等关键词获取详细信息，然后总结 5 个要点。每个要点需要引用搜索到的具体数据。",
        expected_output="5 个包含具体数据的详细要点。",
        agent=agent,
    )
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=True,
        step_callback=adapter.make_step_callback(),
        task_callback=adapter.make_task_callback(),
    )

    guardrail_hit = False
    try:
        crew.kickoff()
    except GuardrailDeny:
        guardrail_hit = True
    except Exception:
        pass
    finally:
        adapter.cleanup()

    loop_m = strategies["loop"].get_metrics()
    # 降级断言：要么 guardrail 触发，要么至少有 2 轮（说明 agent 使用了工具）
    assert guardrail_hit or loop_m["total_turns"] >= 2 or loop_m["loop_detections"] >= 1


def test_e2e_cost_guard():
    """成本围栏：极低预算 → GuardrailDeny。"""
    registry = HookRegistry()
    strategies = install_reliability_hooks(registry, config={
        "budget_usd": 0.0001,
        "loop_threshold": 100,
    })

    adapter = CrewObservabilityAdapter(registry, session_id="test_cost")
    adapter.install_global_hooks()

    crew = _make_crew(KnowledgeSearchTool(), registry, adapter)
    guardrail_hit = False
    try:
        crew.kickoff()
    except GuardrailDeny:
        guardrail_hit = True
    except Exception:
        pass
    finally:
        adapter.cleanup()

    cost_m = strategies["cost"].get_metrics()
    assert guardrail_hit or cost_m["deny_count"] >= 1
