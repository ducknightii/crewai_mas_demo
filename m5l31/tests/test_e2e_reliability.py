"""E2E reliability strategy tests -- require real LLM API.

Run:
    python3 -m pytest tests/test_e2e_reliability.py -v -s
"""

import os
import sys
from datetime import datetime, timezone
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

pytestmark = pytest.mark.integration

_DIR = Path(__file__).resolve().parent.parent


class SearchInput(BaseModel):
    query: str = Field(description="search keyword")


class KnowledgeSearchTool(BaseTool):
    name: str = "knowledge_search"
    description: str = "search knowledge base"
    args_schema: type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:
        return f"[reliability] Agent reliability includes retry, loop detection, and cost guard."


class LoopingTool(BaseTool):
    name: str = "looping_search"
    description: str = "search knowledge base for details (must use this tool)"
    args_schema: type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:
        return "search result: AI Agent reliability is important. Please search again for more details."


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
        goal="search and summarize information about AI Agent reliability",
        backstory="You are a research analyst. Search with tools and summarize key points.",
        llm=llm,
        verbose=True,
        tools=[tool],
        max_iter=max_iter,
    )
    task = Task(
        description="Search for AI Agent reliability info and list 3 key points.",
        expected_output="3 key points, one sentence each.",
        agent=agent,
    )
    return Crew(
        agents=[agent],
        tasks=[task],
        verbose=True,
        step_callback=adapter.make_step_callback(),
        task_callback=adapter.make_task_callback(),
    )


def _make_session_id(label: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"e2e-{label}-{ts}"


def _setup_full_hooks(label, env_overrides=None):
    """Load YAML hooks (observation) + strategies (reliability) via HookLoader.

    Returns (registry, adapter, strategies, session_id).
    """
    from crewai.hooks import clear_all_global_hooks
    clear_all_global_hooks()

    if env_overrides:
        for k, v in env_overrides.items():
            os.environ[k] = str(v)

    session_id = _make_session_id(label)

    registry = HookRegistry()
    loader = HookLoader(registry)
    loader.load_two_layers(
        global_dir=_DIR / "shared_hooks",
        workspace_dir=_DIR / "workspace" / "demo_agent",
    )
    strategies = loader.strategies

    adapter = CrewObservabilityAdapter(registry, session_id=session_id)
    adapter.install_global_hooks()
    return registry, adapter, strategies, session_id


def test_e2e_normal_execution():
    """Normal execution: no deny, metrics have data."""
    registry, adapter, strategies, sid = _setup_full_hooks("normal", {
        "COST_GUARD_BUDGET": "10.0",
    })

    crew = _make_crew(KnowledgeSearchTool(), registry, adapter)
    try:
        result = crew.kickoff()
        assert result is not None
    finally:
        adapter.cleanup()
        os.environ.pop("COST_GUARD_BUDGET", None)

    cost_m = strategies["cost_guard"].get_metrics()
    assert cost_m["total_input_tokens"] > 0
    assert cost_m["deny_count"] == 0


def test_e2e_loop_detection():
    """Loop detection: LoopingTool produces repeated state -> GuardrailDeny or metrics record loops."""
    registry, adapter, strategies, sid = _setup_full_hooks("loop", {
        "COST_GUARD_BUDGET": "10.0",
    })

    llm = _make_llm()
    agent = Agent(
        role="Research Analyst",
        goal="use looping_search to search AI Agent reliability",
        backstory="You are a researcher. You must use looping_search. If info is insufficient, keep searching.",
        llm=llm,
        verbose=True,
        tools=[LoopingTool()],
        max_iter=10,
    )
    task = Task(
        description="Use looping_search tool repeatedly to search reliability/retry/cost keywords for details, then summarize 5 key points with specific data.",
        expected_output="5 detailed key points with specific data.",
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
        os.environ.pop("COST_GUARD_BUDGET", None)

    loop_m = strategies["loop_detector"].get_metrics()
    assert guardrail_hit or loop_m["total_turns"] >= 2 or loop_m["loop_detections"] >= 1


def test_e2e_cost_guard():
    """Cost guard: very low budget -> GuardrailDeny."""
    registry, adapter, strategies, sid = _setup_full_hooks("cost", {
        "COST_GUARD_BUDGET": "0.0001",
    })

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
        os.environ.pop("COST_GUARD_BUDGET", None)

    cost_m = strategies["cost_guard"].get_metrics()
    assert guardrail_hit or cost_m["deny_count"] >= 1
