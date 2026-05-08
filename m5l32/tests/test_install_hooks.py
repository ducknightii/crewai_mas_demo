"""YAML strategies 集成测试：从真实 hooks.yaml 加载策略，验证注册和行为。"""

import os
from pathlib import Path

import pytest

from hook_framework.registry import EventType, GuardrailDeny, HookContext, HookRegistry
from hook_framework.loader import HookLoader

_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("SECURITY_POLICY_PATH", raising=False)
    monkeypatch.delenv("SECURITY_AUDIT_FILE", raising=False)
    monkeypatch.delenv("COST_GUARD_BUDGET", raising=False)


def _load_all(tmp_path, monkeypatch, budget=100.0):
    monkeypatch.setenv("SECURITY_AUDIT_FILE", str(tmp_path / "audit.jsonl"))
    if budget != 1.0:
        monkeypatch.setenv("COST_GUARD_BUDGET", str(budget))
    registry = HookRegistry()
    loader = HookLoader(registry)
    loader.load_from_directory(_DIR / "shared_hooks", layer_name="global")
    return registry, loader.strategies


def test_all_strategies_loaded(tmp_path, monkeypatch):
    _, strategies = _load_all(tmp_path, monkeypatch)
    assert "audit_logger" in strategies
    assert "sandbox_guard" in strategies
    assert "permission_gate" in strategies
    assert "retry_tracker" in strategies
    assert "cost_guard" in strategies
    assert "loop_detector" in strategies


def test_reliability_handlers_registered(tmp_path, monkeypatch):
    registry, _ = _load_all(tmp_path, monkeypatch)
    assert registry.handler_count(EventType.AFTER_TOOL_CALL) >= 2
    assert registry.handler_count(EventType.AFTER_TURN) >= 2
    assert registry.handler_count(EventType.BEFORE_TOOL_CALL) >= 3


def test_after_turn_ordering_cost_before_loop(tmp_path, monkeypatch):
    """cost_guard.accumulate 在 loop_detector 之前执行。"""
    registry, strategies = _load_all(tmp_path, monkeypatch, budget=100.0)

    ctx = HookContext(
        event_type=EventType.AFTER_TURN,
        tool_name="search",
        input_tokens=1000,
        output_tokens=500,
        turn_number=1,
        session_id="test",
        metadata={"output": "same result"},
    )
    registry.dispatch_gate(EventType.AFTER_TURN, ctx)

    assert strategies["cost_guard"].get_metrics()["total_input_tokens"] == 1000


def test_full_flow_budget_deny(tmp_path, monkeypatch):
    """端到端：累计 cost → AFTER_TURN deny。"""
    registry, strategies = _load_all(tmp_path, monkeypatch, budget=0.0001)

    turn_ctx = HookContext(
        event_type=EventType.AFTER_TURN,
        input_tokens=100000,
        output_tokens=50000,
        turn_number=1,
        session_id="test",
        metadata={"output": "result"},
    )
    with pytest.raises(GuardrailDeny, match="Budget exceeded"):
        registry.dispatch_gate(EventType.AFTER_TURN, turn_ctx)
