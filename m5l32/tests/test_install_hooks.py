"""install_reliability_hooks 集成测试：注册顺序 + 端到端流程。"""

import pytest

from hook_framework.registry import EventType, GuardrailDeny, HookContext, HookRegistry

from reliability import install_reliability_hooks


def test_install_registers_all_handlers():
    r = HookRegistry()
    strategies = install_reliability_hooks(r)
    assert r.handler_count(EventType.AFTER_TOOL_CALL) >= 1
    assert r.handler_count(EventType.AFTER_TURN) >= 2
    assert r.handler_count(EventType.BEFORE_TOOL_CALL) >= 1
    assert "retry" in strategies
    assert "loop" in strategies
    assert "cost" in strategies


def test_after_turn_ordering_cost_before_loop():
    """cost_guard.accumulate 必须在 loop_detector 之前执行。
    验证：即使 loop_detector deny，cost 已累加。
    """
    r = HookRegistry()
    strategies = install_reliability_hooks(r, config={
        "loop_threshold": 2,
        "budget_usd": 100.0,
    })

    ctx = HookContext(
        event_type=EventType.AFTER_TURN,
        tool_name="search",
        input_tokens=1000,
        output_tokens=500,
        turn_number=1,
        session_id="test",
        metadata={"output": "same result"},
    )
    r.dispatch_gate(EventType.AFTER_TURN, ctx)

    assert strategies["cost"].get_metrics()["total_input_tokens"] == 1000

    with pytest.raises(GuardrailDeny):
        r.dispatch_gate(EventType.AFTER_TURN, ctx)

    assert strategies["cost"].get_metrics()["total_input_tokens"] == 2000


def test_full_flow_budget_deny():
    """端到端：累计 cost → before_tool_call deny。"""
    r = HookRegistry()
    strategies = install_reliability_hooks(r, config={
        "budget_usd": 0.0001,
    })

    turn_ctx = HookContext(
        event_type=EventType.AFTER_TURN,
        input_tokens=100000,
        output_tokens=50000,
        turn_number=1,
        session_id="test",
        metadata={"output": "result"},
    )
    with pytest.raises(GuardrailDeny, match="Budget exceeded"):
        r.dispatch_gate(EventType.AFTER_TURN, turn_ctx)
