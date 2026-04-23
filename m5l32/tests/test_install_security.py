"""T20-T22: install_security_hooks 集成测试。"""

import pytest

from hook_framework.registry import EventType, GuardrailDeny, HookContext, HookRegistry

from reliability import install_reliability_hooks
from security import install_security_hooks


# T20: install 注册所有 handler
def test_install_registers_all_handlers():
    r = HookRegistry()
    strategies = install_security_hooks(r)
    assert r.handler_count(EventType.BEFORE_TOOL_CALL) == 2
    assert r.handler_count(EventType.SESSION_END) == 1
    assert "permission" in strategies
    assert "sandbox" in strategies
    assert "audit" in strategies


# T21: 安全 handler 在可靠性 handler 之前
def test_security_before_reliability_ordering():
    r = HookRegistry()
    install_security_hooks(r)
    install_reliability_hooks(r)

    summary = r.summary()
    before_tool = summary["before_tool_call"]
    assert before_tool[0] == "sandbox_guard"
    assert before_tool[1] == "permission_gate"
    assert before_tool[2] == "cost_guard.gate"


# T22: sandbox 拦截后 permission 和 cost 不执行
def test_sandbox_deny_stops_chain():
    r = HookRegistry()
    sec = install_security_hooks(r)
    rel = install_reliability_hooks(r)

    ctx = HookContext(
        event_type=EventType.BEFORE_TOOL_CALL,
        tool_name="search",
        tool_input={"query": "../../etc/passwd"},
        session_id="test",
    )

    with pytest.raises(GuardrailDeny, match="Path traversal"):
        r.dispatch_gate(EventType.BEFORE_TOOL_CALL, ctx)

    assert sec["permission"].get_metrics()["total_decisions"] == 0
    assert rel["cost"].get_metrics()["deny_count"] == 0


# T_extra: permission deny 后 cost 不执行
def test_permission_deny_stops_cost():
    r = HookRegistry()
    sec = install_security_hooks(r, config={"default_permission": "deny"})
    rel = install_reliability_hooks(r)

    ctx = HookContext(
        event_type=EventType.BEFORE_TOOL_CALL,
        tool_name="any_tool",
        tool_input={},
        session_id="test",
    )

    with pytest.raises(GuardrailDeny, match="Permission denied"):
        r.dispatch_gate(EventType.BEFORE_TOOL_CALL, ctx)

    assert rel["cost"].get_metrics()["deny_count"] == 0
