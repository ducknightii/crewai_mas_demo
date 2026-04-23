"""T1-T5: dispatch_gate 单元测试。"""

import pytest
from unittest.mock import MagicMock

from hook_framework.registry import EventType, GuardrailDeny, HookContext, HookRegistry


def _ctx(event_type=EventType.BEFORE_TOOL_CALL):
    return HookContext(event_type=event_type, session_id="test")


# T1: dispatch_gate 正常分发（无 deny）
def test_dispatch_gate_normal():
    r = HookRegistry()
    handler = MagicMock()
    r.register(EventType.BEFORE_TOOL_CALL, handler)
    r.dispatch_gate(EventType.BEFORE_TOOL_CALL, _ctx())
    handler.assert_called_once()


# T2: dispatch_gate 传播 GuardrailDeny
def test_dispatch_gate_propagates_guardrail_deny():
    r = HookRegistry()

    def deny_handler(ctx):
        raise GuardrailDeny("budget exceeded")

    r.register(EventType.BEFORE_TOOL_CALL, deny_handler)
    with pytest.raises(GuardrailDeny, match="budget exceeded"):
        r.dispatch_gate(EventType.BEFORE_TOOL_CALL, _ctx())


# T3: dispatch_gate 吞掉非 GuardrailDeny 异常，后续 handler 仍执行
def test_dispatch_gate_swallows_non_deny_and_continues():
    r = HookRegistry()
    good_before = MagicMock()
    good_after = MagicMock()

    def bad_handler(ctx):
        raise RuntimeError("boom")

    r.register(EventType.BEFORE_TOOL_CALL, good_before)
    r.register(EventType.BEFORE_TOOL_CALL, bad_handler)
    r.register(EventType.BEFORE_TOOL_CALL, good_after)

    r.dispatch_gate(EventType.BEFORE_TOOL_CALL, _ctx())
    good_before.assert_called_once()
    good_after.assert_called_once()


# T4: dispatch_gate 只传播第一个 deny，后续 handler 不执行
def test_dispatch_gate_stops_at_first_deny():
    r = HookRegistry()
    before = MagicMock()
    after = MagicMock()

    def deny_handler(ctx):
        raise GuardrailDeny("first deny")

    r.register(EventType.BEFORE_TOOL_CALL, before)
    r.register(EventType.BEFORE_TOOL_CALL, deny_handler)
    r.register(EventType.BEFORE_TOOL_CALL, after)

    with pytest.raises(GuardrailDeny, match="first deny"):
        r.dispatch_gate(EventType.BEFORE_TOOL_CALL, _ctx())

    before.assert_called_once()
    after.assert_not_called()


# T5: GuardrailDeny 携带 reason 属性
def test_guardrail_deny_has_reason():
    deny = GuardrailDeny("test reason")
    assert deny.reason == "test reason"
    assert str(deny) == "test reason"
