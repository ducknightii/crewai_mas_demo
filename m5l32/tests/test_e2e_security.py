"""T23-T24: 安全策略端到端集成测试。"""

import pytest

from hook_framework.registry import EventType, GuardrailDeny, HookContext, HookRegistry

from reliability import install_reliability_hooks
from security import install_security_hooks
from security.permission_gate import PermissionLevel


# T23: 权限拦截端到端
@pytest.mark.integration
def test_permission_deny_e2e(tmp_path):
    yaml_content = """
permissions:
  default: ask
  tools:
    knowledge_search: allow
    shell_executor: deny
"""
    policy = tmp_path / "security.yaml"
    policy.write_text(yaml_content)

    r = HookRegistry()
    sec = install_security_hooks(r, config={
        "policy_path": str(policy),
        "audit_file": str(tmp_path / "audit.jsonl"),
    })
    install_reliability_hooks(r, config={"budget_usd": 100.0})

    allow_ctx = HookContext(
        event_type=EventType.BEFORE_TOOL_CALL,
        tool_name="knowledge_search",
        tool_input={"query": "AI安全"},
        session_id="test",
    )
    r.dispatch_gate(EventType.BEFORE_TOOL_CALL, allow_ctx)

    deny_ctx = HookContext(
        event_type=EventType.BEFORE_TOOL_CALL,
        tool_name="shell_executor",
        tool_input={"query": "ls -la"},
        session_id="test",
    )
    with pytest.raises(GuardrailDeny, match="Permission denied"):
        r.dispatch_gate(EventType.BEFORE_TOOL_CALL, deny_ctx)

    m = sec["permission"].get_metrics()
    assert m["deny_count"] == 1
    assert m["allow_count"] == 1
    assert "shell_executor" in m["denied_tools"]

    audit_lines = (tmp_path / "audit.jsonl").read_text().strip().split("\n")
    assert len(audit_lines) >= 1


# T24: 安全+可靠性完整链路
@pytest.mark.integration
def test_security_plus_reliability_e2e(tmp_path):
    yaml_content = """
permissions:
  default: ask
  tools:
    knowledge_search: allow
"""
    policy = tmp_path / "security.yaml"
    policy.write_text(yaml_content)

    r = HookRegistry()
    sec = install_security_hooks(r, config={
        "policy_path": str(policy),
        "audit_file": str(tmp_path / "audit.jsonl"),
    })
    rel = install_reliability_hooks(r, config={"budget_usd": 100.0})

    tool_ctx = HookContext(
        event_type=EventType.BEFORE_TOOL_CALL,
        tool_name="knowledge_search",
        tool_input={"query": "AI安全"},
        session_id="test",
    )
    r.dispatch_gate(EventType.BEFORE_TOOL_CALL, tool_ctx)

    turn_ctx = HookContext(
        event_type=EventType.AFTER_TURN,
        input_tokens=1000,
        output_tokens=500,
        turn_number=1,
        session_id="test",
        metadata={"output": "result"},
    )
    r.dispatch_gate(EventType.AFTER_TURN, turn_ctx)

    end_ctx = HookContext(
        event_type=EventType.SESSION_END,
        session_id="test",
    )
    r.dispatch(EventType.SESSION_END, end_ctx)

    sec_m = sec["permission"].get_metrics()
    assert sec_m["deny_count"] == 0
    assert sec_m["allow_count"] == 1

    assert sec["sandbox"].get_metrics()["total_violations"] == 0

    rel_m = rel["cost"].get_metrics()
    assert rel_m["total_input_tokens"] == 1000
    assert rel_m["estimated_cost_usd"] > 0

    audit_lines = (tmp_path / "audit.jsonl").read_text().strip().split("\n")
    assert any("session_summary" in line for line in audit_lines)
