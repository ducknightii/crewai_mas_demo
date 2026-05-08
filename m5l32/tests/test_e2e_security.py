"""T23-T24: 安全策略端到端集成测试——YAML strategies 加载。"""

import os
from pathlib import Path

import pytest

from hook_framework.registry import EventType, GuardrailDeny, HookContext, HookRegistry
from hook_framework.loader import HookLoader
from shared_hooks.permission_gate import PermissionLevel

_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv("SECURITY_POLICY_PATH", raising=False)
    monkeypatch.delenv("SECURITY_AUDIT_FILE", raising=False)
    monkeypatch.delenv("COST_GUARD_BUDGET", raising=False)


def _load_all(tmp_path, monkeypatch, policy_yaml=None, budget=100.0):
    audit_file = tmp_path / "audit.jsonl"
    monkeypatch.setenv("SECURITY_AUDIT_FILE", str(audit_file))

    policy_path = tmp_path / "security.yaml"
    if policy_yaml:
        policy_path.write_text(policy_yaml)
    else:
        policy_path.write_text(
            "permissions:\n"
            "  default: ask\n"
            "  tools:\n"
            "    knowledge_search: allow\n"
            "    shell_executor: deny\n"
        )
    monkeypatch.setenv("SECURITY_POLICY_PATH", str(policy_path))

    if budget != 1.0:
        monkeypatch.setenv("COST_GUARD_BUDGET", str(budget))

    registry = HookRegistry()
    loader = HookLoader(registry)
    loader.load_from_directory(_DIR / "shared_hooks", layer_name="global")
    strategies = loader.strategies
    return registry, strategies, audit_file


# T23: 权限拦截端到端
@pytest.mark.integration
def test_permission_deny_e2e(tmp_path, monkeypatch):
    registry, strategies, audit_file = _load_all(tmp_path, monkeypatch)

    allow_ctx = HookContext(
        event_type=EventType.BEFORE_TOOL_CALL,
        tool_name="knowledge_search",
        tool_input={"query": "AI安全"},
        session_id="test",
    )
    registry.dispatch_gate(EventType.BEFORE_TOOL_CALL, allow_ctx)

    deny_ctx = HookContext(
        event_type=EventType.BEFORE_TOOL_CALL,
        tool_name="shell_executor",
        tool_input={"query": "ls -la"},
        session_id="test",
    )
    with pytest.raises(GuardrailDeny, match="Permission denied"):
        registry.dispatch_gate(EventType.BEFORE_TOOL_CALL, deny_ctx)

    m = strategies["permission_gate"].get_metrics()
    assert m["deny_count"] == 1
    assert m["allow_count"] == 1
    assert "shell_executor" in m["denied_tools"]

    audit_lines = audit_file.read_text().strip().split("\n")
    assert len(audit_lines) >= 1


# T24: 安全+可靠性完整链路
@pytest.mark.integration
def test_security_plus_reliability_e2e(tmp_path, monkeypatch):
    registry, strategies, audit_file = _load_all(tmp_path, monkeypatch, budget=100.0)

    tool_ctx = HookContext(
        event_type=EventType.BEFORE_TOOL_CALL,
        tool_name="knowledge_search",
        tool_input={"query": "AI安全"},
        session_id="test",
    )
    registry.dispatch_gate(EventType.BEFORE_TOOL_CALL, tool_ctx)

    turn_ctx = HookContext(
        event_type=EventType.AFTER_TURN,
        input_tokens=1000,
        output_tokens=500,
        turn_number=1,
        session_id="test",
        metadata={"output": "result"},
    )
    registry.dispatch_gate(EventType.AFTER_TURN, turn_ctx)

    end_ctx = HookContext(
        event_type=EventType.SESSION_END,
        session_id="test",
    )
    registry.dispatch(EventType.SESSION_END, end_ctx)

    perm_m = strategies["permission_gate"].get_metrics()
    assert perm_m["deny_count"] == 0
    assert perm_m["allow_count"] == 1

    assert strategies["sandbox_guard"].get_metrics()["total_violations"] == 0

    cost_m = strategies["cost_guard"].get_metrics()
    assert cost_m["total_input_tokens"] == 1000
    assert cost_m["estimated_cost_usd"] > 0

    audit_lines = audit_file.read_text().strip().split("\n")
    assert any("session_summary" in line for line in audit_lines)
