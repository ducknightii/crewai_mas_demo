"""T20-T22: YAML strategies 安全集成测试——验证 deps 注入和执行顺序。"""

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


def _load_all(tmp_path, monkeypatch, policy_yaml=None, default_permission=None):
    audit_file = tmp_path / "audit.jsonl"
    monkeypatch.setenv("SECURITY_AUDIT_FILE", str(audit_file))

    if policy_yaml:
        policy_path = tmp_path / "security.yaml"
        policy_path.write_text(policy_yaml)
        monkeypatch.setenv("SECURITY_POLICY_PATH", str(policy_path))

    registry = HookRegistry()
    loader = HookLoader(registry)
    loader.load_from_directory(_DIR / "shared_hooks", layer_name="global")
    return registry, loader.strategies, audit_file


# T20: 所有安全 handler 注册
def test_install_registers_all_handlers(tmp_path, monkeypatch):
    registry, strategies, _ = _load_all(tmp_path, monkeypatch)
    assert registry.handler_count(EventType.BEFORE_TOOL_CALL) >= 3
    assert registry.handler_count(EventType.SESSION_END) >= 1
    assert "permission_gate" in strategies
    assert "sandbox_guard" in strategies
    assert "audit_logger" in strategies


# T20b: deps 注入验证——sandbox 和 permission 共享同一个 audit 实例
def test_deps_audit_shared(tmp_path, monkeypatch):
    _, strategies, _ = _load_all(tmp_path, monkeypatch)
    audit = strategies["audit_logger"]
    assert strategies["sandbox_guard"]._audit is audit
    assert strategies["permission_gate"]._audit is audit


# T21: 安全 handler 在可靠性 handler 之前
def test_security_before_reliability_ordering(tmp_path, monkeypatch):
    registry, _, _ = _load_all(tmp_path, monkeypatch)
    summary = registry.summary()
    before_tool = summary["before_tool_call"]
    sandbox_idx = next(i for i, h in enumerate(before_tool) if "sandbox_guard" in h.lower())
    permission_idx = next(i for i, h in enumerate(before_tool) if "permission_gate" in h.lower())
    cost_idx = next(i for i, h in enumerate(before_tool) if "cost_guard" in h.lower())
    assert sandbox_idx < permission_idx < cost_idx


# T22: sandbox 拦截后 permission 和 cost 不执行
def test_sandbox_deny_stops_chain(tmp_path, monkeypatch):
    registry, strategies, _ = _load_all(tmp_path, monkeypatch)

    ctx = HookContext(
        event_type=EventType.BEFORE_TOOL_CALL,
        tool_name="search",
        tool_input={"query": "../../etc/passwd"},
        session_id="test",
    )

    with pytest.raises(GuardrailDeny, match="Path traversal"):
        registry.dispatch_gate(EventType.BEFORE_TOOL_CALL, ctx)

    assert strategies["permission_gate"].get_metrics()["total_decisions"] == 0
    assert strategies["cost_guard"].get_metrics()["deny_count"] == 0


# T_extra: permission deny 后 cost 不执行
def test_permission_deny_stops_cost(tmp_path, monkeypatch):
    policy_yaml = """\
permissions:
  default: deny
  tools: {}
"""
    registry, strategies, _ = _load_all(tmp_path, monkeypatch, policy_yaml=policy_yaml)

    ctx = HookContext(
        event_type=EventType.BEFORE_TOOL_CALL,
        tool_name="any_tool",
        tool_input={},
        session_id="test",
    )

    with pytest.raises(GuardrailDeny, match="Permission denied"):
        registry.dispatch_gate(EventType.BEFORE_TOOL_CALL, ctx)

    assert strategies["cost_guard"].get_metrics()["deny_count"] == 0
