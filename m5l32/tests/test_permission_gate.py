"""T1-T6: PermissionGate 单元测试。"""

import pytest
from pathlib import Path

from hook_framework.registry import EventType, GuardrailDeny, HookContext

from shared_hooks.permission_gate import PermissionGate, PermissionLevel


def _tool_ctx(tool_name: str = "knowledge_search"):
    return HookContext(
        event_type=EventType.BEFORE_TOOL_CALL,
        tool_name=tool_name,
        session_id="test",
    )


def _gate_from_dict(tools: dict, default: str = "ask") -> PermissionGate:
    gate = PermissionGate(default=default)
    for tool_name, level in tools.items():
        gate._tool_permissions[tool_name] = PermissionLevel(level)
    return gate


# T1: DENY 工具被拦截
def test_deny_tool_blocked():
    gate = _gate_from_dict({"shell_executor": "deny"})
    with pytest.raises(GuardrailDeny, match="Permission denied"):
        gate.before_tool_handler(_tool_ctx("shell_executor"))


# T2: ALLOW 工具放行
def test_allow_tool_passes():
    gate = _gate_from_dict({"knowledge_search": "allow"})
    gate.before_tool_handler(_tool_ctx("knowledge_search"))


# T3: ASK 工具记录警告但放行
def test_ask_tool_passes_with_record():
    gate = _gate_from_dict({"file_reader": "ask"})
    gate.before_tool_handler(_tool_ctx("file_reader"))
    assert len(gate._decisions) == 1
    assert gate._decisions[0]["permission"] == "ask"


# T4: 未列出工具使用默认策略
def test_unlisted_tool_uses_default():
    gate = _gate_from_dict({"shell_executor": "deny"}, default="ask")
    gate.before_tool_handler(_tool_ctx("new_tool"))
    assert gate._decisions[0]["permission"] == "ask"
    assert gate._decisions[0]["policy_source"] == "default"


# T5: Default-Deny 模式
def test_default_deny_blocks_unlisted():
    gate = _gate_from_dict({}, default="deny")
    with pytest.raises(GuardrailDeny, match="Permission denied"):
        gate.before_tool_handler(_tool_ctx("any_tool"))


# T6: metrics 正确
def test_metrics_accuracy():
    gate = _gate_from_dict({
        "search": "allow",
        "calc": "allow",
        "reader": "allow",
        "shell": "deny",
    })
    gate.before_tool_handler(_tool_ctx("search"))
    gate.before_tool_handler(_tool_ctx("calc"))
    gate.before_tool_handler(_tool_ctx("reader"))
    with pytest.raises(GuardrailDeny):
        gate.before_tool_handler(_tool_ctx("shell"))

    m = gate.get_metrics()
    assert m["total_decisions"] == 4
    assert m["allow_count"] == 3
    assert m["deny_count"] == 1
    assert m["denied_tools"] == ["shell"]


# T_extra: 从 YAML 文件加载策略
def test_load_from_yaml(tmp_path):
    yaml_content = """
permissions:
  default: ask
  tools:
    knowledge_search: allow
    shell_executor: deny
"""
    policy = tmp_path / "security.yaml"
    policy.write_text(yaml_content)

    gate = PermissionGate(policy_path=policy)
    gate.before_tool_handler(_tool_ctx("knowledge_search"))
    with pytest.raises(GuardrailDeny):
        gate.before_tool_handler(_tool_ctx("shell_executor"))
