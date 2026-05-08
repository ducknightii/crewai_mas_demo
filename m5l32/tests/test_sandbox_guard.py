"""T7-T12: SandboxGuard 单元测试。"""

import pytest

from hook_framework.registry import EventType, GuardrailDeny, HookContext

from shared_hooks.sandbox_guard import SandboxGuard


def _tool_ctx(tool_input: str, tool_name: str = "knowledge_search"):
    return HookContext(
        event_type=EventType.BEFORE_TOOL_CALL,
        tool_name=tool_name,
        tool_input={"query": tool_input},
        session_id="test",
    )


# T7: 路径遍历被阻断
def test_path_traversal_blocked():
    guard = SandboxGuard()
    with pytest.raises(GuardrailDeny, match="Path traversal"):
        guard.before_tool_handler(_tool_ctx("../../etc/passwd"))


# T8: 正常路径放行
def test_normal_path_passes():
    guard = SandboxGuard()
    guard.before_tool_handler(_tool_ctx("./data/report.txt"))


# T9: 危险命令被阻断
def test_dangerous_command_blocked():
    guard = SandboxGuard()
    with pytest.raises(GuardrailDeny, match="Dangerous command"):
        guard.before_tool_handler(_tool_ctx("rm -rf /"))


# T10: Shell 注入字符被阻断
def test_shell_injection_blocked():
    guard = SandboxGuard()
    with pytest.raises(GuardrailDeny, match="Shell injection"):
        guard.before_tool_handler(_tool_ctx("query; cat /etc/passwd"))


# T11: 环境变量引用只警告不阻断
def test_env_var_warns_not_blocks():
    guard = SandboxGuard()
    # $ 不再由 _SHELL_INJECTION 拦截，由 _ENV_VAR_REF 单独处理（警告不阻断）
    guard.before_tool_handler(_tool_ctx("path is $HOME/data"))
    assert guard.get_metrics()["total_violations"] == 0


# T12: 空输入安全通过
def test_empty_input_passes():
    guard = SandboxGuard()
    ctx = HookContext(
        event_type=EventType.BEFORE_TOOL_CALL,
        tool_name="search",
        tool_input={},
        session_id="test",
    )
    guard.before_tool_handler(ctx)


# T_extra: 自然语言括号不误报
def test_parentheses_in_natural_language_passes():
    guard = SandboxGuard()
    guard.before_tool_handler(_tool_ctx("search (AI agent) security"))
    assert guard.get_metrics()["total_violations"] == 0


# T_extra: metrics 累计
def test_violation_metrics():
    guard = SandboxGuard()
    with pytest.raises(GuardrailDeny):
        guard.before_tool_handler(_tool_ctx("../../secret"))
    with pytest.raises(GuardrailDeny):
        guard.before_tool_handler(_tool_ctx("rm -rf /tmp"))

    m = guard.get_metrics()
    assert m["total_violations"] == 2
    assert "path_traversal" in m["violations_by_type"]
    assert "dangerous_command" in m["violations_by_type"]


# T_extra: sudo 被检测
def test_sudo_blocked():
    guard = SandboxGuard()
    with pytest.raises(GuardrailDeny, match="Dangerous command"):
        guard.before_tool_handler(_tool_ctx("sudo apt install"))


# T_extra: chmod 777 被检测
def test_chmod_777_blocked():
    guard = SandboxGuard()
    with pytest.raises(GuardrailDeny, match="Dangerous command"):
        guard.before_tool_handler(_tool_ctx("chmod 777 /tmp/script.sh"))


# T_extra: 管道符被检测
def test_pipe_blocked():
    guard = SandboxGuard()
    with pytest.raises(GuardrailDeny, match="Shell injection"):
        guard.before_tool_handler(_tool_ctx("cat file | grep secret"))


# T_extra: 反引号被检测
def test_backtick_blocked():
    guard = SandboxGuard()
    with pytest.raises(GuardrailDeny, match="Shell injection"):
        guard.before_tool_handler(_tool_ctx("echo `whoami`"))
