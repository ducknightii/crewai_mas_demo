"""安全策略：权限网关 + 沙箱输入消毒 + 安全审计。

用法：
    from security import install_security_hooks
    strategies = install_security_hooks(registry, config={...})
"""

from pathlib import Path

from hook_framework.registry import EventType

from .audit_logger import SecurityAuditLogger
from .permission_gate import PermissionGate
from .sandbox_guard import SandboxGuard


def install_security_hooks(
    registry,
    config: dict | None = None,
) -> dict:
    """在 HookRegistry 上注册所有安全策略。

    Args:
        registry: HookRegistry 实例
        config: 可选配置 {
            "policy_path": Path | str (security.yaml 路径),
            "default_permission": str ("deny" | "ask" | "allow", default "ask"),
            "workspace_root": str (沙箱根目录),
            "audit_file": Path | str (审计日志路径),
        }

    Returns:
        dict with "permission", "sandbox", "audit" keys -> strategy instances
    """
    config = config or {}

    audit = SecurityAuditLogger(
        audit_file=Path(config["audit_file"]) if config.get("audit_file") else None,
    )

    policy_path = config.get("policy_path")
    permission = PermissionGate(
        policy_path=Path(policy_path) if policy_path else None,
        default=config.get("default_permission", "ask"),
        audit=audit,
    )

    sandbox = SandboxGuard(
        workspace_root=config.get("workspace_root", ""),
        audit=audit,
    )

    # 注册顺序（CRITICAL）：
    # 安全检查在可靠性检查之前注册。
    # install_security_hooks() 必须在 install_reliability_hooks() 之前调用。
    #
    # BEFORE_TOOL_CALL 执行顺序：
    #   1. sandbox_guard    （输入消毒——最先检查，脏数据不进后续流程）
    #   2. permission_gate  （权限检查——输入干净后检查权限）
    #   3. cost_guard.gate  （成本检查——有权限后检查预算，来自31课）

    registry.register(
        EventType.BEFORE_TOOL_CALL,
        sandbox.before_tool_handler,
        name="sandbox_guard",
    )
    registry.register(
        EventType.BEFORE_TOOL_CALL,
        permission.before_tool_handler,
        name="permission_gate",
    )
    registry.register(
        EventType.SESSION_END,
        audit.session_end_handler,
        name="security_audit",
    )

    return {"permission": permission, "sandbox": sandbox, "audit": audit}
