"""沙箱输入消毒——确定性规则检查，零 LLM 依赖。

设计参照：Claude Code cyberRiskInstruction.ts 的四层确定性防御。
本模块实现其中三层：路径归一化 / 危险命令检测 / 环境变量引用检测。
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import TYPE_CHECKING

from hook_framework.registry import GuardrailDeny

if TYPE_CHECKING:
    from .audit_logger import SecurityAuditLogger

_PATH_TRAVERSAL = re.compile(r"\.\.[/\\]")
_ENV_VAR_REF = re.compile(r"\$\{?\w+\}?")
_DANGEROUS_COMMANDS = re.compile(
    r"\b(rm\s+-rf|sudo|chmod\s+777|curl\s.*\|.*sh|wget\s.*\|.*sh|eval|exec)\b",
    re.IGNORECASE,
)
# 不含 () 和 $：括号在自然语言常见，$ 由 _ENV_VAR_REF 单独处理
_SHELL_INJECTION = re.compile(r"[;&|`]")


class SandboxGuard:
    def __init__(
        self,
        workspace_root: str = "",
        audit: SecurityAuditLogger | None = None,
    ):
        self._workspace_root = os.path.abspath(workspace_root) if workspace_root else ""
        self._violations: list[dict] = []
        self._audit = audit

    def before_tool_handler(self, ctx):
        """BEFORE_TOOL_CALL: 对工具输入做安全消毒。"""
        tool_input = str(ctx.tool_input) if ctx.tool_input else ""

        if _PATH_TRAVERSAL.search(tool_input):
            self._record_violation(ctx, "path_traversal", tool_input)
            raise GuardrailDeny(
                f"Path traversal blocked in tool '{ctx.tool_name}': "
                f"input contains '../'"
            )

        match = _DANGEROUS_COMMANDS.search(tool_input)
        if match:
            self._record_violation(ctx, "dangerous_command", tool_input)
            raise GuardrailDeny(
                f"Dangerous command blocked in tool '{ctx.tool_name}': "
                f"'{match.group()}'"
            )

        if _SHELL_INJECTION.search(tool_input):
            self._record_violation(ctx, "shell_injection", tool_input)
            raise GuardrailDeny(
                f"Shell injection characters blocked in tool '{ctx.tool_name}'"
            )

        if _ENV_VAR_REF.search(tool_input):
            self._record_warning(ctx, "env_var_reference", tool_input)

    def _record_violation(self, ctx, violation_type: str, input_preview: str):
        violation = {
            "type": violation_type,
            "tool": ctx.tool_name,
            "input_preview": input_preview[:200],
            "session_id": ctx.session_id,
        }
        self._violations.append(violation)
        record = {
            "level": "CRITICAL",
            "guardrail": "sandbox_guard",
            "violation": violation_type,
            "tool": ctx.tool_name,
            "blocked": True,
        }
        print(json.dumps(record, ensure_ascii=False), file=sys.stderr)
        if self._audit:
            self._audit.record_event(f"sandbox_{violation_type}", {
                "tool": ctx.tool_name,
                "input_preview": input_preview[:100],
            })

    def _record_warning(self, ctx, warning_type: str, input_preview: str):
        record = {
            "level": "WARNING",
            "guardrail": "sandbox_guard",
            "warning": warning_type,
            "tool": ctx.tool_name,
            "input_preview": input_preview[:100],
        }
        print(json.dumps(record, ensure_ascii=False), file=sys.stderr)

    def get_metrics(self) -> dict:
        by_type: dict[str, int] = {}
        for v in self._violations:
            by_type[v["type"]] = by_type.get(v["type"], 0) + 1
        return {
            "total_violations": len(self._violations),
            "violations_by_type": by_type,
            "blocked_tools": list({v["tool"] for v in self._violations}),
        }
