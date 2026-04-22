"""
共享工作区初始化工具 — 供单元测试和本地工具直接导入。

同一逻辑以 CLI 形式在 workspace/manager/skills/init_project/scripts/init_workspace.py 中实现，
供 Agent 在沙盒中通过 Bash 调用。
"""

from __future__ import annotations

from pathlib import Path


def create_workspace(
    shared_dir: Path,
    roles: list[str],
    project_name: str = "",
) -> dict:
    """创建共享工作区目录结构（幂等：已存在的文件不覆盖）。

    目录结构：
      shared_dir/
        needs/                  # 需求文档（Manager 写入）
        design/                 # 设计文档（PM 写入）
        mailboxes/              # 各角色邮箱
          {role}.json           # 初始空数组
        WORKSPACE_RULES.md      # 工作区访问规范

    返回创建报告：
      {
        "created_dirs":   [相对路径, ...],
        "created_files":  [相对路径, ...],
        "skipped_files":  [相对路径, ...],
      }
    """
    created_dirs: list[str]  = []
    created_files: list[str] = []
    skipped_files: list[str] = []

    def _mkdir(path: Path, label: str) -> None:
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            created_dirs.append(label)

    def _write_if_absent(path: Path, content: str, label: str) -> None:
        if path.exists():
            skipped_files.append(label)
        else:
            path.write_text(content, encoding="utf-8")
            created_files.append(label)

    # 创建子目录
    _mkdir(shared_dir / "needs",    "needs/")
    _mkdir(shared_dir / "design",   "design/")
    _mkdir(shared_dir / "mailboxes","mailboxes/")

    # 创建各角色邮箱（初始空数组）
    for role in roles:
        _write_if_absent(
            shared_dir / "mailboxes" / f"{role}.json",
            "[]",
            f"mailboxes/{role}.json",
        )

    # 创建工作区访问规范
    rules_content = _build_workspace_rules(project_name, roles)
    _write_if_absent(
        shared_dir / "WORKSPACE_RULES.md",
        rules_content,
        "WORKSPACE_RULES.md",
    )

    return {
        "created_dirs":  created_dirs,
        "created_files": created_files,
        "skipped_files": skipped_files,
    }


def _build_workspace_rules(project_name: str, roles: list[str]) -> str:
    project_line = f"# 共享工作区访问规范\n\n项目：{project_name or '（未命名）'}\n"
    return (
        f"{project_line}\n"
        "## 目录权限\n\n"
        "| 目录 | 权限 | 说明 |\n"
        "|------|------|------|\n"
        "| `/mnt/shared/needs/` | 只读（所有角色）| 需求文档来源，不得修改 |\n"
        "| `/mnt/shared/design/` | 可读写（PM）| PM 输出产品文档 |\n"
        "| `/mnt/shared/mailboxes/` | 可读写（通过 mailbox skill）| 角色间通信 |\n\n"
        "## 邮箱规范\n\n"
        "- 邮件内容只写路径引用，不把文档全文放进邮件\n"
        "- 消息类型：`task_assign`（任务分配）/ `task_done`（任务完成）\n"
        "- 消息状态：`unread` → `in_progress` → `done`（三态状态机）\n"
    )
