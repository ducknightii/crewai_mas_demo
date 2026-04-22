"""
邮箱操作 Python API — 供单元测试和本地工具直接导入。

三态状态机（类比 AWS SQS Visibility Timeout）：
  send_mail  → status: "unread"
  read_inbox → status: "in_progress" + processing_since（原子操作，防重复取走）
  mark_done  → status: "done"（Agent 处理完后调用）
  reset_stale→ in_progress 超时 → unread（崩溃恢复）

注意：同一逻辑以 CLI 形式在 workspace/*/skills/mailbox/scripts/mailbox_cli.py 中实现，
供 Agent 在沙盒中通过 Bash 调用。两者保持同步。
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock

# ── 三态常量 ──────────────────────────────────────────────────────────────────
STATUS_UNREAD      = "unread"
STATUS_IN_PROGRESS = "in_progress"
STATUS_DONE        = "done"


def send_mail(
    mailbox_dir: Path,
    to: str,
    from_: str,
    type_: str,
    subject: str,
    content: str,
) -> str:
    """发送邮件到目标邮箱（filelock 保护）。

    返回消息 ID。消息初始状态为 unread。
    """
    inbox_path = mailbox_dir / f"{to}.json"
    lock_path  = inbox_path.with_suffix(".json.lock")

    msg: dict = {
        "id":               f"msg-{uuid.uuid4().hex[:8]}",
        "from":             from_,
        "to":               to,
        "type":             type_,
        "subject":          subject,
        "content":          content,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "status":           STATUS_UNREAD,
        "processing_since": None,
    }

    with FileLock(str(lock_path)):
        inbox: list[dict] = (
            json.loads(inbox_path.read_text(encoding="utf-8"))
            if inbox_path.exists()
            else []
        )
        inbox.append(msg)
        inbox_path.write_text(
            json.dumps(inbox, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return msg["id"]


def read_inbox(mailbox_dir: Path, role: str) -> list[dict]:
    """读取未读消息并原子标记为 in_progress（filelock 保护）。

    返回消息快照（副本），调用方修改快照不影响文件中的数据。
    """
    inbox_path = mailbox_dir / f"{role}.json"
    lock_path  = inbox_path.with_suffix(".json.lock")

    unread_snapshots: list[dict] = []

    with FileLock(str(lock_path)):
        inbox: list[dict] = (
            json.loads(inbox_path.read_text(encoding="utf-8"))
            if inbox_path.exists()
            else []
        )
        for msg in inbox:
            if msg.get("status") == STATUS_UNREAD:
                msg["status"]           = STATUS_IN_PROGRESS
                msg["processing_since"] = datetime.now(timezone.utc).isoformat()
                unread_snapshots.append(dict(msg))  # 返回副本
        inbox_path.write_text(
            json.dumps(inbox, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return unread_snapshots


def mark_done(
    mailbox_dir: Path,
    role: str,
    msg_ids: list[str],
) -> int:
    """将指定 in_progress 消息标记为 done。

    返回实际标记数量。
    """
    inbox_path = mailbox_dir / f"{role}.json"
    lock_path  = inbox_path.with_suffix(".json.lock")
    target_ids = set(msg_ids)
    count = 0

    with FileLock(str(lock_path)):
        inbox: list[dict] = (
            json.loads(inbox_path.read_text(encoding="utf-8"))
            if inbox_path.exists()
            else []
        )
        for msg in inbox:
            if msg["id"] in target_ids and msg.get("status") == STATUS_IN_PROGRESS:
                msg["status"]           = STATUS_DONE
                msg["processing_since"] = None
                count += 1
        inbox_path.write_text(
            json.dumps(inbox, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return count


def mark_done_all_in_progress(mailbox_dir: Path, role: str) -> int:
    """批量将所有 in_progress 消息标记为 done（Crew 成功后编排器调用）。

    返回标记数量。
    """
    inbox_path = mailbox_dir / f"{role}.json"
    lock_path  = inbox_path.with_suffix(".json.lock")
    count = 0

    with FileLock(str(lock_path)):
        inbox: list[dict] = (
            json.loads(inbox_path.read_text(encoding="utf-8"))
            if inbox_path.exists()
            else []
        )
        for msg in inbox:
            if msg.get("status") == STATUS_IN_PROGRESS:
                msg["status"]           = STATUS_DONE
                msg["processing_since"] = None
                count += 1
        inbox_path.write_text(
            json.dumps(inbox, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return count


def reset_stale(
    mailbox_dir: Path,
    role: str,
    timeout_seconds: int = 900,
) -> int:
    """崩溃恢复：将超时的 in_progress 消息重置为 unread。

    类比 AWS SQS Visibility Timeout 到期后消息重新可见。
    返回重置数量。
    """
    inbox_path = mailbox_dir / f"{role}.json"
    lock_path  = inbox_path.with_suffix(".json.lock")
    count = 0
    now   = datetime.now(timezone.utc)

    with FileLock(str(lock_path)):
        inbox: list[dict] = (
            json.loads(inbox_path.read_text(encoding="utf-8"))
            if inbox_path.exists()
            else []
        )
        for msg in inbox:
            if msg.get("status") == STATUS_IN_PROGRESS and msg.get("processing_since"):
                # 兼容带 +00:00 时区后缀的 ISO 格式（Python 3.11+ 原生支持，低版本需替换）
                started = datetime.fromisoformat(
                    msg["processing_since"].replace("Z", "+00:00")
                )
                if (now - started).total_seconds() > timeout_seconds:
                    msg["status"]           = STATUS_UNREAD
                    msg["processing_since"] = None
                    count += 1
        inbox_path.write_text(
            json.dumps(inbox, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return count
