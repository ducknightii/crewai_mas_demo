"""
第28课·数字员工的自我进化（v6）
tools/log_ops.py — 三层日志读写工具

教学要点（对应第28课 P2）：
  三层日志各有用途：
    L1 人类交互层：由 mailbox_ops.send_mail(to="human") 自动写入
    L2 任务-Agent 层：由 l2_task_callback 在每次 Crew 结束后自动写入
    L3 ReAct 循环层：v6 直接复用 DigitalWorkerCrew 的 session 日志
       → workspace/<agent>/sessions/<session_id>_raw.jsonl
       → workspace/<agent>/sessions/index.jsonl（按 task_id 定位行段）

工程约定（与 mailbox_ops.py 风格一致）：
  - 所有写操作使用 FileLock(path.with_suffix(".lock"))
  - purge_old_l3 基于 record 内的 timestamp 字段判断（不依赖 file mtime）
  - 损坏记录跳过时使用 logging.warning 输出诊断信息
  - 排序使用解析后的 datetime 对象（避免时区格式混用导致的字符串比较错误）
  - 路径规则：
      L1 → logs_dir/l1_human/{msg_id}.json
      L2 → logs_dir/l2_task/{agent_id}_{task_id}.json
      L3（旧）→ logs_dir/l3_react/{agent_id}/{task_id}/step_{N}.json
      L3（v6）→ workspace/<agent>/sessions/ (raw.jsonl + index.jsonl)
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from filelock import FileLock

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# L2：任务-Agent 层
# ─────────────────────────────────────────────────────────────────────────────

def write_l2(
    logs_dir: Path,
    agent_id: str,
    task_id: str,
    record: dict,
) -> Path:
    """
    写入一条 L2 日志（任务-Agent 层）。

    文件路径：{logs_dir}/l2_task/{agent_id}_{task_id}.json
    写操作使用 FileLock，与 mailbox_ops 风格一致。

    Args:
        logs_dir:  workspace/shared/logs/ 根目录
        agent_id:  Agent 标识（如 "pm"、"manager"）
        task_id:   任务唯一 ID
        record:    日志内容 dict，应符合 L2LogRecord schema

    Returns:
        写入的文件路径
    """
    l2_dir = logs_dir / "l2_task"
    l2_dir.mkdir(parents=True, exist_ok=True)

    file_path = l2_dir / f"{agent_id}_{task_id}.json"
    lock_path  = file_path.with_suffix(".lock")

    with FileLock(str(lock_path)):
        file_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return file_path


def read_l2(
    logs_dir: Path,
    agent_id: str,
    days: int = 7,
) -> list[dict]:
    """
    读取指定 Agent 在 days 天内的 L2 日志，按 timestamp 升序返回。

    只返回 timestamp 在 [now - days, now] 窗口内的记录。
    timestamp 格式为 ISO 8601，解析失败的记录跳过（容错）并写 warning。
    排序基于解析后的 datetime 对象，避免时区格式混用问题。
    """
    l2_dir = logs_dir / "l2_task"
    if not l2_dir.exists():
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    results: list[tuple[datetime, dict]] = []

    for f in l2_dir.glob(f"{agent_id}_*.json"):
        try:
            record = json.loads(f.read_text(encoding="utf-8"))
            ts = datetime.fromisoformat(record.get("timestamp", ""))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts >= cutoff:
                results.append((ts, record))
        except Exception as exc:  # noqa: BLE001
            logger.warning("read_l2: 跳过损坏文件 %s: %s", f, exc)
            continue

    results.sort(key=lambda pair: pair[0])
    return [record for _, record in results]


# ─────────────────────────────────────────────────────────────────────────────
# L3：ReAct 循环层
# ─────────────────────────────────────────────────────────────────────────────

def write_l3(
    logs_dir: Path,
    agent_id: str,
    task_id: str,
    step_idx: int,
    record: dict,
) -> Path:
    """
    写入一条 L3 日志（ReAct 循环层，每个推理-行动步骤一条）。

    文件路径：{logs_dir}/l3_react/{agent_id}/{task_id}/step_{step_idx}.json
    """
    l3_dir = logs_dir / "l3_react" / agent_id / task_id
    l3_dir.mkdir(parents=True, exist_ok=True)

    file_path = l3_dir / f"step_{step_idx}.json"
    lock_path  = file_path.with_suffix(".lock")

    with FileLock(str(lock_path)):
        file_path.write_text(
            json.dumps(record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return file_path


def read_l3(
    logs_dir: Path,
    agent_id: str,
    task_id: str,
) -> list[dict]:
    """读取某 Agent 某任务的全部 L3 步骤日志，按 step_idx 升序。"""
    l3_dir = logs_dir / "l3_react" / agent_id / task_id
    if not l3_dir.exists():
        return []

    steps: list[dict] = []
    for f in sorted(l3_dir.glob("step_*.json")):
        try:
            steps.append(json.loads(f.read_text(encoding="utf-8")))
        except Exception as exc:  # noqa: BLE001
            logger.warning("read_l3: 跳过损坏文件 %s: %s", f, exc)
            continue
    return steps


def purge_old_l3(
    logs_dir: Path,
    retention_days: int = 30,
) -> int:
    """
    清理 L3 日志中 retention_days 天前的记录。

    判断依据：record 内的 "timestamp" 字段（ISO 8601），而非 file mtime。
    这样测试中无需 os.utime，直接在 record 里写过去的时间即可。

    Returns:
        删除的文件数量
    """
    l3_base = logs_dir / "l3_react"
    if not l3_base.exists():
        return 0

    cutoff  = datetime.now(timezone.utc) - timedelta(days=retention_days)
    deleted = 0

    for f in l3_base.rglob("step_*.json"):
        try:
            record = json.loads(f.read_text(encoding="utf-8"))
            ts_str = record.get("timestamp", "")
            if not ts_str:
                continue
            ts = datetime.fromisoformat(ts_str)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts < cutoff:
                f.unlink(missing_ok=True)
                deleted += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("purge_old_l3: 跳过损坏文件 %s: %s", f, exc)
            continue

    return deleted


# ─────────────────────────────────────────────────────────────────────────────
# L3（v6）：从 session 日志读取 ReAct 步骤
# ─────────────────────────────────────────────────────────────────────────────

def read_session_index(sessions_dir: Path) -> list[dict]:
    """
    读取 sessions/index.jsonl，返回全部索引条目。

    每行格式：
      {"session_id","task_id","agent_id","start_ts","end_ts","start_line","end_line"}
    """
    idx_file = sessions_dir / "index.jsonl"
    if not idx_file.exists():
        return []

    entries: list[dict] = []
    for line in idx_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            entries.append(json.loads(line))
        except Exception as exc:  # noqa: BLE001
            logger.warning("read_session_index: 跳过损坏行: %s", exc)
    return entries


def read_l3_from_sessions(
    sessions_dir: Path,
    task_id: str | None = None,
    agent_id: str | None = None,
    only_failed: bool = False,
) -> list[dict]:
    """
    v6：从 session 原始日志读取 L3 级别的 ReAct 步骤。

    流程：
      1. 读 index.jsonl 定位 session_id + 行范围
      2. 读 {session_id}_raw.jsonl 的对应行段
      3. 可选按 task_id / agent_id 过滤

    Args:
        sessions_dir: workspace/<agent>/sessions/ 目录
        task_id:      按 task_id 过滤（None 则不过滤）
        agent_id:     按 agent_id 过滤（None 则不过滤）
        only_failed:  只返回含 error/fail 关键词的步骤

    Returns:
        按时间顺序排列的消息列表
    """
    entries = read_session_index(sessions_dir)
    if not entries:
        return []

    if task_id:
        entries = [e for e in entries if e.get("task_id") == task_id]
    if agent_id:
        entries = [e for e in entries if e.get("agent_id") == agent_id]
    if not entries:
        return []

    results: list[dict] = []
    for entry in entries:
        raw_file = sessions_dir / f"{entry['session_id']}_raw.jsonl"
        if not raw_file.exists():
            logger.warning("read_l3_from_sessions: 文件不存在 %s", raw_file)
            continue

        start_line = entry.get("start_line", 0)
        end_line = entry.get("end_line")

        try:
            lines = raw_file.read_text(encoding="utf-8").splitlines()
        except Exception as exc:  # noqa: BLE001
            logger.warning("read_l3_from_sessions: 读取失败 %s: %s", raw_file, exc)
            continue

        for i, line in enumerate(lines):
            if i < start_line:
                continue
            if end_line is not None and i >= end_line:
                break
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except Exception:  # noqa: BLE001
                continue

            if only_failed:
                content = str(record.get("content", ""))
                if not any(kw in content.lower() for kw in ("error", "fail", "exception", "traceback")):
                    continue

            results.append(record)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# L1：人类交互层（只读，写入由 mailbox_ops 负责）
# ─────────────────────────────────────────────────────────────────────────────

def read_l1(
    logs_dir: Path,
    days: int = 7,
) -> list[dict]:
    """
    读取 L1 日志（人类交互层）中 days 天内的记录，按 timestamp 升序返回。

    L1 日志由 mailbox_ops.send_mail(to="human") 自动写入，
    复盘函数通过此接口读取"人类纠正事件"。
    排序基于解析后的 datetime 对象，避免时区格式混用问题。
    """
    l1_dir = logs_dir / "l1_human"
    if not l1_dir.exists():
        return []

    cutoff  = datetime.now(timezone.utc) - timedelta(days=days)
    results: list[tuple[datetime, dict]] = []

    for f in l1_dir.glob("*.json"):
        try:
            record = json.loads(f.read_text(encoding="utf-8"))
            ts = datetime.fromisoformat(record.get("timestamp", ""))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts >= cutoff:
                results.append((ts, record))
        except Exception as exc:  # noqa: BLE001
            logger.warning("read_l1: 跳过损坏文件 %s: %s", f, exc)
            continue

    results.sort(key=lambda pair: pair[0])
    return [record for _, record in results]


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────

def new_task_id() -> str:
    """生成短 task ID，用于演示。"""
    return str(uuid.uuid4())[:8]


def count_l2_since(logs_dir: Path, agent_id: str, hours: int = 24) -> int:
    """统计 agent_id 在最近 hours 小时内的 L2 日志条数（scheduler 用）。"""
    l2_dir = logs_dir / "l2_task"
    if not l2_dir.exists():
        return 0

    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    count = 0
    for f in l2_dir.glob(f"{agent_id}_*.json"):
        try:
            record = json.loads(f.read_text(encoding="utf-8"))
            ts = datetime.fromisoformat(record.get("timestamp", ""))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts >= cutoff:
                count += 1
        except Exception:  # noqa: BLE001
            continue
    return count
