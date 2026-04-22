"""
第28课·数字员工的自我进化（v6）
run_validation.py — 周期性验证 + 僵尸态巡检

职责：
  1. 对所有 status=验证中 的提案，执行 validation_check 结构化判据
  2. 扫描所有中间态提案，超过 48h 发 retro_stuck 提醒

用法：
  python run_validation.py --workspace workspace/shared
"""

from __future__ import annotations

import json
import operator
from datetime import datetime, timezone
from pathlib import Path

from tools.log_ops import read_l2, read_l1, count_l2_since
from tools.log_ops import read_l3_from_sessions
from tools.mailbox_ops import send_mail
from tools.proposal_ops import list_proposals, update_proposal_status
from schemas import ValidationCheck

STALE_THRESHOLD_H = 48
INTERMEDIATE_STATES = {"待审批", "LLM预审中", "已批准", "已实施", "验证中"}

OPS = {
    ">=": operator.ge, ">": operator.gt,
    "<=": operator.le, "<": operator.lt,
    "==": operator.eq, "!=": operator.ne,
}

SCRIPT_REGISTRY = {
    "stats_l2": "_run_stats_l2",
    "find_low_quality_tasks": "_run_find_low_quality",
    "tool_call_stats": "_run_tool_call_stats",
    "stats_all_agents": "_run_stats_all_agents",
}


def _run_stats_l2(logs_dir: Path, agent_id: str = "pm", days: int = 7, **_) -> dict:
    records = read_l2(logs_dir, agent_id, days=days)
    qualities = [r.get("result_quality", 0.0) for r in records]
    return {
        "task_count": len(records),
        "avg_quality": sum(qualities) / max(len(qualities), 1),
        "failure_count": sum(1 for q in qualities if q < 0.5),
    }


def _run_find_low_quality(logs_dir: Path, agent_id: str = "pm", days: int = 7, top_k: int = 3, **_) -> dict:
    records = read_l2(logs_dir, agent_id, days=days)
    sorted_records = sorted(records, key=lambda r: r.get("result_quality", 1.0))
    return {"low_quality_count": len(sorted_records[:top_k])}


def _run_tool_call_stats(sessions_dir: Path, agent_id: str | None = None, **_) -> dict:
    steps = read_l3_from_sessions(sessions_dir, agent_id=agent_id)
    error_count = sum(
        1 for s in steps
        if any(kw in str(s.get("content", "")).lower() for kw in ("error", "fail"))
    )
    return {
        "total_steps": len(steps),
        "error_rate": error_count / max(len(steps), 1),
    }


def _run_stats_all_agents(logs_dir: Path, days: int = 7, **_) -> dict:
    result = {}
    for agent_id in ("pm", "manager"):
        records = read_l2(logs_dir, agent_id, days=days)
        qualities = [r.get("result_quality", 0.0) for r in records]
        result[agent_id] = {
            "task_count": len(records),
            "avg_quality": sum(qualities) / max(len(qualities), 1),
        }
    return result


SCRIPT_FN_MAP = {
    "stats_l2": _run_stats_l2,
    "find_low_quality_tasks": _run_find_low_quality,
    "tool_call_stats": _run_tool_call_stats,
    "stats_all_agents": _run_stats_all_agents,
}


def check(vc: ValidationCheck, logs_dir: Path, sessions_dir: Path | None = None) -> tuple[bool, str]:
    fn = SCRIPT_FN_MAP.get(vc.script)
    if not fn:
        return False, f"script 不在白名单: {vc.script}"

    kwargs = dict(vc.args)
    kwargs["logs_dir"] = logs_dir
    if sessions_dir:
        kwargs["sessions_dir"] = sessions_dir

    result = fn(**kwargs)
    actual = result.get(vc.metric)
    if actual is None:
        return False, f"metric 不存在: {vc.metric}"

    op_fn = OPS.get(vc.op)
    if not op_fn:
        return False, f"未知操作符: {vc.op}"

    passed = op_fn(actual, vc.threshold)
    return passed, f"{vc.metric}={actual} {vc.op} {vc.threshold} → {passed}"


def scan_stuck(proposals_dir: Path, mailbox_dir: Path) -> int:
    """扫描中间态超过 48h 的提案，发 retro_stuck 邮件。"""
    now = datetime.now(timezone.utc)
    stuck_count = 0

    for proposal_id, p in list_proposals(proposals_dir):
        if p.status not in INTERMEDIATE_STATES:
            continue
        if not p.status_entered_at:
            continue
        try:
            entered = datetime.fromisoformat(p.status_entered_at)
            if entered.tzinfo is None:
                entered = entered.replace(tzinfo=timezone.utc)
        except ValueError:
            continue

        age_h = (now - entered).total_seconds() / 3600
        if age_h > STALE_THRESHOLD_H:
            send_mail(
                mailbox_dir=mailbox_dir,
                to="manager",
                from_="manager",
                type_="retro_stuck",
                subject=f"提案 {proposal_id} 卡在 {p.status} 超过 {age_h:.0f}h",
                content=json.dumps({"proposal_id": proposal_id, "status": p.status}),
            )
            stuck_count += 1

    return stuck_count


def run(workspace_dir: Path) -> None:
    """执行一轮验证 + 僵尸态扫描。"""
    proposals_dir = workspace_dir / "proposals"
    logs_dir = workspace_dir / "logs"
    mailbox_dir = workspace_dir / "mailboxes"

    for proposal_id, p in list_proposals(proposals_dir):
        if p.status != "验证中" or p.validation_check is None:
            continue

        passed, msg = check(p.validation_check, logs_dir)
        if passed:
            update_proposal_status(proposals_dir, proposal_id, "已验证", note=msg)
        else:
            update_proposal_status(proposals_dir, proposal_id, "已回滚", note=msg)

    scan_stuck(proposals_dir, mailbox_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default="workspace/shared")
    args = parser.parse_args()
    run(Path(args.workspace))
