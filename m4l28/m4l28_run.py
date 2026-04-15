"""
课程：28｜数字员工的自我进化
示例文件：m4l28_run.py

在第27课四步任务链基础上，新增三层日志记录 + 两种复盘演示：

【复用 m4l27 四步（不变）】
  步骤1（Manager）：需求澄清 → 写 requirements.md
  [人工确认节点1] 需求文档确认
  步骤2（Manager）：读SOP → 向 PM 发送 task_assign
  步骤3（PM）：读邮件 → 写产品文档 → 发 manager.json:task_done
  [人工确认节点2] 设计文档 Checkpoint
  步骤4（Manager）：验收产品文档

【第28课新增】
  [日志写入演示]     步骤3结束后写 PM 的 L2 日志；human.json 写入时自动写 L1
  [预置历史数据]     seed_logs() 预置7天模拟运行数据（含3次 checkpoint 退回）
  [自我复盘演示]     PM 自我复盘 → 生成改进提案 → 写 human.json
  [团队复盘演示]     Manager 团队复盘 → 识别瓶颈 → 发邮件+周报

核心教学点（对应第28课）：
  P2 三层日志：L1 自动写入（mailbox_ops 内置），L2 run.py 手动写
  P4 自我复盘：结构化 prompt + json_object 模式 + Pydantic 校验
  P5 团队复盘：聚合统计 + 识别瓶颈 + 触发下级复盘
  P7 反模式：最小样本量保护、L3 滚动清理
"""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from filelock import FileLock

_M4L28_DIR    = Path(__file__).resolve().parent
_PROJECT_ROOT = _M4L28_DIR.parent
for _p in [str(_PROJECT_ROOT), str(_M4L28_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from crewai.hooks import clear_before_llm_call_hooks  # noqa: E402
from tools.mailbox_ops import send_mail               # noqa: E402
from tools.log_ops import write_l2, purge_old_l3      # noqa: E402

SHARED_DIR    = _M4L28_DIR / "workspace" / "shared"
MAILBOXES_DIR = SHARED_DIR / "mailboxes"
LOGS_DIR      = SHARED_DIR / "logs"
DESIGN_DIR    = SHARED_DIR / "design"


# ─────────────────────────────────────────────────────────────────────────────
# 人工确认（复用 m4l27 逻辑）
# ─────────────────────────────────────────────────────────────────────────────

def wait_for_human(human_inbox: Path, expected_type: str, step_label: str) -> bool:
    lock_path = human_inbox.with_suffix(".lock")

    with FileLock(str(lock_path)):
        if not human_inbox.exists():
            human_inbox.write_text("[]", encoding="utf-8")
        messages = json.loads(human_inbox.read_text(encoding="utf-8"))

    target = next(
        (m for m in messages if m.get("type") == expected_type and not m.get("read", False)),
        None,
    )

    if target is None:
        print(f"\n[debug] ⚠️  human.json 中未找到未读的 '{expected_type}' 消息")
        print(f"[debug] 实际内容：{json.dumps(messages, ensure_ascii=False, indent=2)}")
        return False

    print(f"\n{'='*60}")
    print(f"  ⏸️  [人工确认节点] {step_label}")
    print(f"  来自：{target.get('from', 'manager')}")
    print(f"  主题：{target.get('subject', '')}")
    print(f"  内容：{target.get('content', '')[:300]}")
    print(f"{'='*60}")

    decision = input("  你的决定 (y/n)：").strip().lower()

    if decision == "y":
        with FileLock(str(lock_path)):
            messages = json.loads(human_inbox.read_text(encoding="utf-8"))
            for m in messages:
                if m.get("id") == target["id"]:
                    m["read"] = True
                    break
            human_inbox.write_text(
                json.dumps(messages, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        print(f"  ✅ 已确认：{step_label}\n")
        return True
    else:
        print(f"  ❌ 已拒绝：{step_label}，演示终止\n")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# 步骤检查（复用 m4l27 逻辑）
# ─────────────────────────────────────────────────────────────────────────────

def check_requirements_exists() -> bool:
    return (SHARED_DIR / "needs" / "requirements.md").exists()


def check_pm_inbox_has_task_assign() -> bool:
    pm_inbox = MAILBOXES_DIR / "pm.json"
    if not pm_inbox.exists():
        return False
    return any(m.get("type") == "task_assign"
               for m in json.loads(pm_inbox.read_text(encoding="utf-8")))


def check_product_spec_exists() -> bool:
    return (DESIGN_DIR / "product_spec.md").exists()


def check_manager_inbox_has_task_done() -> bool:
    manager_inbox = MAILBOXES_DIR / "manager.json"
    if not manager_inbox.exists():
        return False
    return any(m.get("type") == "task_done"
               for m in json.loads(manager_inbox.read_text(encoding="utf-8")))


# ─────────────────────────────────────────────────────────────────────────────
# 主演示流程
# ─────────────────────────────────────────────────────────────────────────────

def run_demo(initial_request: str = "") -> None:
    from m4l28_manager import (
        RequirementsDiscoveryCrew,
        ManagerAssignCrew,
        ManagerReviewCrew,
        ManagerTeamRetroCrew,
        save_session as manager_save,
    )
    from m4l28_pm import PMExecuteCrew, PMSelfRetroCrew, save_session as pm_save
    from seed_logs import seed_logs

    session_id = str(uuid.uuid4())
    if not initial_request:
        initial_request = input("请告诉 Manager 你要做什么：").strip()

    print(f"\n{'='*60}")
    print(f"  M4L28 数字员工的自我进化演示  |  session: {session_id[:8]}...")
    print(f"{'='*60}\n")

    # ══════════════════════════════════════════════════════════════════════════
    # 【复用 m4l27 四步】
    # ══════════════════════════════════════════════════════════════════════════

    print("【步骤1】Manager 需求澄清中...\n")
    clear_before_llm_call_hooks()
    req_crew = RequirementsDiscoveryCrew(session_id=session_id)
    result1  = req_crew.crew().kickoff(inputs={"user_request": initial_request})
    manager_save(req_crew, session_id)
    print(f"\n步骤1 输出：{getattr(result1, 'raw', str(result1))[:200]}...\n")

    if not check_requirements_exists():
        print("⚠️  步骤1未完成：requirements.md 未生成，终止运行")
        return

    send_mail(MAILBOXES_DIR, to="human", from_="manager",
              type_="needs_confirm", subject="需求文档已整理完毕，请确认",
              content="需求文档路径：shared/needs/requirements.md")
    # ↑ 此处 send_mail(to="human") 同时自动写入 L1 日志（第28课新增）

    if not wait_for_human(MAILBOXES_DIR / "human.json", "needs_confirm", "需求文档确认"):
        return

    print("【步骤2】Manager 按SOP分配任务给PM...\n")
    clear_before_llm_call_hooks()
    assign_crew = ManagerAssignCrew(session_id=session_id)
    assign_crew.crew().kickoff(inputs={"user_request": "请读取需求文档和产品设计SOP，向PM发送产品文档设计任务"})
    manager_save(assign_crew, session_id)

    if not check_pm_inbox_has_task_assign():
        print("⚠️  步骤2未完成：PM 邮箱中未找到 task_assign，终止运行")
        return
    print("✅ 步骤2检查通过：PM 邮箱已有任务分配邮件\n")

    print("【步骤3】PM 撰写产品文档...\n")
    clear_before_llm_call_hooks()
    pm_crew = PMExecuteCrew(session_id=session_id)
    pm_crew.crew().kickoff(inputs={"user_request": "请读取邮箱中的任务，根据需求文档撰写产品规格文档，完成后通知Manager"})
    pm_save(pm_crew, session_id)

    if not check_product_spec_exists() or not check_manager_inbox_has_task_done():
        print("⚠️  步骤3未完成：产品文档未生成或 Manager 未收到完成通知，终止运行")
        return
    print("✅ 步骤3检查通过：产品文档已生成\n")

    # ──────────────────────────────────────────────────────────────────────────
    # 第28课新增：写入本次 PM 任务的 L2 日志
    # ──────────────────────────────────────────────────────────────────────────
    task_id = session_id[:8]
    write_l2(
        logs_dir = LOGS_DIR,
        agent_id = "pm",
        task_id  = f"live_{task_id}",
        record   = {
            "agent_id":       "pm",
            "task_id":        f"live_{task_id}",
            "task_desc":      "产品规格文档设计任务（步骤3）",
            "result_quality": 0.75,   # 演示用固定值
            "duration_sec":   0,
            "error_type":     None,
            "timestamp":      datetime.now(timezone.utc).isoformat(),
        },
    )
    print("✅ 第28课：PM L2 日志已写入\n")

    send_mail(MAILBOXES_DIR, to="human", from_="manager",
              type_="checkpoint_request", subject="产品文档已完成，请确认后继续",
              content="产品文档路径：shared/design/product_spec.md")
    # ↑ 同时自动写入 L1 日志

    if not wait_for_human(MAILBOXES_DIR / "human.json", "checkpoint_request", "设计文档 Checkpoint"):
        return

    print("【步骤4】Manager 验收中...\n")
    clear_before_llm_call_hooks()
    review_crew = ManagerReviewCrew(session_id=session_id)
    review_crew.crew().kickoff(inputs={"user_request": "请读取邮箱中的完成通知，验收产品文档并保存验收结论"})
    manager_save(review_crew, session_id)

    print(f"\n{'='*60}")
    print("  ✅ m4l27 四步任务链演示完成！")
    print(f"{'='*60}\n")

    # ══════════════════════════════════════════════════════════════════════════
    # 【第28课新增：日志 + 复盘演示】
    # ══════════════════════════════════════════════════════════════════════════

    print(f"\n{'='*60}")
    print("  📊 第28课新增：预置历史日志 + 触发复盘")
    print(f"{'='*60}\n")

    # 清理 30 天前的 L3 日志（演示 L3 滚动清理）
    deleted = purge_old_l3(LOGS_DIR, retention_days=30)
    if deleted > 0:
        print(f"[清理] 已删除 {deleted} 条 30 天前的 L3 日志\n")

    # 预置 7 天历史数据
    print("⏳ 预置历史日志（模拟团队已运行 7 天）...\n")
    seed_logs(base_dir=_M4L28_DIR / "workspace")

    # 自我复盘（通过 PMSelfRetroCrew + self-retrospective Skill 在沙盒中执行）
    print(f"\n{'─'*50}")
    print("  🔍 触发 PM Agent 自我复盘（Skill 体系）")
    print(f"{'─'*50}\n")
    clear_before_llm_call_hooks()
    self_retro_crew = PMSelfRetroCrew(session_id=session_id)
    self_retro_crew.crew().kickoff(inputs={"user_request": (
        "请执行 PM 自我复盘：分析过去7天的任务日志，"
        "使用 self-retrospective skill 生成改进提案，并发送至 human.json 等待审批。"
    )})
    pm_save(self_retro_crew, session_id)

    # 团队复盘（通过 ManagerTeamRetroCrew + team-retrospective Skill 在沙盒中执行）
    print(f"\n{'─'*50}")
    print("  📋 触发 Manager 团队复盘（Skill 体系）")
    print(f"{'─'*50}\n")
    clear_before_llm_call_hooks()
    team_retro_crew = ManagerTeamRetroCrew(session_id=session_id)
    team_retro_crew.crew().kickoff(inputs={"user_request": (
        "请执行团队复盘：统计 PM 和 Manager 过去7天的质量指标，"
        "识别瓶颈 Agent，生成团队级改进提案，并向 human.json 发送周报。"
    )})
    manager_save(team_retro_crew, session_id)

    print(f"\n{'='*60}")
    print("  ✅ 第28课演示完成！")
    print(f"  查看以下文件了解完整闭环：")
    print(f"  - workspace/shared/logs/    （三层日志）")
    print(f"  - workspace/shared/proposals/proposals.json  （改进提案）")
    print(f"  - workspace/shared/mailboxes/human.json       （待审批队列+周报）")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_demo()
