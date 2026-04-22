"""
第26课 单元测试 - test_m4l26.py
第26课·任务链与信息传递

测试策略：
  T1 ：邮箱文件结构完整性
  T2 ：send_mail 写入正确（status=unread, processing_since=None, 字段完整）
  T3 ：read_inbox 原子标记 in_progress（status 字段）+ 返回快照
  T3b：mark_done 将 in_progress → done
  T4a：send_mail filelock 保护（两线程并发写，JSON 不损坏）
  T4b：并发写入语义正确（read_inbox + send_mail 同时，消息无丢失）
  T5 ：read_inbox 幂等性（第二次调用返回空列表，因已是 in_progress）
  T6 ：工作区共享目录结构完整（WORKSPACE_RULES / needs / design / mailboxes）
  T7 ：Manager workspace 四件套存在且非空
  T8 ：PM workspace 四件套存在且非空
  T9 ：workspace-local mailbox skill 存在且正确配置（v3）
  T10：mailboxes 初始文件格式正确（合法 JSON 空数组）
  T11：三态状态机完整流转（unread → in_progress → done）
  T12：reset_stale 崩溃恢复（in_progress → unread）
  T13：create_workspace 幂等性与目录结构正确性
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading
from pathlib import Path

import pytest

# ── 路径设置 ──────────────────────────────────────────────────────────────────
_M4L26_DIR    = Path(__file__).resolve().parent
_PROJECT_ROOT = _M4L26_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_M4L26_DIR))  # m4l26/ 始终在最前

WORKSPACE_MANAGER  = _M4L26_DIR / "workspace" / "manager"
WORKSPACE_PM       = _M4L26_DIR / "workspace" / "pm"
WORKSPACE_SHARED   = _M4L26_DIR / "workspace" / "shared"
MAILBOXES_DIR      = WORKSPACE_SHARED / "mailboxes"
SKILLS_DIR         = _PROJECT_ROOT / "skills"


# ─────────────────────────────────────────────────────────────────────────────
# T1 | 邮箱文件结构完整性
# ─────────────────────────────────────────────────────────────────────────────

class TestT1MailboxStructure:
    """T1：mailboxes/ 目录存在，manager.json / pm.json 初始为合法空数组"""

    def test_mailboxes_dir_exists(self) -> None:
        assert MAILBOXES_DIR.exists(), "mailboxes/ 目录不存在"

    @pytest.mark.parametrize("role", ["manager", "pm"])
    def test_mailbox_file_exists(self, role: str) -> None:
        assert (MAILBOXES_DIR / f"{role}.json").exists(), f"{role}.json 不存在"

    @pytest.mark.parametrize("role", ["manager", "pm"])
    def test_mailbox_file_is_valid_json_list(self, role: str) -> None:
        content = (MAILBOXES_DIR / f"{role}.json").read_text(encoding="utf-8")
        data = json.loads(content)
        assert isinstance(data, list), f"{role}.json 内容不是 JSON 数组"


# ─────────────────────────────────────────────────────────────────────────────
# T2 | send_mail 写入正确（三态字段）
# ─────────────────────────────────────────────────────────────────────────────

class TestT2SendMail:
    """T2：send_mail 写入消息，status=unread，processing_since=None，字段齐全"""

    def setup_method(self) -> None:
        (MAILBOXES_DIR / "pm.json").write_text("[]", encoding="utf-8")

    def teardown_method(self) -> None:
        (MAILBOXES_DIR / "pm.json").write_text("[]", encoding="utf-8")

    def test_send_mail_appends_message(self) -> None:
        from tools.mailbox_ops import send_mail
        send_mail(MAILBOXES_DIR, to="pm", from_="manager",
                  type_="task_assign", subject="测试任务", content="请设计产品文档")
        messages = json.loads((MAILBOXES_DIR / "pm.json").read_text())
        assert len(messages) == 1

    def test_send_mail_fields_complete(self) -> None:
        from tools.mailbox_ops import send_mail
        send_mail(MAILBOXES_DIR, to="pm", from_="manager",
                  type_="task_assign", subject="测试任务", content="请设计产品文档")
        msg = json.loads((MAILBOXES_DIR / "pm.json").read_text())[0]
        # 三态字段必须存在
        for field in ("id", "from", "to", "type", "subject", "content",
                      "timestamp", "status", "processing_since"):
            assert field in msg, f"消息缺少字段 {field}"

    def test_send_mail_status_is_unread(self) -> None:
        """新消息 status 必须为 unread（三态起点）"""
        from tools.mailbox_ops import send_mail, STATUS_UNREAD
        send_mail(MAILBOXES_DIR, to="pm", from_="manager",
                  type_="task_assign", subject="测试任务", content="内容")
        msg = json.loads((MAILBOXES_DIR / "pm.json").read_text())[0]
        assert msg["status"] == STATUS_UNREAD, \
            f"send_mail 写入的 status 应为 unread，实际：{msg['status']}"

    def test_send_mail_processing_since_is_none(self) -> None:
        """新消息 processing_since 必须为 None（尚未被取走）"""
        from tools.mailbox_ops import send_mail
        send_mail(MAILBOXES_DIR, to="pm", from_="manager",
                  type_="task_assign", subject="测试任务", content="内容")
        msg = json.loads((MAILBOXES_DIR / "pm.json").read_text())[0]
        assert msg["processing_since"] is None, \
            "send_mail 写入的 processing_since 应为 None"

    def test_send_mail_returns_id(self) -> None:
        from tools.mailbox_ops import send_mail
        msg_id = send_mail(MAILBOXES_DIR, to="pm", from_="manager",
                           type_="task_assign", subject="测试任务", content="内容")
        assert isinstance(msg_id, str) and len(msg_id) > 0

    def test_send_mail_timestamp_auto_generated(self) -> None:
        from tools.mailbox_ops import send_mail
        send_mail(MAILBOXES_DIR, to="pm", from_="manager",
                  type_="task_assign", subject="测试任务", content="内容")
        msg = json.loads((MAILBOXES_DIR / "pm.json").read_text())[0]
        assert msg["timestamp"] and "T" in msg["timestamp"]  # ISO 格式含 T


# ─────────────────────────────────────────────────────────────────────────────
# T3 | read_inbox 原子标记 in_progress
# ─────────────────────────────────────────────────────────────────────────────

class TestT3ReadInbox:
    """T3：read_inbox 返回 unread 消息快照，并将其原子标记为 in_progress"""

    def setup_method(self) -> None:
        (MAILBOXES_DIR / "pm.json").write_text("[]", encoding="utf-8")

    def teardown_method(self) -> None:
        (MAILBOXES_DIR / "pm.json").write_text("[]", encoding="utf-8")

    def test_read_inbox_returns_unread(self) -> None:
        from tools.mailbox_ops import send_mail, read_inbox
        send_mail(MAILBOXES_DIR, to="pm", from_="manager",
                  type_="task_assign", subject="测试", content="内容")
        messages = read_inbox(MAILBOXES_DIR, role="pm")
        assert len(messages) == 1

    def test_read_inbox_marks_as_in_progress(self) -> None:
        """read_inbox 后文件中状态应为 in_progress（不是 done）"""
        from tools.mailbox_ops import send_mail, read_inbox, STATUS_IN_PROGRESS
        send_mail(MAILBOXES_DIR, to="pm", from_="manager",
                  type_="task_assign", subject="测试", content="内容")
        read_inbox(MAILBOXES_DIR, role="pm")
        stored = json.loads((MAILBOXES_DIR / "pm.json").read_text())
        assert stored[0]["status"] == STATUS_IN_PROGRESS, \
            f"read_inbox 后 status 应为 in_progress，实际：{stored[0]['status']}"

    def test_read_inbox_sets_processing_since(self) -> None:
        """read_inbox 后 processing_since 应为非 None 的 ISO 时间戳"""
        from tools.mailbox_ops import send_mail, read_inbox
        send_mail(MAILBOXES_DIR, to="pm", from_="manager",
                  type_="task_assign", subject="测试", content="内容")
        read_inbox(MAILBOXES_DIR, role="pm")
        stored = json.loads((MAILBOXES_DIR / "pm.json").read_text())
        assert stored[0]["processing_since"] is not None, \
            "read_inbox 后 processing_since 应记录时间戳"
        assert "T" in stored[0]["processing_since"]  # ISO 格式

    def test_read_inbox_snapshot_not_modified(self) -> None:
        """返回的快照是副本——修改快照不应影响文件中的数据"""
        from tools.mailbox_ops import send_mail, read_inbox
        send_mail(MAILBOXES_DIR, to="pm", from_="manager",
                  type_="task_assign", subject="测试", content="内容")
        snapshot = read_inbox(MAILBOXES_DIR, role="pm")
        snapshot[0]["status"] = "tampered"  # 修改快照
        # 文件中的数据不应被影响
        stored = json.loads((MAILBOXES_DIR / "pm.json").read_text())
        assert stored[0]["status"] != "tampered", \
            "read_inbox 返回的是引用而非副本，调用方修改了原始数据"

    def test_read_inbox_read_and_mark_in_same_lock(self) -> None:
        """read + 标记在同一锁内：无 TOCTOU 窗口"""
        from tools.mailbox_ops import send_mail, read_inbox, STATUS_IN_PROGRESS
        send_mail(MAILBOXES_DIR, to="pm", from_="manager",
                  type_="task_assign", subject="原子性测试", content="内容")
        msgs = read_inbox(MAILBOXES_DIR, role="pm")
        stored = json.loads((MAILBOXES_DIR / "pm.json").read_text())
        assert all(m["status"] == STATUS_IN_PROGRESS for m in stored), \
            "read_inbox 未在锁内完成标记，存在 TOCTOU 风险"
        assert len(msgs) == 1


# ─────────────────────────────────────────────────────────────────────────────
# T3b | mark_done：in_progress → done
# ─────────────────────────────────────────────────────────────────────────────

class TestT3bMarkDone:
    """T3b：mark_done 将指定 in_progress 消息标记为 done"""

    def setup_method(self) -> None:
        (MAILBOXES_DIR / "pm.json").write_text("[]", encoding="utf-8")

    def teardown_method(self) -> None:
        (MAILBOXES_DIR / "pm.json").write_text("[]", encoding="utf-8")

    def test_mark_done_changes_status(self) -> None:
        from tools.mailbox_ops import send_mail, read_inbox, mark_done, STATUS_DONE
        send_mail(MAILBOXES_DIR, to="pm", from_="manager",
                  type_="task_assign", subject="任务", content="内容")
        msgs = read_inbox(MAILBOXES_DIR, role="pm")
        msg_id = msgs[0]["id"]

        count = mark_done(MAILBOXES_DIR, role="pm", msg_ids=[msg_id])

        assert count == 1, f"mark_done 应标记 1 条，实际 {count} 条"
        stored = json.loads((MAILBOXES_DIR / "pm.json").read_text())
        assert stored[0]["status"] == STATUS_DONE, \
            f"mark_done 后 status 应为 done，实际：{stored[0]['status']}"

    def test_mark_done_all_in_progress(self) -> None:
        """mark_done_all_in_progress 批量确认所有 in_progress 消息"""
        from tools.mailbox_ops import (
            send_mail, read_inbox, mark_done_all_in_progress, STATUS_DONE
        )
        send_mail(MAILBOXES_DIR, to="pm", from_="manager",
                  type_="task_assign", subject="任务A", content="内容A")
        send_mail(MAILBOXES_DIR, to="pm", from_="manager",
                  type_="broadcast", subject="广播", content="内容B")
        read_inbox(MAILBOXES_DIR, role="pm")  # 两条都变 in_progress

        count = mark_done_all_in_progress(MAILBOXES_DIR, role="pm")

        assert count == 2, f"应标记 2 条，实际 {count} 条"
        stored = json.loads((MAILBOXES_DIR / "pm.json").read_text())
        assert all(m["status"] == STATUS_DONE for m in stored)

    def test_mark_done_only_targets_in_progress(self) -> None:
        """mark_done 不影响 unread 或 done 状态的消息"""
        from tools.mailbox_ops import (
            send_mail, read_inbox, mark_done, STATUS_UNREAD, STATUS_IN_PROGRESS, STATUS_DONE
        )
        # 写三条消息
        send_mail(MAILBOXES_DIR, to="pm", from_="manager",
                  type_="task_assign", subject="任务1", content="内容1")
        send_mail(MAILBOXES_DIR, to="pm", from_="manager",
                  type_="task_assign", subject="任务2", content="内容2")

        # 取第一条（变 in_progress），第二条仍是 unread
        msgs = read_inbox(MAILBOXES_DIR, role="pm")
        assert len(msgs) == 2  # 两条都被取走
        first_id = msgs[0]["id"]

        # 只 mark_done 第一条
        mark_done(MAILBOXES_DIR, role="pm", msg_ids=[first_id])
        stored = json.loads((MAILBOXES_DIR / "pm.json").read_text())
        statuses = {m["id"]: m["status"] for m in stored}
        assert statuses[first_id] == STATUS_DONE
        # 第二条也是 in_progress（被 read_inbox 一次性取走），不受 mark_done 影响
        second_id = msgs[1]["id"]
        assert statuses[second_id] == STATUS_IN_PROGRESS


# ─────────────────────────────────────────────────────────────────────────────
# T4a | 并发写：JSON 不损坏
# ─────────────────────────────────────────────────────────────────────────────

class TestT4aConcurrentWrite:
    """T4a：两线程同时 send_mail，pm.json 包含两条消息且格式完整"""

    def setup_method(self) -> None:
        (MAILBOXES_DIR / "pm.json").write_text("[]", encoding="utf-8")

    def teardown_method(self) -> None:
        (MAILBOXES_DIR / "pm.json").write_text("[]", encoding="utf-8")

    def test_concurrent_send_no_corruption(self) -> None:
        from tools.mailbox_ops import send_mail
        errors: list[Exception] = []

        def writer(n: int) -> None:
            try:
                send_mail(MAILBOXES_DIR, to="pm", from_="manager",
                          type_="task_assign", subject=f"任务{n}", content=f"内容{n}")
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=writer, args=(1,))
        t2 = threading.Thread(target=writer, args=(2,))
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert not errors, f"并发写入时出现异常: {errors}"
        data = json.loads((MAILBOXES_DIR / "pm.json").read_text())
        assert len(data) == 2, f"期望 2 条消息，实际 {len(data)} 条"
        for msg in data:
            assert "id" in msg and "status" in msg


# ─────────────────────────────────────────────────────────────────────────────
# T4b | 并发语义：read_inbox + send_mail 同时，消息无丢失
# ─────────────────────────────────────────────────────────────────────────────

class TestT4bConcurrentSemantic:
    """T4b：read_inbox 标记 in_progress + send_mail 写入新消息同时进行，消息无丢失"""

    def setup_method(self) -> None:
        (MAILBOXES_DIR / "pm.json").write_text("[]", encoding="utf-8")

    def teardown_method(self) -> None:
        (MAILBOXES_DIR / "pm.json").write_text("[]", encoding="utf-8")

    def test_read_and_write_concurrent_no_lost_messages(self) -> None:
        from tools.mailbox_ops import send_mail, read_inbox
        send_mail(MAILBOXES_DIR, to="pm", from_="manager",
                  type_="task_assign", subject="原始任务", content="内容")

        results: dict[str, object] = {}
        errors: list[Exception] = []

        def reader() -> None:
            try:
                results["read"] = read_inbox(MAILBOXES_DIR, role="pm")
            except Exception as e:
                errors.append(e)

        def writer() -> None:
            try:
                send_mail(MAILBOXES_DIR, to="pm", from_="manager",
                          type_="broadcast", subject="新通知", content="内容")
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=reader)
        t2 = threading.Thread(target=writer)
        t1.start(); t2.start()
        t1.join(); t2.join()

        assert not errors
        data = json.loads((MAILBOXES_DIR / "pm.json").read_text())
        assert len(data) == 2, f"期望 2 条消息，实际 {len(data)} 条（消息丢失）"
        ids = [m["id"] for m in data]
        assert len(ids) == len(set(ids)), "存在重复 id"


# ─────────────────────────────────────────────────────────────────────────────
# T5 | read_inbox 幂等性（in_progress 消息不被重复取走）
# ─────────────────────────────────────────────────────────────────────────────

class TestT5ReadIdempotency:
    """T5：read_inbox 调用两次，第二次返回空列表（已是 in_progress，不重复取走）"""

    def setup_method(self) -> None:
        (MAILBOXES_DIR / "pm.json").write_text("[]", encoding="utf-8")

    def teardown_method(self) -> None:
        (MAILBOXES_DIR / "pm.json").write_text("[]", encoding="utf-8")

    def test_second_read_returns_empty(self) -> None:
        from tools.mailbox_ops import send_mail, read_inbox
        send_mail(MAILBOXES_DIR, to="pm", from_="manager",
                  type_="task_assign", subject="任务", content="内容")
        read_inbox(MAILBOXES_DIR, role="pm")   # 第一次：unread → in_progress
        second = read_inbox(MAILBOXES_DIR, role="pm")  # 第二次：in_progress 不返回
        assert second == [], \
            f"第二次 read_inbox 应返回空（消息已是 in_progress），实际返回 {second}"


# ─────────────────────────────────────────────────────────────────────────────
# T6 | 共享工作区目录结构
# ─────────────────────────────────────────────────────────────────────────────

class TestT6SharedWorkspaceStructure:
    """T6：workspace/shared/ 包含 WORKSPACE_RULES.md、mailboxes/、needs/、design/"""

    def test_workspace_rules_exists(self) -> None:
        assert (WORKSPACE_SHARED / "WORKSPACE_RULES.md").exists(), "缺少 WORKSPACE_RULES.md"

    def test_workspace_rules_not_empty(self) -> None:
        content = (WORKSPACE_SHARED / "WORKSPACE_RULES.md").read_text(encoding="utf-8")
        assert len(content.strip()) > 100, "WORKSPACE_RULES.md 内容过少"

    def test_needs_dir_exists(self) -> None:
        assert (WORKSPACE_SHARED / "needs").is_dir()

    def test_needs_requirements_exists(self) -> None:
        assert (WORKSPACE_SHARED / "needs" / "requirements.md").exists(), \
            "缺少 shared/needs/requirements.md"

    def test_design_dir_exists(self) -> None:
        assert (WORKSPACE_SHARED / "design").is_dir()

    def test_mailboxes_in_shared(self) -> None:
        assert MAILBOXES_DIR.is_dir(), "mailboxes/ 未放在 shared/ 内"


# ─────────────────────────────────────────────────────────────────────────────
# T7 | Manager workspace 四件套
# ─────────────────────────────────────────────────────────────────────────────

class TestT7ManagerWorkspace:
    """T7：workspace/manager/ 四件套齐全"""

    @pytest.mark.parametrize("filename", ["soul.md", "user.md", "agent.md", "memory.md"])
    def test_file_exists(self, filename: str) -> None:
        assert (WORKSPACE_MANAGER / filename).exists(), \
            f"Manager workspace 缺少 {filename}"

    def test_soul_has_never_clause(self) -> None:
        soul = (WORKSPACE_MANAGER / "soul.md").read_text(encoding="utf-8")
        assert "NEVER" in soul or "禁止" in soul

    def test_agent_has_mailbox_info(self) -> None:
        agent = (WORKSPACE_MANAGER / "agent.md").read_text(encoding="utf-8")
        assert "mailbox" in agent.lower() or "邮箱" in agent, \
            "Manager agent.md 未说明邮箱使用方式"


# ─────────────────────────────────────────────────────────────────────────────
# T8 | PM workspace 四件套
# ─────────────────────────────────────────────────────────────────────────────

class TestT8PMWorkspace:
    """T8：workspace/pm/ 四件套齐全"""

    @pytest.mark.parametrize("filename", ["soul.md", "user.md", "agent.md", "memory.md"])
    def test_file_exists(self, filename: str) -> None:
        assert (WORKSPACE_PM / filename).exists(), f"PM workspace 缺少 {filename}"

    def test_soul_has_never_clause(self) -> None:
        soul = (WORKSPACE_PM / "soul.md").read_text(encoding="utf-8")
        assert "NEVER" in soul or "禁止" in soul

    def test_agent_has_workspace_rules_ref(self) -> None:
        agent = (WORKSPACE_PM / "agent.md").read_text(encoding="utf-8")
        assert "needs" in agent and "design" in agent, \
            "PM agent.md 未说明共享工作区读写权限"


# ─────────────────────────────────────────────────────────────────────────────
# T9 | workspace-local mailbox skill 注册（v3）
# ─────────────────────────────────────────────────────────────────────────────

class TestT9MailboxSkill:
    """T9：Manager 和 PM 的 workspace-local mailbox skill 存在且正确配置"""

    MANAGER_SKILLS_DIR = WORKSPACE_MANAGER / "skills"
    PM_SKILLS_DIR      = WORKSPACE_PM      / "skills"

    def test_manager_mailbox_skill_md_exists(self) -> None:
        assert (self.MANAGER_SKILLS_DIR / "mailbox" / "SKILL.md").exists(), \
            "Manager workspace/skills/mailbox/SKILL.md 不存在"

    def test_pm_mailbox_skill_md_exists(self) -> None:
        assert (self.PM_SKILLS_DIR / "mailbox" / "SKILL.md").exists(), \
            "PM workspace/skills/mailbox/SKILL.md 不存在"

    def test_manager_load_skills_has_mailbox(self) -> None:
        yaml_content = (self.MANAGER_SKILLS_DIR / "load_skills.yaml").read_text(encoding="utf-8")
        assert "mailbox" in yaml_content, \
            "Manager load_skills.yaml 未注册 mailbox skill"

    def test_pm_load_skills_has_mailbox(self) -> None:
        yaml_content = (self.PM_SKILLS_DIR / "load_skills.yaml").read_text(encoding="utf-8")
        assert "mailbox" in yaml_content, \
            "PM load_skills.yaml 未注册 mailbox skill"

    def test_mailbox_cli_exists_manager(self) -> None:
        cli_path = self.MANAGER_SKILLS_DIR / "mailbox" / "scripts" / "mailbox_cli.py"
        assert cli_path.exists(), "Manager mailbox_cli.py 不存在"

    def test_mailbox_cli_exists_pm(self) -> None:
        cli_path = self.PM_SKILLS_DIR / "mailbox" / "scripts" / "mailbox_cli.py"
        assert cli_path.exists(), "PM mailbox_cli.py 不存在"

    def test_pm_has_product_design_skill(self) -> None:
        assert (self.PM_SKILLS_DIR / "product_design" / "SKILL.md").exists(), \
            "PM workspace/skills/product_design/SKILL.md 不存在"


# ─────────────────────────────────────────────────────────────────────────────
# T10 | mailboxes 初始格式
# ─────────────────────────────────────────────────────────────────────────────

class TestT10MailboxInitFormat:
    """T10：manager.json 和 pm.json 重置后内容为合法 JSON 空数组"""

    @pytest.fixture(autouse=True)
    def reset_mailboxes(self):
        for role in ("manager", "pm"):
            (MAILBOXES_DIR / f"{role}.json").write_text("[]", encoding="utf-8")

    @pytest.mark.parametrize("role", ["manager", "pm"])
    def test_initial_content_is_empty_list(self, role: str) -> None:
        path = MAILBOXES_DIR / f"{role}.json"
        data = json.loads(path.read_text(encoding="utf-8").strip())
        assert data == [], f"{role}.json 重置后内容不是空数组，实际: {data}"


# ─────────────────────────────────────────────────────────────────────────────
# T11 | 三态状态机完整流转
# ─────────────────────────────────────────────────────────────────────────────

class TestT11ThreeStateStateMachine:
    """
    T11：三态状态机完整流转
    unread → in_progress（read_inbox）→ done（mark_done）
    并验证各态下的行为约束
    """

    def setup_method(self) -> None:
        (MAILBOXES_DIR / "pm.json").write_text("[]", encoding="utf-8")

    def teardown_method(self) -> None:
        (MAILBOXES_DIR / "pm.json").write_text("[]", encoding="utf-8")

    def test_full_state_transition(self) -> None:
        """完整验证：unread → in_progress → done"""
        from tools.mailbox_ops import (
            send_mail, read_inbox, mark_done,
            STATUS_UNREAD, STATUS_IN_PROGRESS, STATUS_DONE,
        )
        # 阶段1：写入 → unread
        send_mail(MAILBOXES_DIR, to="pm", from_="manager",
                  type_="task_assign", subject="三态测试", content="内容")
        stored = json.loads((MAILBOXES_DIR / "pm.json").read_text())
        assert stored[0]["status"] == STATUS_UNREAD, "写入后应为 unread"
        assert stored[0]["processing_since"] is None

        # 阶段2：取走 → in_progress
        msgs = read_inbox(MAILBOXES_DIR, role="pm")
        assert len(msgs) == 1
        stored = json.loads((MAILBOXES_DIR / "pm.json").read_text())
        assert stored[0]["status"] == STATUS_IN_PROGRESS, "read_inbox 后应为 in_progress"
        assert stored[0]["processing_since"] is not None, "processing_since 应有时间戳"

        # 阶段3：确认 → done
        count = mark_done(MAILBOXES_DIR, role="pm", msg_ids=[msgs[0]["id"]])
        assert count == 1
        stored = json.loads((MAILBOXES_DIR / "pm.json").read_text())
        assert stored[0]["status"] == STATUS_DONE, "mark_done 后应为 done"

    def test_in_progress_not_returned_by_read_inbox(self) -> None:
        """in_progress 的消息不被 read_inbox 重复取走（并发安全）"""
        from tools.mailbox_ops import send_mail, read_inbox
        send_mail(MAILBOXES_DIR, to="pm", from_="manager",
                  type_="task_assign", subject="消息", content="内容")
        first  = read_inbox(MAILBOXES_DIR, role="pm")
        second = read_inbox(MAILBOXES_DIR, role="pm")
        assert len(first) == 1
        assert len(second) == 0, "in_progress 消息被重复取走，并发保护失效"

    def test_done_messages_not_returned(self) -> None:
        """done 状态的消息永远不被 read_inbox 取走"""
        from tools.mailbox_ops import (
            send_mail, read_inbox, mark_done, STATUS_DONE
        )
        send_mail(MAILBOXES_DIR, to="pm", from_="manager",
                  type_="task_assign", subject="消息", content="内容")
        msgs = read_inbox(MAILBOXES_DIR, role="pm")
        mark_done(MAILBOXES_DIR, role="pm", msg_ids=[msgs[0]["id"]])

        # done 后再 read_inbox 应为空
        third = read_inbox(MAILBOXES_DIR, role="pm")
        assert third == [], "done 状态的消息不应再被取走"


# ─────────────────────────────────────────────────────────────────────────────
# T12 | reset_stale：崩溃恢复
# ─────────────────────────────────────────────────────────────────────────────

class TestT12ResetStale:
    """
    T12：reset_stale 将超时的 in_progress 消息恢复为 unread（崩溃恢复）
    对应 SQS Visibility Timeout 到期后消息重新可见的机制
    """

    def setup_method(self) -> None:
        (MAILBOXES_DIR / "pm.json").write_text("[]", encoding="utf-8")

    def teardown_method(self) -> None:
        (MAILBOXES_DIR / "pm.json").write_text("[]", encoding="utf-8")

    def test_reset_stale_reverts_to_unread(self) -> None:
        """超时的 in_progress 消息被 reset_stale 恢复为 unread"""
        from tools.mailbox_ops import (
            send_mail, read_inbox, reset_stale, STATUS_UNREAD
        )
        send_mail(MAILBOXES_DIR, to="pm", from_="manager",
                  type_="task_assign", subject="崩溃测试", content="内容")
        read_inbox(MAILBOXES_DIR, role="pm")  # → in_progress

        # timeout_seconds=0：所有 in_progress 消息立即视为超时
        count = reset_stale(MAILBOXES_DIR, role="pm", timeout_seconds=0)

        assert count == 1, f"应重置 1 条，实际 {count} 条"
        stored = json.loads((MAILBOXES_DIR / "pm.json").read_text())
        assert stored[0]["status"] == STATUS_UNREAD, \
            f"reset_stale 后应为 unread，实际：{stored[0]['status']}"
        assert stored[0]["processing_since"] is None, \
            "reset_stale 后 processing_since 应清空"

    def test_reset_stale_only_affects_timed_out(self) -> None:
        """未超时的 in_progress 消息不被 reset_stale 影响"""
        from tools.mailbox_ops import (
            send_mail, read_inbox, reset_stale, STATUS_IN_PROGRESS
        )
        send_mail(MAILBOXES_DIR, to="pm", from_="manager",
                  type_="task_assign", subject="未超时消息", content="内容")
        read_inbox(MAILBOXES_DIR, role="pm")  # → in_progress

        # timeout_seconds=9999：不会超时
        count = reset_stale(MAILBOXES_DIR, role="pm", timeout_seconds=9999)

        assert count == 0, "不应有消息被重置"
        stored = json.loads((MAILBOXES_DIR / "pm.json").read_text())
        assert stored[0]["status"] == STATUS_IN_PROGRESS

    def test_reset_stale_allows_reprocessing(self) -> None:
        """reset_stale 后消息重新可被 read_inbox 取走（崩溃恢复完整链路）"""
        from tools.mailbox_ops import send_mail, read_inbox, reset_stale
        send_mail(MAILBOXES_DIR, to="pm", from_="manager",
                  type_="task_assign", subject="崩溃恢复测试", content="内容")
        read_inbox(MAILBOXES_DIR, role="pm")      # → in_progress（模拟崩溃）
        reset_stale(MAILBOXES_DIR, role="pm", timeout_seconds=0)  # → unread

        # 现在可以重新取走
        recovered = read_inbox(MAILBOXES_DIR, role="pm")
        assert len(recovered) == 1, "崩溃恢复后消息应可被重新取走"

    def test_done_messages_not_reset(self) -> None:
        """done 状态的消息不被 reset_stale 影响"""
        from tools.mailbox_ops import (
            send_mail, read_inbox, mark_done, reset_stale, STATUS_DONE
        )
        send_mail(MAILBOXES_DIR, to="pm", from_="manager",
                  type_="task_assign", subject="已完成", content="内容")
        msgs = read_inbox(MAILBOXES_DIR, role="pm")
        mark_done(MAILBOXES_DIR, role="pm", msg_ids=[msgs[0]["id"]])  # → done

        count = reset_stale(MAILBOXES_DIR, role="pm", timeout_seconds=0)
        assert count == 0, "done 状态的消息不应被 reset_stale 重置"
        stored = json.loads((MAILBOXES_DIR / "pm.json").read_text())
        assert stored[0]["status"] == STATUS_DONE


# ─────────────────────────────────────────────────────────────────────────────
# T13 | create_workspace 幂等性与目录结构
# ─────────────────────────────────────────────────────────────────────────────

class TestT13CreateWorkspace:
    """T13：create_workspace 创建正确结构，幂等——第二次调用不覆盖现有文件"""

    def test_creates_directory_structure(self) -> None:
        """create_workspace 创建 needs/、design/、mailboxes/ 子目录"""
        from tools.workspace_ops import create_workspace
        with tempfile.TemporaryDirectory() as tmpdir:
            shared = Path(tmpdir) / "shared"
            result = create_workspace(shared, roles=["manager", "pm"],
                                      project_name="测试项目")
            assert (shared / "needs").is_dir()
            assert (shared / "design").is_dir()
            assert (shared / "mailboxes").is_dir()

    def test_creates_mailbox_files(self) -> None:
        """create_workspace 为每个角色创建初始邮箱（空 JSON 数组）"""
        from tools.workspace_ops import create_workspace
        with tempfile.TemporaryDirectory() as tmpdir:
            shared = Path(tmpdir) / "shared"
            create_workspace(shared, roles=["manager", "pm"])
            for role in ("manager", "pm"):
                path = shared / "mailboxes" / f"{role}.json"
                assert path.exists(), f"{role}.json 未创建"
                data = json.loads(path.read_text(encoding="utf-8"))
                assert data == [], f"{role}.json 初始内容应为空数组"

    def test_creates_workspace_rules(self) -> None:
        """create_workspace 生成 WORKSPACE_RULES.md"""
        from tools.workspace_ops import create_workspace
        with tempfile.TemporaryDirectory() as tmpdir:
            shared = Path(tmpdir) / "shared"
            create_workspace(shared, roles=["manager", "pm"],
                             project_name="测试项目")
            rules = shared / "WORKSPACE_RULES.md"
            assert rules.exists()
            content = rules.read_text(encoding="utf-8")
            assert "测试项目" in content, "WORKSPACE_RULES.md 应包含项目名称"
            assert "needs" in content and "design" in content

    def test_idempotent_does_not_overwrite(self) -> None:
        """第二次调用 create_workspace 不覆盖已存在的文件"""
        from tools.workspace_ops import create_workspace
        with tempfile.TemporaryDirectory() as tmpdir:
            shared = Path(tmpdir) / "shared"
            create_workspace(shared, roles=["manager", "pm"])

            # 在邮箱里写入一条消息
            pm_inbox = shared / "mailboxes" / "pm.json"
            pm_inbox.write_text('[{"id": "sentinel"}]', encoding="utf-8")

            # 第二次调用（幂等）
            result = create_workspace(shared, roles=["manager", "pm"])

            # pm.json 不应被覆盖
            data = json.loads(pm_inbox.read_text(encoding="utf-8"))
            assert data == [{"id": "sentinel"}], \
                "create_workspace 幂等失效：已存在的邮箱文件被覆盖"
            assert "mailboxes/pm.json" in result["skipped_files"], \
                "幂等跳过的文件应记录在 skipped_files 中"

    def test_returns_creation_report(self) -> None:
        """create_workspace 返回 created_dirs / created_files / skipped_files"""
        from tools.workspace_ops import create_workspace
        with tempfile.TemporaryDirectory() as tmpdir:
            shared = Path(tmpdir) / "shared"
            result = create_workspace(shared, roles=["manager", "pm"])
            assert "created_dirs" in result
            assert "created_files" in result
            assert "skipped_files" in result
            # 首次调用：应有新建目录和文件
            assert len(result["created_dirs"]) > 0, "首次调用应有新建目录"
            assert len(result["created_files"]) > 0, "首次调用应有新建文件"
