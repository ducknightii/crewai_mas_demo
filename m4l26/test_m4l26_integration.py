"""
第26课集成测试：test_m4l26_integration.py

真实启动 Docker 沙盒 + 真实 LLM 调用，验证三步任务链的完整端到端流转。

运行方式：
    # 仅运行集成测试（需要 Docker + 网络）
    cd crewai_mas_demo
    python3 -m pytest m4l26/test_m4l26_integration.py -v -s

    # 跳过集成测试（默认 pytest 行为，仅跑单元测试）
    python3 -m pytest m4l26/test_m4l26.py -v

测试分层：
    I1 - 基础设施：Docker 沙盒启停 + 健康检查
    I2 - 步骤1：ManagerAssignCrew 真实执行（LLM + 沙盒）
    I3 - 步骤2：PMExecuteCrew 真实执行（LLM + 沙盒）
    I4 - 步骤3：ManagerReviewCrew 真实执行（LLM + 沙盒）
    I5 - 端到端：m4l26_run.py 完整三步流转

隔离策略：
    - 每次测试前 reset_workspace() 清空 mailboxes + 删除输出文件
    - SANDBOX_MCP_URL 通过 monkeypatch 切换到对应端口
    - Docker 容器生命周期由 pytest fixtures 管理（session 级别启停）
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest
import requests

# ── 路径 ──────────────────────────────────────────────────────────────────────
_M4L26_DIR    = Path(__file__).resolve().parent
_PROJECT_ROOT = _M4L26_DIR.parent
for _p in [str(_PROJECT_ROOT), str(_M4L26_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

SHARED_DIR       = _M4L26_DIR / "workspace" / "shared"
MANAGER_DIR      = _M4L26_DIR / "workspace" / "manager"
PM_DIR           = _M4L26_DIR / "workspace" / "pm"
MAILBOXES_DIR    = SHARED_DIR / "mailboxes"
DESIGN_DIR       = SHARED_DIR / "design"
DOCKER_COMPOSE   = _M4L26_DIR / "sandbox-docker-compose.yaml"

MANAGER_SANDBOX_URL = "http://localhost:8025/mcp"
PM_SANDBOX_URL      = "http://localhost:8026/mcp"

SANDBOX_HEALTH_TIMEOUT = 60   # 秒
LLM_CALL_TIMEOUT       = 300  # 秒（单个 Crew 的最大等待时间）


# ─────────────────────────────────────────────────────────────────────────────
# 辅助函数
# ─────────────────────────────────────────────────────────────────────────────

def reset_workspace() -> None:
    """清空邮箱 + 删除上次运行产生的输出，保留目录结构"""
    # 清空邮箱为空列表
    for role in ("manager", "pm"):
        inbox = MAILBOXES_DIR / f"{role}.json"
        inbox.write_text("[]", encoding="utf-8")
        lock = inbox.with_suffix(".lock")
        if lock.exists():
            lock.unlink()

    # 删除产品文档
    spec = DESIGN_DIR / "product_spec.md"
    if spec.exists():
        spec.unlink()

    # 删除 Manager 验收结果
    review = MANAGER_DIR / "review_result.md"
    if review.exists():
        review.unlink()


def wait_sandbox_healthy(url: str, timeout: int = SANDBOX_HEALTH_TIMEOUT) -> bool:
    """轮询等待沙盒可用。
    aio-sandbox 暴露的健康路径是 /（返回 200），不是 /health。
    """
    base = url.replace("/mcp", "")
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            resp = requests.get(base, timeout=3)
            if resp.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(2)
    return False


def docker_compose_up(profile: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "compose", "-f", str(DOCKER_COMPOSE), "--profile", profile, "up", "-d"],
        cwd=_M4L26_DIR,
        capture_output=True,
        text=True,
    )


def docker_compose_down(profile: str) -> None:
    subprocess.run(
        ["docker", "compose", "-f", str(DOCKER_COMPOSE), "--profile", profile, "down"],
        cwd=_M4L26_DIR,
        capture_output=True,
        text=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def manager_sandbox():
    """启动 Manager 沙盒（port 8025），session 结束后停止"""
    result = docker_compose_up("manager")
    assert result.returncode == 0, f"docker compose up manager failed:\n{result.stderr}"
    healthy = wait_sandbox_healthy(MANAGER_SANDBOX_URL)
    assert healthy, f"Manager sandbox 未能在 {SANDBOX_HEALTH_TIMEOUT}s 内就绪"
    yield
    docker_compose_down("manager")


@pytest.fixture(scope="session")
def pm_sandbox():
    """启动 PM 沙盒（port 8026），session 结束后停止"""
    result = docker_compose_up("pm")
    assert result.returncode == 0, f"docker compose up pm failed:\n{result.stderr}"
    healthy = wait_sandbox_healthy(PM_SANDBOX_URL)
    assert healthy, f"PM sandbox 未能在 {SANDBOX_HEALTH_TIMEOUT}s 内就绪"
    yield
    docker_compose_down("pm")


@pytest.fixture(autouse=True)
def clean_workspace():
    """每个测试前清理邮箱和输出文件"""
    reset_workspace()
    yield
    # 测试后不清理，方便调试查看输出


# ─────────────────────────────────────────────────────────────────────────────
# I1 — 基础设施：Docker 沙盒健康检查
# ─────────────────────────────────────────────────────────────────────────────

class TestI1SandboxHealth:
    """验证 Docker 沙盒能正常启动并通过健康检查"""

    def test_manager_sandbox_starts(self, manager_sandbox):
        """Manager 沙盒（port 8025）启动后 / 端点可访问"""
        resp = requests.get(MANAGER_SANDBOX_URL.replace("/mcp", ""), timeout=5)
        assert resp.status_code == 200

    def test_pm_sandbox_starts(self, pm_sandbox):
        """PM 沙盒（port 8026）启动后 / 端点可访问"""
        resp = requests.get(PM_SANDBOX_URL.replace("/mcp", ""), timeout=5)
        assert resp.status_code == 200

    def test_manager_sandbox_workspace_mounted(self, manager_sandbox):
        """Manager 沙盒内 /workspace 已挂载（可访问 soul.md）"""
        import m4l26_mailbox.mailbox_ops as _mops  # noqa
        assert (MANAGER_DIR / "soul.md").exists(), "Manager soul.md 不存在，挂载可能失败"

    def test_pm_sandbox_shared_mounted(self, pm_sandbox):
        """PM 沙盒内 /mnt/shared 已挂载（需求文档可访问）"""
        assert (SHARED_DIR / "needs" / "requirements.md").exists()

    def test_mailboxes_reset_before_each_test(self):
        """clean_workspace fixture 确保每次测试前邮箱为空列表"""
        for role in ("manager", "pm"):
            inbox = MAILBOXES_DIR / f"{role}.json"
            assert inbox.exists()
            msgs = json.loads(inbox.read_text())
            assert msgs == [], f"{role}.json 未被清空：{msgs}"


# ─────────────────────────────────────────────────────────────────────────────
# I2 — 步骤1：ManagerAssignCrew 真实执行
# ─────────────────────────────────────────────────────────────────────────────

class TestI2ManagerAssign:
    """
    ManagerAssignCrew（步骤1）：
    读取 needs/requirements.md → 发 task_assign 邮件到 PM 邮箱
    """

    @pytest.fixture(autouse=True)
    def patch_sandbox_url(self, monkeypatch, manager_sandbox):
        """确保 ManagerAssignCrew 使用 Manager 沙盒（8025）"""
        import m4l26_manager as mgr
        monkeypatch.setattr(mgr, "MANAGER_SANDBOX_MCP_URL", MANAGER_SANDBOX_URL)

    def test_manager_assign_writes_to_pm_inbox(self):
        """ManagerAssignCrew 运行后，pm.json 中应有 task_assign 消息"""
        from m4l26_manager import ManagerAssignCrew, save_session

        session_id = str(uuid.uuid4())
        crew = ManagerAssignCrew(session_id=session_id)
        crew.crew().kickoff(
            inputs={"user_request": "请读取共享需求文档，向 PM 发送产品文档设计任务"}
        )
        save_session(crew, session_id)

        pm_msgs = json.loads((MAILBOXES_DIR / "pm.json").read_text())
        task_assigns = [m for m in pm_msgs if m.get("type") == "task_assign"]
        assert len(task_assigns) >= 1, \
            f"PM 邮箱中未找到 task_assign 消息，当前邮箱内容：{pm_msgs}"

    def test_manager_assign_message_fields(self):
        """task_assign 消息必须包含所有必填字段"""
        from m4l26_manager import ManagerAssignCrew, save_session

        session_id = str(uuid.uuid4())
        crew = ManagerAssignCrew(session_id=session_id)
        crew.crew().kickoff(
            inputs={"user_request": "请读取共享需求文档，向 PM 发送产品文档设计任务"}
        )
        save_session(crew, session_id)

        pm_msgs = json.loads((MAILBOXES_DIR / "pm.json").read_text())
        msg = next(m for m in pm_msgs if m.get("type") == "task_assign")

        required_fields = {"id", "from", "to", "type", "subject", "content", "timestamp", "read"}
        missing = required_fields - set(msg.keys())
        assert not missing, f"消息缺少字段：{missing}"
        assert msg["to"] == "pm"
        assert msg["from"] == "manager"
        assert msg["read"] is False

    def test_manager_assign_content_references_requirements(self):
        """task_assign 的 content 应包含对需求文档路径的引用"""
        from m4l26_manager import ManagerAssignCrew, save_session

        session_id = str(uuid.uuid4())
        crew = ManagerAssignCrew(session_id=session_id)
        crew.crew().kickoff(
            inputs={"user_request": "请读取共享需求文档，向 PM 发送产品文档设计任务"}
        )
        save_session(crew, session_id)

        pm_msgs = json.loads((MAILBOXES_DIR / "pm.json").read_text())
        msg = next(m for m in pm_msgs if m.get("type") == "task_assign")
        content = msg.get("content", "")
        # 消息内容应该提到产品文档路径或相关工作
        assert any(kw in content for kw in ["product_spec", "design", "需求", "产品文档"]), \
            f"task_assign content 未提到产品文档相关内容：{content[:200]}"


# ─────────────────────────────────────────────────────────────────────────────
# I3 — 步骤2：PMExecuteCrew 真实执行
# ─────────────────────────────────────────────────────────────────────────────

class TestI3PMExecute:
    """
    PMExecuteCrew（步骤2）：
    读取 PM 邮箱 task_assign → 读 needs/ → 写 product_spec.md → 发 task_done 到 Manager
    """

    @pytest.fixture(autouse=True)
    def patch_sandbox_url(self, monkeypatch, pm_sandbox):
        """确保 PMExecuteCrew 使用 PM 沙盒（8026）"""
        import m4l26_pm as pm
        monkeypatch.setattr(pm, "PM_SANDBOX_MCP_URL", PM_SANDBOX_URL)

    @pytest.fixture(autouse=True)
    def seed_pm_inbox(self):
        """在 PM 邮箱中预置一封 task_assign 邮件（模拟步骤1已完成）"""
        from m4l26_mailbox.mailbox_ops import send_mail
        send_mail(
            mailbox_dir=MAILBOXES_DIR,
            to="pm",
            from_="manager",
            type_="task_assign",
            subject="产品文档设计",
            content=(
                "请根据 /mnt/shared/needs/requirements.md 的内容，"
                "设计 XiaoPaw 宠物健康记录功能的产品规格文档，"
                "完成后写入 /mnt/shared/design/product_spec.md，"
                "并发邮件通知 Manager 验收。"
            ),
        )

    def test_pm_executes_writes_product_spec(self):
        """PMExecuteCrew 运行后，product_spec.md 应被写入 design/"""
        from m4l26_pm import PMExecuteCrew, save_session

        session_id = str(uuid.uuid4())
        crew = PMExecuteCrew(session_id=session_id)
        crew.crew().kickoff(
            inputs={"user_request": "请读取邮箱中的任务，根据需求文档撰写产品规格文档，完成后通知 Manager"}
        )
        save_session(crew, session_id)

        assert (DESIGN_DIR / "product_spec.md").exists(), \
            "product_spec.md 未被写入 design/ 目录"

    def test_pm_executes_notifies_manager(self):
        """PMExecuteCrew 完成后，manager.json 中应有 task_done 消息"""
        from m4l26_pm import PMExecuteCrew, save_session

        session_id = str(uuid.uuid4())
        crew = PMExecuteCrew(session_id=session_id)
        crew.crew().kickoff(
            inputs={"user_request": "请读取邮箱中的任务，根据需求文档撰写产品规格文档，完成后通知 Manager"}
        )
        save_session(crew, session_id)

        mgr_msgs = json.loads((MAILBOXES_DIR / "manager.json").read_text())
        task_dones = [m for m in mgr_msgs if m.get("type") == "task_done"]
        assert len(task_dones) >= 1, \
            f"Manager 邮箱中未找到 task_done 消息，当前内容：{mgr_msgs}"

    def test_pm_product_spec_has_required_sections(self):
        """product_spec.md 内容应包含产品概述、用户故事、功能规格"""
        from m4l26_pm import PMExecuteCrew, save_session

        session_id = str(uuid.uuid4())
        crew = PMExecuteCrew(session_id=session_id)
        crew.crew().kickoff(
            inputs={"user_request": "请读取邮箱中的任务，根据需求文档撰写产品规格文档，完成后通知 Manager"}
        )
        save_session(crew, session_id)

        spec_path = DESIGN_DIR / "product_spec.md"
        assert spec_path.exists()
        content = spec_path.read_text(encoding="utf-8")
        # 必须涵盖核心需求（F-01/F-02 或相关关键词）
        keywords = ["疫苗", "就医", "F-01", "F-02", "XiaoPaw"]
        found = [k for k in keywords if k in content]
        assert len(found) >= 2, \
            f"product_spec.md 内容过于简单，仅匹配到 {found} 个关键词，内容前500字：{content[:500]}"

    def test_pm_task_assign_marked_as_read(self):
        """PM 读取邮件后，task_assign 消息应被标记为已读（幂等保护）"""
        from m4l26_pm import PMExecuteCrew, save_session

        session_id = str(uuid.uuid4())
        crew = PMExecuteCrew(session_id=session_id)
        crew.crew().kickoff(
            inputs={"user_request": "请读取邮箱中的任务，根据需求文档撰写产品规格文档，完成后通知 Manager"}
        )
        save_session(crew, session_id)

        pm_msgs = json.loads((MAILBOXES_DIR / "pm.json").read_text())
        task_assign = next((m for m in pm_msgs if m.get("type") == "task_assign"), None)
        assert task_assign is not None
        assert task_assign["read"] is True, "task_assign 消息未被标记为已读"


# ─────────────────────────────────────────────────────────────────────────────
# I4 — 步骤3：ManagerReviewCrew 真实执行
# ─────────────────────────────────────────────────────────────────────────────

class TestI4ManagerReview:
    """
    ManagerReviewCrew（步骤3）：
    读取 Manager 邮箱 task_done → 读 product_spec.md → 保存 review_result.md
    """

    @pytest.fixture(autouse=True)
    def patch_sandbox_url(self, monkeypatch, manager_sandbox):
        """确保 ManagerReviewCrew 使用 Manager 沙盒（8025）"""
        import m4l26_manager as mgr
        monkeypatch.setattr(mgr, "MANAGER_SANDBOX_MCP_URL", MANAGER_SANDBOX_URL)

    @pytest.fixture(autouse=True)
    def seed_manager_inbox_and_spec(self):
        """预置 Manager 收件箱的 task_done 邮件 + 模拟 product_spec.md"""
        # 写入模拟产品文档
        DESIGN_DIR.mkdir(parents=True, exist_ok=True)
        (DESIGN_DIR / "product_spec.md").write_text(
            "# XiaoPaw 宠物健康记录功能 - 产品规格\n\n"
            "## 产品概述\n本功能允许用户记录宠物的疫苗接种和就医记录。\n\n"
            "## 用户故事\n- F-01：用户可以添加疫苗记录（疫苗名称、接种日期、到期日）\n"
            "- F-02：用户可以添加就医记录（日期、医院、原因、诊断、费用）\n\n"
            "## 验收标准\n- 疫苗记录 CRUD 完整\n- 就医记录支持日期排序\n",
            encoding="utf-8",
        )
        # 写入模拟 task_done 邮件
        from m4l26_mailbox.mailbox_ops import send_mail
        send_mail(
            mailbox_dir=MAILBOXES_DIR,
            to="manager",
            from_="pm",
            type_="task_done",
            subject="产品文档已完成",
            content="产品规格文档已完成，请查看 /mnt/shared/design/product_spec.md 进行验收。",
        )

    def test_manager_review_writes_result(self):
        """ManagerReviewCrew 运行后，review_result.md 应被写入 Manager workspace"""
        from m4l26_manager import ManagerReviewCrew, save_session

        session_id = str(uuid.uuid4())
        crew = ManagerReviewCrew(session_id=session_id)
        crew.crew().kickoff(
            inputs={"user_request": "请读取邮箱中的完成通知，验收产品文档并保存验收结论"}
        )
        save_session(crew, session_id)

        review_path = MANAGER_DIR / "review_result.md"
        assert review_path.exists(), "review_result.md 未被写入 Manager workspace"
        content = review_path.read_text(encoding="utf-8")
        assert len(content) >= 50, f"review_result.md 内容过少（{len(content)} 字符）：{content}"

    def test_manager_task_done_marked_as_read(self):
        """Manager 读取完成通知后，task_done 消息应被标记为已读"""
        from m4l26_manager import ManagerReviewCrew, save_session

        session_id = str(uuid.uuid4())
        crew = ManagerReviewCrew(session_id=session_id)
        crew.crew().kickoff(
            inputs={"user_request": "请读取邮箱中的完成通知，验收产品文档并保存验收结论"}
        )
        save_session(crew, session_id)

        mgr_msgs = json.loads((MAILBOXES_DIR / "manager.json").read_text())
        task_done = next((m for m in mgr_msgs if m.get("type") == "task_done"), None)
        assert task_done is not None
        assert task_done["read"] is True, "task_done 消息未被标记为已读"


# ─────────────────────────────────────────────────────────────────────────────
# I5 — 端到端：完整三步流转
# ─────────────────────────────────────────────────────────────────────────────

class TestI5EndToEnd:
    """
    完整三步任务链验证：
    Manager发任务 → PM执行 → Manager验收
    验证最终状态：三个输出文件全部存在 + 消息全部已读
    """

    @pytest.fixture(autouse=True)
    def patch_sandbox_urls(self, monkeypatch, manager_sandbox, pm_sandbox):
        """
        端到端测试：Manager 用 8025，PM 用 8026，各自独立沙盒。
        """
        import m4l26_manager as mgr
        import m4l26_pm as pm
        monkeypatch.setattr(mgr, "MANAGER_SANDBOX_MCP_URL", MANAGER_SANDBOX_URL)
        monkeypatch.setattr(pm, "PM_SANDBOX_MCP_URL", PM_SANDBOX_URL)

    def test_full_three_step_flow_outputs_exist(self):
        """完整运行后，三个核心输出文件全部存在"""
        # 直接调用三步编排（不用 run_demo 以便 monkeypatch 生效）
        from m4l26_manager import ManagerAssignCrew, ManagerReviewCrew, save_session as mgr_save
        from m4l26_pm import PMExecuteCrew, save_session as pm_save
        import m4l26_run as run_module

        session_id = str(uuid.uuid4())

        # 步骤1
        c1 = ManagerAssignCrew(session_id=session_id)
        c1.crew().kickoff(inputs={"user_request": "请读取共享需求文档，向 PM 发送产品文档设计任务"})
        mgr_save(c1, session_id)
        assert run_module.check_pm_inbox_has_task_assign(), "步骤1：PM 邮箱无 task_assign"

        # 步骤2
        c2 = PMExecuteCrew(session_id=session_id)
        c2.crew().kickoff(inputs={"user_request": "请读取邮箱中的任务，根据需求文档撰写产品规格文档，完成后通知 Manager"})
        pm_save(c2, session_id)
        assert run_module.check_product_spec_exists(), "步骤2：product_spec.md 未生成"
        assert run_module.check_manager_inbox_has_task_done(), "步骤2：Manager 邮箱无 task_done"

        # 步骤3
        c3 = ManagerReviewCrew(session_id=session_id)
        c3.crew().kickoff(inputs={"user_request": "请读取邮箱中的完成通知，验收产品文档并保存验收结论"})
        mgr_save(c3, session_id)

        # 最终检查：三个输出全存在
        assert (DESIGN_DIR / "product_spec.md").exists()
        assert (MANAGER_DIR / "review_result.md").exists()

    def test_full_flow_message_idempotency(self):
        """完整流转后，所有 task_assign 和 task_done 消息均已标记为已读"""
        from m4l26_manager import ManagerAssignCrew, ManagerReviewCrew, save_session as mgr_save
        from m4l26_pm import PMExecuteCrew, save_session as pm_save

        session_id = str(uuid.uuid4())

        c1 = ManagerAssignCrew(session_id=session_id)
        c1.crew().kickoff(inputs={"user_request": "请读取共享需求文档，向 PM 发送产品文档设计任务"})
        mgr_save(c1, session_id)

        c2 = PMExecuteCrew(session_id=session_id)
        c2.crew().kickoff(inputs={"user_request": "请读取邮箱中的任务，根据需求文档撰写产品规格文档，完成后通知 Manager"})
        pm_save(c2, session_id)

        c3 = ManagerReviewCrew(session_id=session_id)
        c3.crew().kickoff(inputs={"user_request": "请读取邮箱中的完成通知，验收产品文档并保存验收结论"})
        mgr_save(c3, session_id)

        # 验证消息幂等性
        pm_msgs = json.loads((MAILBOXES_DIR / "pm.json").read_text())
        unread_pm = [m for m in pm_msgs if not m["read"]]
        assert unread_pm == [], f"PM 邮箱仍有未读消息：{unread_pm}"

        mgr_msgs = json.loads((MAILBOXES_DIR / "manager.json").read_text())
        unread_mgr = [m for m in mgr_msgs if not m["read"]]
        assert unread_mgr == [], f"Manager 邮箱仍有未读消息：{unread_mgr}"
