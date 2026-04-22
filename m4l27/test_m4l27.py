"""
第27课：Human as 甲方 — 测试套件（v3）

测试分层：
  - 单元测试（无需 LLM，每次 CI 必跑）：
    * TestSingleInterfaceConstraint：单一接口约束
    * TestHumanJsonSchema：human.json 使用二态 read 字段
    * TestHumanCli：human_cli.py 核心功能
    * TestCheckHumanCommand：check-human 子命令
    * TestInitWorkspace：第27课新增 sop/ 和 human.json

运行单元测试：
  pytest test_m4l27.py -v -m "not e2e"
"""

from __future__ import annotations

import json
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

# ── 路径设置
_M4L27_DIR    = Path(__file__).resolve().parent
_PROJECT_ROOT = _M4L27_DIR.parent
for _p in [str(_M4L27_DIR), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

MAILBOX_CLI = (
    _M4L27_DIR
    / "workspace" / "manager" / "skills" / "mailbox" / "scripts" / "mailbox_cli.py"
)
INIT_SCRIPT = (
    _M4L27_DIR
    / "workspace" / "manager" / "skills" / "init_project" / "scripts"
    / "init_workspace.py"
)

e2e = pytest.mark.e2e


# ── Fixtures

@pytest.fixture()
def tmp_mailboxes(tmp_path: Path) -> Path:
    mb = tmp_path / "mailboxes"
    mb.mkdir()
    for role in ["manager", "pm", "human"]:
        (mb / f"{role}.json").write_text("[]", encoding="utf-8")
    return mb


@pytest.fixture()
def human_inbox(tmp_mailboxes: Path) -> Path:
    return tmp_mailboxes / "human.json"


# ── Helpers

def _run_mailbox_cli(*args: str) -> dict:
    result = subprocess.run(
        [sys.executable, str(MAILBOX_CLI), *args],
        capture_output=True, text=True,
    )
    try:
        return json.loads(result.stdout.strip())
    except json.JSONDecodeError:
        return {"errcode": -1, "raw": result.stdout + result.stderr}


def _send_to_human(mailboxes_dir: Path, type_: str, from_: str = "manager") -> dict:
    return _run_mailbox_cli(
        "send",
        "--mailboxes-dir", str(mailboxes_dir),
        "--from", from_,
        "--to", "human",
        "--type", type_,
        "--subject", f"测试（{type_}）",
        "--content", "测试内容",
    )


def _inject_human_msg(
    human_inbox: Path,
    type_: str = "needs_confirm",
    read: bool = False,
    rejected: bool = False,
    feedback: str | None = None,
) -> dict:
    msg: dict = {
        "id": f"msg-{uuid.uuid4().hex[:8]}",
        "from": "manager",
        "to": "human",
        "type": type_,
        "subject": f"测试（{type_}）",
        "content": "测试内容",
        "timestamp": "2026-04-15T00:00:00+00:00",
        "read": read,
    }
    if rejected:
        msg["rejected"] = True
    if feedback:
        msg["human_feedback"] = feedback
    inbox = json.loads(human_inbox.read_text(encoding="utf-8"))
    inbox.append(msg)
    human_inbox.write_text(json.dumps(inbox, ensure_ascii=False), encoding="utf-8")
    return msg


# ── 单一接口约束

class TestSingleInterfaceConstraint:
    def test_pm_cannot_write_human(self, tmp_mailboxes: Path) -> None:
        result = _send_to_human(tmp_mailboxes, type_="checkpoint_request", from_="pm")
        assert result.get("errcode") == 1
        assert "单一接口约束" in result.get("errmsg", "")

    def test_other_role_cannot_write_human(self, tmp_mailboxes: Path) -> None:
        result = _send_to_human(tmp_mailboxes, type_="task_done", from_="dev")
        assert result.get("errcode") == 1

    def test_manager_can_write_human(self, tmp_mailboxes: Path, human_inbox: Path) -> None:
        result = _send_to_human(tmp_mailboxes, type_="needs_confirm", from_="manager")
        assert result.get("errcode") == 0
        inbox = json.loads(human_inbox.read_text(encoding="utf-8"))
        assert len(inbox) == 1
        assert inbox[0]["type"] == "needs_confirm"


# ── human.json Schema

class TestHumanJsonSchema:
    def test_human_message_has_read_field_not_status(
        self, tmp_mailboxes: Path, human_inbox: Path
    ) -> None:
        _send_to_human(tmp_mailboxes, type_="needs_confirm")
        inbox = json.loads(human_inbox.read_text(encoding="utf-8"))
        msg = inbox[0]
        assert "read" in msg
        assert msg["read"] is False
        assert "status" not in msg

    def test_agent_message_has_status_not_read(self, tmp_mailboxes: Path) -> None:
        result = _run_mailbox_cli(
            "send",
            "--mailboxes-dir", str(tmp_mailboxes),
            "--from", "manager",
            "--to", "pm",
            "--type", "task_assign",
            "--subject", "测试",
            "--content", "内容",
        )
        assert result.get("errcode") == 0
        pm_inbox = json.loads((tmp_mailboxes / "pm.json").read_text(encoding="utf-8"))
        msg = pm_inbox[0]
        assert "status" in msg
        assert msg["status"] == "unread"
        assert "read" not in msg


# ── human_cli.py 功能（通过 importlib 动态加载，覆盖路径常量）

def _load_human_cli(human_inbox: Path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("human_cli", _M4L27_DIR / "human_cli.py")
    assert spec is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    # 覆盖模块级常量（必须在 exec_module 之后设置，否则会被模块代码覆盖）
    mod.HUMAN_INBOX = human_inbox  # type: ignore[attr-defined]
    mod.LOCK_PATH = human_inbox.with_suffix(".json.lock")  # type: ignore[attr-defined]
    # 校验覆盖是否生效（防止 human_cli.py 内部将常量作为默认参数捕获）
    assert mod.HUMAN_INBOX == human_inbox, f"常量覆盖失败：{mod.HUMAN_INBOX!r} != {human_inbox!r}"
    return mod


class TestHumanCli:
    def test_no_unread_returns_empty(self, human_inbox: Path) -> None:
        mod = _load_human_cli(human_inbox)
        assert mod.check_messages() == []

    def test_check_returns_unread(self, human_inbox: Path) -> None:
        msg = _inject_human_msg(human_inbox, read=False)
        mod = _load_human_cli(human_inbox)
        unread = mod.check_messages()
        assert len(unread) == 1
        assert unread[0]["id"] == msg["id"]

    def test_read_message_not_in_unread(self, human_inbox: Path) -> None:
        _inject_human_msg(human_inbox, read=True)
        mod = _load_human_cli(human_inbox)
        assert mod.check_messages() == []

    def test_respond_confirm_marks_read_true(self, human_inbox: Path) -> None:
        msg = _inject_human_msg(human_inbox, read=False)
        mod = _load_human_cli(human_inbox)
        ok = mod.respond(msg["id"], confirmed=True)
        assert ok is True
        updated = json.loads(human_inbox.read_text(encoding="utf-8"))
        target = next(m for m in updated if m["id"] == msg["id"])
        assert target["read"] is True
        assert "rejected" not in target

    def test_respond_reject_sets_rejected_and_feedback(self, human_inbox: Path) -> None:
        msg = _inject_human_msg(human_inbox, read=False)
        mod = _load_human_cli(human_inbox)
        ok = mod.respond(msg["id"], confirmed=False, feedback="需要增加多语言支持")
        assert ok is True
        updated = json.loads(human_inbox.read_text(encoding="utf-8"))
        target = next(m for m in updated if m["id"] == msg["id"])
        assert target["read"] is True
        assert target.get("rejected") is True
        assert target.get("human_feedback") == "需要增加多语言支持"

    def test_respond_nonexistent_msg_returns_false(self, human_inbox: Path) -> None:
        mod = _load_human_cli(human_inbox)
        ok = mod.respond("msg-nonexistent", confirmed=True)
        assert ok is False


# ── check-human 子命令

class TestCheckHumanCommand:
    def test_empty_inbox_returns_not_confirmed(self, tmp_mailboxes: Path) -> None:
        result = _run_mailbox_cli(
            "check-human", "--mailboxes-dir", str(tmp_mailboxes), "--type", "needs_confirm",
        )
        assert result["errcode"] == 0
        assert result["data"]["confirmed"] is False

    def test_unread_returns_not_confirmed(self, tmp_mailboxes: Path, human_inbox: Path) -> None:
        _inject_human_msg(human_inbox, type_="needs_confirm", read=False)
        result = _run_mailbox_cli(
            "check-human", "--mailboxes-dir", str(tmp_mailboxes), "--type", "needs_confirm",
        )
        assert result["errcode"] == 0
        assert result["data"]["confirmed"] is False

    def test_read_true_returns_confirmed(self, tmp_mailboxes: Path, human_inbox: Path) -> None:
        _inject_human_msg(human_inbox, type_="needs_confirm", read=True)
        result = _run_mailbox_cli(
            "check-human", "--mailboxes-dir", str(tmp_mailboxes), "--type", "needs_confirm",
        )
        assert result["errcode"] == 0
        assert result["data"]["confirmed"] is True

    def test_rejected_returns_not_confirmed(self, tmp_mailboxes: Path, human_inbox: Path) -> None:
        _inject_human_msg(
            human_inbox, type_="needs_confirm",
            read=True, rejected=True, feedback="需要多语言",
        )
        result = _run_mailbox_cli(
            "check-human", "--mailboxes-dir", str(tmp_mailboxes), "--type", "needs_confirm",
        )
        assert result["errcode"] == 0
        assert result["data"]["confirmed"] is False
        assert result["data"].get("rejected") is True
        assert result["data"].get("human_feedback") == "需要多语言"


# ── init_workspace.py

class TestInitWorkspace:
    def test_creates_sop_directory(self, tmp_path: Path) -> None:
        shared = tmp_path / "shared"
        result = subprocess.run(
            [sys.executable, str(INIT_SCRIPT),
             "--shared-dir", str(shared),
             "--roles", "manager", "pm", "human"],
            capture_output=True, text=True,
        )
        out = json.loads(result.stdout.strip())
        assert out["errcode"] == 0
        assert (shared / "sop").is_dir()

    def test_creates_human_json(self, tmp_path: Path) -> None:
        shared = tmp_path / "shared"
        subprocess.run(
            [sys.executable, str(INIT_SCRIPT),
             "--shared-dir", str(shared),
             "--roles", "manager", "pm", "human"],
            capture_output=True, text=True,
        )
        human_json = shared / "mailboxes" / "human.json"
        assert human_json.exists()
        assert json.loads(human_json.read_text()) == []

    def test_idempotent_preserves_messages(self, tmp_path: Path) -> None:
        shared = tmp_path / "shared"
        cmd = [sys.executable, str(INIT_SCRIPT),
               "--shared-dir", str(shared),
               "--roles", "manager", "pm", "human"]
        subprocess.run(cmd, capture_output=True, text=True)
        human_json = shared / "mailboxes" / "human.json"
        human_json.write_text(
            json.dumps([{"id": "existing", "read": False}]), encoding="utf-8"
        )
        subprocess.run(cmd, capture_output=True, text=True)
        content = json.loads(human_json.read_text())
        assert len(content) == 1, "已有消息不应被清空"
