"""
第28课：数字员工的自我进化 — Manager 入口（v3）

复用 DigitalWorkerCrew（与 m4l26/m4l27 完全一致），
角色身份由 workspace/ 文件决定，代码层面零角色特异性。

运行方式（三终端协作）：
  # Terminal 1 — Manager 发起项目
  python main.py "帮我把用户注册流程的产品设计做出来"

  # Terminal 2 — Human 查看并确认消息
  python human_cli.py

  # Terminal 1 — Manager 继续（Human 已确认需求后）
  python main.py "需求已确认，请选择 SOP 并分配任务"

  # Terminal 3 — PM 独立工作
  python start_pm.py

  # Terminal 1 — Manager 验收
  python main.py "设计已确认，请审核产品文档"

端到端演示（单终端顺序跑完全流程）：
  python run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_M4L28_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _M4L28_DIR.parent
for _p in [str(_M4L28_DIR), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared.digital_worker import DigitalWorkerCrew  # noqa: E402

WORKSPACE_DIR = _M4L28_DIR / "workspace" / "manager"
SANDBOX_PORT  = 8027
SESSION_ID    = "demo_m4l28_manager"


def main() -> None:
    user_request = " ".join(sys.argv[1:]).strip()
    if not user_request:
        user_request = (
            "你是团队的 Manager，同时也是 Human 甲方的唯一对接窗口。\n"
            "请根据你的工作规范（agent.md）开始新项目：\n"
            "宠物健康记录App 产品设计，支持多宠物管理和疫苗提醒。\n\n"
            "按照以下顺序推进：\n"
            "1. 初始化共享工作区（init_project Skill，包含 human 角色）\n"
            "2. 使用 requirements_discovery Skill 进行需求澄清，将结果写入 needs/requirements.md\n"
            "3. 使用 notify_human Skill 通知 Human 确认需求文档（type: needs_confirm）\n"
            "4. 完成本轮，等待 Human 通过 human_cli.py 确认"
        )

    worker = DigitalWorkerCrew(
        workspace_dir=WORKSPACE_DIR,
        sandbox_port=SANDBOX_PORT,
        session_id=SESSION_ID,
        model="glm-5.1",
        has_shared=True,
    )

    print(f"\n{'='*60}")
    print("第28课：数字员工的自我进化 — Manager 启动（v3 异步模式）")
    print(f"{'='*60}")
    print(f"Session ID : {SESSION_ID}")
    print(f"Workspace  : {WORKSPACE_DIR}")
    print(f"沙盒端口   : {SANDBOX_PORT}")
    print(f"{'─'*60}")
    print("Human 端请在另一个终端运行：python human_cli.py")
    print(f"{'─'*60}\n")

    result = worker.kickoff(user_request)

    print(f"\n{'─'*60}")
    print("Manager 输出：")
    print(result)
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
