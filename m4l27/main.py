"""
第27课：Human as 甲方 — Manager 入口（v3）

教学对比（v3 vs v2）：
  v2：run_demo.py 编排器 + wait_for_human() 阻塞等待 Human 输入
  v3：Manager 和 PM 各自独立启动，Human 在独立终端通过 human_cli.py 异步确认
      Manager 完成当前能做的事就结束，不阻塞等待 Human

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

核心教学点（v3）：
  - 无编排器：Agent 自主决策，靠 Workspace Skill 驱动
  - 异步 Human：human_cli.py 独立运行，Manager 不阻塞
  - 单一接口：只有 Manager 可以向 human.json 发消息（mailbox_cli.py 校验）
"""

from __future__ import annotations

import sys
from pathlib import Path

_M4L27_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _M4L27_DIR.parent
for _p in [str(_M4L27_DIR), str(_PROJECT_ROOT)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from shared.digital_worker import DigitalWorkerCrew  # noqa: E402

WORKSPACE_DIR = _M4L27_DIR / "workspace" / "manager"
SANDBOX_PORT  = 8027
SESSION_ID    = "demo_m4l27_manager"


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
    print("第27课：Human as 甲方 — Manager 启动（v3 异步模式）")
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
