"""
课程：28｜数字员工的自我进化
示例文件：m4l28_pm.py

PM 两个 Crew：
  PMExecuteCrew      复用 L26/L27：读邮件 → 写产品文档 → 发 task_done 给 Manager
  PMSelfRetroCrew    第28课新增：自我复盘，调用 self-retrospective skill

单一接口约束体现：
  PM 完成设计后只向 manager.json 发 task_done，
  不接触 human.json——由 run.py 在检测到 task_done 后，
  以 manager 身份转发 checkpoint_request 给人类。

第28课新增：
  PMSelfRetroCrew 通过 self-retrospective Skill 在沙盒中执行复盘，
  不再硬编码 Python 函数直接导入（原 run.py 硬编码调用已改为 Skill 体系）。
"""

from __future__ import annotations

import sys
from pathlib import Path

from crewai import Agent, Crew, Task
from crewai.hooks import LLMCallHookContext, before_llm_call, clear_before_llm_call_hooks
from crewai.project import CrewBase, agent, crew, task

# ── 路径设置 ──────────────────────────────────────────────────────────────────
_M4L28_DIR    = Path(__file__).resolve().parent
_PROJECT_ROOT = _M4L28_DIR.parent
for _p in [str(_PROJECT_ROOT), str(_M4L28_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from llm import aliyun_llm                           # noqa: E402
from tools.skill_loader_tool import SkillLoaderTool  # noqa: E402

from m3l20.m3l20_file_memory import (                # noqa: E402
    build_bootstrap_prompt,
    prune_tool_results,
    maybe_compress,
    append_session_raw,
    save_session_ctx,
)
from m4l28_manager import _SessionMixin              # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# 路径常量
# ─────────────────────────────────────────────────────────────────────────────

WORKSPACE_DIR  = _M4L28_DIR / "workspace" / "pm"
SESSIONS_DIR   = WORKSPACE_DIR / "sessions"
SHARED_DIR     = _M4L28_DIR / "workspace" / "shared"
MAILBOXES_DIR  = SHARED_DIR / "mailboxes"

# ─────────────────────────────────────────────────────────────────────────────
# 沙盒挂载描述（PM）
# ─────────────────────────────────────────────────────────────────────────────

M4L28_PM_SANDBOX_MOUNT_DESC = (
    "1. 所有的操作必须在沙盒中执行，不得操作本地文件系统。\n"
    "   当前已挂载的目录：\n"
    "   - ./workspace/pm:/workspace:rw（PM 个人区，可读写）\n"
    "   - ./workspace/shared:/mnt/shared:rw（共享工作区，含邮箱，可读写）\n"
    "   - ../skills:/mnt/skills:ro（共享 skills 目录，只读）\n\n"
    "2. 记忆文件读写规范：\n"
    "   - 个人区读写：/workspace/<filename>\n"
    "   - 需求文档：/mnt/shared/needs/requirements.md（只读）\n"
    "   - 产品文档：/mnt/shared/design/product_spec.md（PM 负责写入）\n"
    "   - 邮箱：/mnt/shared/mailboxes/（通过 mailbox-ops skill 操作）\n"
    "   - 日志：/mnt/shared/logs/（三层日志，L1/L2/L3，通过 skill 读取）\n"
    "   - 提案：/mnt/shared/proposals/（改进提案，通过 skill 写入）\n\n"
    "3. 参考型 Skill（type: reference）：内容直接注入上下文，无需沙盒\n\n"
    "4. 如遇依赖缺失，先在沙盒中安装再继续\n\n"
    "5. 重要：PM 的邮件只能发给 manager（to: manager），"
    "   不得发给 human——所有与人类的沟通通过 Manager 统一处理。"
)


# ─────────────────────────────────────────────────────────────────────────────
# PM Crew：执行产品设计任务
# ─────────────────────────────────────────────────────────────────────────────

@CrewBase
class PMExecuteCrew(_SessionMixin):
    """
    PM 执行阶段：读取任务邮件 → 读需求 → 写产品文档 → 发完成通知给 Manager

    单一接口约束体现（对应第28课 P2 核心设计原则）：
    - PM 只向 manager.json 发 task_done，不直接写 human.json
    - 沙盒挂载描述中明确说明此约束，帮助 LLM 遵守规范
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._init_session_state(SESSIONS_DIR)

    @agent
    def pm_agent(self) -> Agent:
        backstory = build_bootstrap_prompt(WORKSPACE_DIR)
        return Agent(
            role      = "产品经理（PM）",
            goal      = "读取任务邮件，根据需求文档写出规范的产品设计文档，并通知 Manager 验收",
            backstory = backstory,
            llm       = aliyun_llm.AliyunLLM(model="qwen-max", temperature=0.3),
            tools     = [SkillLoaderTool(
                sandbox_mount_desc=M4L28_PM_SANDBOX_MOUNT_DESC,
                sandbox_mcp_url="http://localhost:8028/mcp",
            )],
            verbose   = True,
            max_iter  = 20,
        )

    @task
    def execute_task(self) -> Task:
        return Task(
            description     = "{user_request}",
            expected_output = (
                "完成以下四步：\n"
                "1. 通过 mailbox-ops skill 读取 PM 邮箱（read_inbox），获取 Manager 的任务分配邮件\n"
                "2. 读取 /mnt/shared/needs/requirements.md 理解功能需求和验收标准\n"
                "3. 根据需求撰写产品规格文档，用 memory-save 保存至 /mnt/shared/design/product_spec.md\n"
                "   文档必须包含：产品概述 + 用户故事 + 功能规格（F-01/F-02/...）+ 验收标准\n"
                "4. 通过 mailbox-ops skill 向 Manager 发送完成通知（只发给 manager，不发给其他人）：\n"
                "   - to: manager\n"
                "   - type: task_done\n"
                "   - subject: 产品文档已完成\n"
                "   - content: 完成说明（包含文档路径引用，不要复制文档全文）\n"
                "确认邮件发送成功后输出：「产品文档已完成，已通知 Manager 验收」"
            ),
            agent = self.pm_agent(),
        )

    @crew
    def crew(self) -> Crew:
        return Crew(agents=self.agents, tasks=self.tasks, verbose=True)

    @before_llm_call
    def before_llm_hook(self, context: LLMCallHookContext) -> bool | None:
        if not self._session_loaded:
            self._restore_session(context)
            self._session_loaded = True
        self._last_msgs = context.messages
        prune_tool_results(context.messages)
        maybe_compress(context.messages, context)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# PM Crew：自我复盘（第28课新增）
# ─────────────────────────────────────────────────────────────────────────────

@CrewBase
class PMSelfRetroCrew(_SessionMixin):
    """
    PM 自我复盘阶段（第28课核心教学点 P4）：

    通过 self-retrospective Skill 在沙盒中执行：
      1. 检查样本量（< min_tasks 时跳过）
      2. 读取 L2 日志，找质量最低的任务
      3. 读取对应 L3 日志，找具体失败节点
      4. 读取 L1 人类纠正记录
      5. 调用 LLM 生成结构化改进提案（json_object 模式）
      6. 写入 proposals.json + 发通知至 human.json

    设计原则：复盘是 Agent 自我学习的"元操作"，逻辑封装在 Skill 脚本内，
    通过 SkillLoaderTool 在沙盒中执行，而非硬编码 Python 函数直接导入。
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._init_session_state(SESSIONS_DIR)

    @agent
    def pm_agent(self) -> Agent:
        backstory = build_bootstrap_prompt(WORKSPACE_DIR)
        return Agent(
            role      = "产品经理（PM）",
            goal      = "执行自我复盘，分析历史任务日志，生成结构化改进提案，提交审批",
            backstory = backstory,
            llm       = aliyun_llm.AliyunLLM(model="qwen-max", temperature=0.3),
            tools     = [SkillLoaderTool(
                sandbox_mount_desc=M4L28_PM_SANDBOX_MOUNT_DESC,
                sandbox_mcp_url="http://localhost:8028/mcp",
            )],
            verbose   = True,
            max_iter  = 20,
        )

    @task
    def self_retro_task(self) -> Task:
        return Task(
            description     = "{user_request}",
            expected_output = (
                "完成以下步骤：\n"
                "1. 安装依赖：pip install openai filelock -q\n"
                "2. 执行自我复盘 skill：\n"
                "   python3 /mnt/skills/self-retrospective/scripts/self_retro.py \\\n"
                "     --logs-dir /mnt/shared/logs \\\n"
                "     --mailbox-dir /mnt/shared/mailboxes \\\n"
                "     --agent-id pm \\\n"
                "     --days 7 \\\n"
                "     --min-tasks 5\n"
                "3. 读取脚本的 JSON 输出（必须实际执行，不得伪造结果）\n"
                "4. 将脚本输出的 JSON 原文包含在回复中，并输出复盘摘要：\n"
                "   - 分析了多少条任务记录\n"
                "   - 生成了几条改进提案（或跳过原因）\n"
                "   - 提案已发送至 human.json 等待审批\n"
                "确认执行完成后输出：「自我复盘完成，结果：{脚本输出 JSON}」"
            ),
            agent = self.pm_agent(),
        )

    @crew
    def crew(self) -> Crew:
        return Crew(agents=self.agents, tasks=self.tasks, verbose=True)

    @before_llm_call
    def before_llm_hook(self, context: LLMCallHookContext) -> bool | None:
        if not self._session_loaded:
            self._restore_session(context)
            self._session_loaded = True
        self._last_msgs = context.messages
        prune_tool_results(context.messages)
        maybe_compress(context.messages, context)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 公共辅助函数
# ─────────────────────────────────────────────────────────────────────────────

def save_session(crew_instance: PMExecuteCrew | PMSelfRetroCrew, session_id: str) -> None:
    """保存 session 上下文（复用 m3l20 逻辑）"""
    if crew_instance._last_msgs:
        new_msgs = list(crew_instance._last_msgs)[crew_instance._history_len:]
        append_session_raw(session_id, new_msgs, SESSIONS_DIR)
        save_session_ctx(session_id, list(crew_instance._last_msgs), SESSIONS_DIR)
