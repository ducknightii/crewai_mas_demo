"""
课程：28｜数字员工的自我进化
示例文件：m4l28_manager.py

Manager 五个 Crew：
  RequirementsDiscoveryCrew  复用 L27：需求澄清，用 requirements-discovery skill 发问，写 requirements.md
  ManagerAssignCrew          复用 L26：读 SOP，向 PM 发送 task_assign
  ManagerReviewCrew          复用 L26：读 PM 回邮，验收产品文档
  ManagerTeamRetroCrew       第28课新增：团队复盘，调用 team-retrospective skill

与 m4l27_manager.py 的复用关系：
  - build_bootstrap_prompt / load_session_ctx / save_session_ctx / append_session_raw：完全复用
  - SkillLoaderTool / prune_tool_results / maybe_compress：完全复用
  - RequirementsDiscoveryCrew / ManagerAssignCrew / ManagerReviewCrew：结构完全复用

第28课新增：
  - ManagerTeamRetroCrew：通过 team-retrospective Skill 执行团队复盘
    （之前硬编码直接调用 Python 函数，现改为通过 Skill 体系在沙盒中执行）
  - 沙盒挂载新增 /mnt/shared/logs（三层日志）、/mnt/shared/proposals（改进提案）
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
    load_session_ctx,
    save_session_ctx,
    append_session_raw,
    prune_tool_results,
    maybe_compress,
)

# ─────────────────────────────────────────────────────────────────────────────
# 路径常量
# ─────────────────────────────────────────────────────────────────────────────

WORKSPACE_DIR  = _M4L28_DIR / "workspace" / "manager"
SESSIONS_DIR   = WORKSPACE_DIR / "sessions"
SHARED_DIR     = _M4L28_DIR / "workspace" / "shared"
MAILBOXES_DIR  = SHARED_DIR / "mailboxes"

# ─────────────────────────────────────────────────────────────────────────────
# 沙盒挂载描述（Manager）
# ─────────────────────────────────────────────────────────────────────────────

M4L28_MANAGER_SANDBOX_MOUNT_DESC = (
    "1. 所有的操作必须在沙盒中执行，不得操作本地文件系统。\n"
    "   当前已挂载的目录：\n"
    "   - ./workspace/manager:/workspace:rw（Manager 个人区，可读写）\n"
    "   - ./workspace/shared:/mnt/shared:rw（共享工作区，含邮箱，可读写）\n"
    "   - ../skills:/mnt/skills:ro（共享 skills 目录，只读）\n\n"
    "2. 记忆文件读写规范：\n"
    "   - 个人区读写：/workspace/<filename>\n"
    "   - 需求文档：/mnt/shared/needs/requirements.md（Manager 可写）\n"
    "   - 产品文档：/mnt/shared/design/product_spec.md（只读，PM 负责写入）\n"
    "   - SOP 目录：/mnt/shared/sop/（只读，Manager 读取 SOP 文件）\n"
    "   - 邮箱：/mnt/shared/mailboxes/（通过 mailbox-ops skill 操作）\n"
    "   - 日志：/mnt/shared/logs/（三层日志，L1/L2/L3，通过 skill 读取）\n"
    "   - 提案：/mnt/shared/proposals/（改进提案，通过 skill 写入）\n\n"
    "3. 参考型 Skill（type: reference）：内容直接注入上下文，无需沙盒\n\n"
    "4. 如遇依赖缺失，先在沙盒中安装再继续"
)


# ─────────────────────────────────────────────────────────────────────────────
# 公共 Mixin：session 管理（所有 Manager Crew 共用）
# ─────────────────────────────────────────────────────────────────────────────

class _SessionMixin:
    """为各 Crew 提供统一的 session 保存/恢复逻辑（Manager 和 PM 均可复用）。"""

    session_id: str
    _sessions_dir: Path
    _session_loaded: bool
    _last_msgs: list[dict]
    _history_len: int

    def _init_session_state(self, sessions_dir: Path) -> None:
        self._sessions_dir   = sessions_dir
        self._session_loaded = False
        self._last_msgs      = []
        self._history_len    = 0

    def _restore_session(self, context: LLMCallHookContext) -> None:
        history = load_session_ctx(self.session_id, self._sessions_dir)
        self._history_len = len(history)
        if not history:
            return
        current_user_msg = next(
            (m for m in reversed(context.messages) if m.get("role") == "user"), None
        )
        context.messages.clear()
        context.messages.extend(history)
        if current_user_msg is not None:
            context.messages.append(current_user_msg)

    def _build_agent(self, role: str, goal: str) -> Agent:
        return Agent(
            role      = role,
            goal      = goal,
            backstory = build_bootstrap_prompt(WORKSPACE_DIR),
            llm       = aliyun_llm.AliyunLLM(model="qwen-max", temperature=0.3),
            tools     = [SkillLoaderTool(
                sandbox_mount_desc=M4L28_MANAGER_SANDBOX_MOUNT_DESC,
                sandbox_mcp_url="http://localhost:8027/mcp",
            )],
            verbose   = True,
            max_iter  = 20,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Crew 1：需求澄清（新增）
# ─────────────────────────────────────────────────────────────────────────────

@CrewBase
class RequirementsDiscoveryCrew(_SessionMixin):
    """
    Manager 需求澄清阶段：
    - 使用 requirements-discovery skill 按四维发问框架（目标/边界/约束/风险）澄清需求
    - 将澄清结果整理成结构化需求文档，写入 /mnt/shared/needs/requirements.md
    - 完成后由 run.py 以 manager 身份写 human.json:needs_confirm（单一接口原则）

    核心教学点（对应第28课 P3）：
    - Manager 是唯一的需求接口，主动发问，而不是被动接收
    - 需求落文档才算数：未写入 requirements.md 的澄清无效
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._init_session_state(SESSIONS_DIR)

    @agent
    def manager_agent(self) -> Agent:
        return self._build_agent(
            role = "项目经理（Manager）",
            goal = "理解用户初步需求，用结构化方式澄清关键问题，整理成完整需求文档",
        )

    @task
    def discover_requirements_task(self) -> Task:
        return Task(
            description     = "{user_request}",
            expected_output = (
                "完成以下三步：\n"
                "1. 阅读用户的初步需求\n"
                "2. 使用 requirements-discovery skill，按四维框架（目标/边界/约束/风险）"
                "   梳理出完整需求要点（如需澄清但无法实时追问，记录为「待确认」）\n"
                "3. 用 memory-save 将整理后的需求文档写入 /mnt/shared/needs/requirements.md\n"
                "   文档格式：\n"
                "   ## 目标\n"
                "   ## 边界（本次做什么 / 不做什么）\n"
                "   ## 约束\n"
                "   ## 风险与待确认项\n"
                "   ## 验收标准\n"
                "确认文档写入后输出：「需求文档已完成，路径：/mnt/shared/needs/requirements.md」"
            ),
            agent = self.manager_agent(),
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
# Crew 2：任务分配（复用 L26 结构）
# ─────────────────────────────────────────────────────────────────────────────

@CrewBase
class ManagerAssignCrew(_SessionMixin):
    """
    Manager 任务分配阶段：读 SOP → 分配任务给 PM

    核心教学点（对应第28课 P4）：
    - 任务执行必须遵循 SOP，Manager 先读 SOP 再分配
    - 邮件只传路径引用，不传文档全文（Design with File 范式）
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._init_session_state(SESSIONS_DIR)

    @agent
    def manager_agent(self) -> Agent:
        return self._build_agent(
            role = "项目经理（Manager）",
            goal = "读取 SOP 和需求文档，通过邮箱向 PM 分配产品文档设计任务",
        )

    @task
    def assign_task(self) -> Task:
        return Task(
            description     = "{user_request}",
            expected_output = (
                "完成以下三步：\n"
                "1. 读取 /mnt/shared/sop/product_design_sop.md，了解产品设计 SOP\n"
                "2. 读取 /mnt/shared/needs/requirements.md，确认需求已完整\n"
                "3. 通过 mailbox-ops skill 向 PM 发送任务分配邮件：\n"
                "   - type: task_assign\n"
                "   - subject: 产品文档设计任务\n"
                "   - content: 任务说明（含需求文档路径、SOP checkpoint 说明、验收标准）\n"
                "   skill 必须实际执行脚本并返回 JSON（含 msg_id）。\n"
                "   将脚本返回的 msg_id 原文包含在最终回复中，证明邮件已成功写入邮箱。\n"
                "确认邮件发送成功后输出：「任务已分配给 PM，msg_id: {从脚本输出中读取}」"
            ),
            agent = self.manager_agent(),
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
# Crew 3：验收（复用 L26 结构）
# ─────────────────────────────────────────────────────────────────────────────

@CrewBase
class ManagerReviewCrew(_SessionMixin):
    """
    Manager 验收阶段：读 PM 回邮 → 验收产品文档 → 保存验收结论

    核心教学点（对应第28课 P4/P5）：
    - Checkpoint 之外的验收由 Manager 自主完成，不再打扰人
    - 验收结论写入个人区（/workspace/review_result.md），形成审计记录
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._init_session_state(SESSIONS_DIR)

    @agent
    def manager_agent(self) -> Agent:
        return self._build_agent(
            role = "项目经理（Manager）",
            goal = "读取 PM 完成通知，验收产品文档，保存验收结论",
        )

    @task
    def review_task(self) -> Task:
        return Task(
            description     = "{user_request}",
            expected_output = (
                "完成以下三步：\n"
                "1. 通过 mailbox-ops skill 读取 Manager 邮箱（read_inbox），获取 PM 的完成通知\n"
                "2. 读取邮件中引用的产品文档路径（/mnt/shared/design/product_spec.md）\n"
                "3. 根据需求的验收标准逐项检查，用 memory-save 保存验收结论至 /workspace/review_result.md\n"
                "   格式：验收结论（通过/需返工）+ 检查项逐条说明\n"
                "输出验收结论摘要"
            ),
            agent = self.manager_agent(),
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
# Crew 4：团队复盘（第28课新增）
# ─────────────────────────────────────────────────────────────────────────────

@CrewBase
class ManagerTeamRetroCrew(_SessionMixin):
    """
    Manager 团队复盘阶段（第28课核心教学点 P5）：

    通过 team-retrospective Skill 在沙盒中执行：
      1. 聚合所有 Agent 的 L2 质量指标
      2. 统计 L1 人类纠正事件
      3. 识别瓶颈 Agent，发邮件触发其自我复盘
      4. 调用 LLM 生成团队级改进提案
      5. 向 human.json 发送周报

    设计原则：复盘逻辑封装在 Skill 脚本内，Manager Agent 只负责调用，
    不再硬编码 Python 函数直接导入。
    """

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self._init_session_state(SESSIONS_DIR)

    @agent
    def manager_agent(self) -> Agent:
        return self._build_agent(
            role = "项目经理（Manager）",
            goal = "执行团队复盘，分析 Agent 质量指标，识别瓶颈，生成团队改进提案，发送周报",
        )

    @task
    def team_retro_task(self) -> Task:
        return Task(
            description     = "{user_request}",
            expected_output = (
                "完成以下步骤：\n"
                "1. 安装依赖：pip install openai filelock -q\n"
                "2. 执行团队复盘 skill：\n"
                "   python3 /mnt/skills/team-retrospective/scripts/team_retro.py \\\n"
                "     --logs-dir /mnt/shared/logs \\\n"
                "     --mailbox-dir /mnt/shared/mailboxes \\\n"
                "     --manager-id manager \\\n"
                "     --agent-ids pm,manager \\\n"
                "     --days 7\n"
                "3. 读取脚本的 JSON 输出（必须实际执行，不得伪造结果）\n"
                "4. 将脚本输出的 JSON 原文包含在回复中，并输出复盘摘要：\n"
                "   - 各 Agent 质量指标\n"
                "   - 识别到的瓶颈 Agent\n"
                "   - 生成的团队提案数量\n"
                "确认执行完成后输出：「团队复盘完成，结果：{脚本输出 JSON}」"
            ),
            agent = self.manager_agent(),
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

def save_session(
    crew_instance: RequirementsDiscoveryCrew | ManagerAssignCrew | ManagerReviewCrew | ManagerTeamRetroCrew,
    session_id: str,
) -> None:
    """保存 session 上下文（复用 m3l20 逻辑）"""
    if crew_instance._last_msgs:
        new_msgs = list(crew_instance._last_msgs)[crew_instance._history_len:]
        append_session_raw(session_id, new_msgs, SESSIONS_DIR)
        save_session_ctx(session_id, list(crew_instance._last_msgs), SESSIONS_DIR)
