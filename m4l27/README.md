# 第27课示例代码：Human as 甲方——人工介入的三个工程节点

本课在第26课四步任务链基础上增加 **3个人工确认节点**，新增 **SOP 制定与选择流程**，并实现了三态状态机的完整运用。

**使用通用 DigitalWorkerCrew 框架**：所有角色（Manager、PM）共用同一个类，角色身份由 workspace 文件决定。

---

## 核心教学点

| 概念 | 说明 |
|------|------|
| **单一接口原则** | `human.json` 只由 Manager 写入，PM 不可直接联系 Human |
| **编排器控制时机** | 何时打扰人由脚本决定，不由 LLM 自行判断 |
| **通用框架** | `DigitalWorkerCrew` × N 实例（同一个类），角色身份来自 workspace |
| **异步 Human** | `human_cli.py` 独立运行，Manager 不阻塞等待 Human |
| **多轮迭代** | 需求澄清和 SOP 制定均支持人工反馈驱动的多轮修订 |
| **两个时点解耦** | 时点A（SOP制定）与时点B（任务执行）完全独立 |
| **三态状态机** | agent 邮箱：`unread → in_progress → done` |

---

## 目录结构

```
m4l27/
├── main.py                   # Manager 入口（v3 异步模式）
├── human_cli.py              # Human 端命令行工具（v3 核心新增）
├── start_pm.py               # PM 入口（独立运行）
├── sop_setup.py              # 时点A：SOP 共创入口
├── test_m4l27.py             # 单元测试（24个）+ 集成测试（7个，需 LLM）
├── conftest.py               # pytest fixtures
├── pytest.ini                # pytest 配置
├── sandbox-docker-compose.yaml
└── workspace/
    ├── manager/              # Manager workspace
    │   ├── soul.md           #   身份与决策偏好
    │   ├── agent.md          #   工作规范（4个场景）
    │   ├── user.md           #   服务对象画像
    │   ├── memory.md         #   跨 session 记忆索引
    │   └── skills/           #   Manager 专属技能
    │       ├── init_project/ #     初始化共享工作区
    │       ├── requirements_discovery/  # 需求澄清框架
    │       ├── sop_creator/  #     SOP 模板创建
    │       ├── sop_selector/ #     SOP 选择
    │       ├── notify_human/ #     通知 Human
    │       └── mailbox/      #     邮箱操作（含 mailbox_cli.py）
    ├── pm/                   # PM workspace
    │   ├── soul.md / agent.md / user.md / memory.md
    │   └── skills/
    │       ├── product_design/  # 产品设计技能
    │       └── mailbox/         # 邮箱操作
    └── shared/               # 共享工作区
        ├── mailboxes/        # manager.json / pm.json / human.json
        ├── needs/            # requirements.md（需求文档产出）
        ├── design/           # product_spec.md（产品规格产出）
        └── sop/              # SOP 模板库
```

---

## 运行前准备

### 环境要求

```bash
# Python 依赖（从项目根目录）
cd /root/course/code/crewai_mas_demo
pip install crewai filelock

# 阿里云 API Key（用于 LLM 调用）
export DASHSCOPE_API_KEY="your-api-key"
```

### 启动沙盒

```bash
cd /root/course/code/crewai_mas_demo/m4l27
docker compose -f sandbox-docker-compose.yaml up -d

# 验证沙盒可用
curl -s http://localhost:8027/mcp | head -1   # Manager 沙盒
curl -s http://localhost:8028/mcp | head -1   # PM 沙盒
```

| 角色 | 沙盒端口 | 个人区挂载 | 共享区挂载 |
|------|---------|-----------|-----------|
| Manager | 8027 | `workspace/manager` → `/workspace` | `workspace/shared` → `/mnt/shared` |
| PM | 8028 | `workspace/pm` → `/workspace` | `workspace/shared` → `/mnt/shared` |

---

## 课程演示流程（三终端协作）

### 时点A（可选）：SOP 共创

```bash
# Terminal 1 — Manager 发起 SOP 共创
cd /root/course/code/crewai_mas_demo
python3 m4l27/sop_setup.py

# Terminal 2 — Human 确认 SOP 草稿
python3 m4l27/human_cli.py
```

> 课程自带示例 SOP（`workspace/shared/sop/product_design_sop.md`），可跳过此步。

### 时点B：任务执行（5步 + 3个确认节点）

**Step 1 — Manager 发起项目、澄清需求**

```bash
# Terminal 1
cd /root/course/code/crewai_mas_demo
python3 m4l27/main.py
```

不带参数时使用默认需求（宠物健康记录App）。Manager 会：
1. 初始化共享工作区
2. 需求澄清 → 写入 `workspace/shared/needs/requirements.md`
3. 通知 Human 确认需求 → 写入 `human.json`

**确认节点1 — Human 确认需求**

```bash
# Terminal 2
python3 m4l27/human_cli.py          # 交互式，输入 y 确认
# 或非交互式：
python3 m4l27/human_cli.py check    # 查看未读消息
python3 m4l27/human_cli.py respond <msg_id> y   # 确认
```

**Step 2 — Manager 选择 SOP**

```bash
# Terminal 1
python3 m4l27/main.py "需求已确认，请选择 SOP 并通知 Human 确认"
```

**确认节点2 — Human 确认 SOP 选择**

```bash
# Terminal 2
python3 m4l27/human_cli.py          # 输入 y 确认 SOP
```

**Step 3 — Manager 分配任务给 PM**

```bash
# Terminal 1
python3 m4l27/main.py "SOP 已确认，请向 PM 分配产品设计任务"
```

**Step 4 — PM 执行任务**

```bash
# Terminal 3
python3 m4l27/start_pm.py
```

PM 会：读取邮箱 → 按 SOP 撰写产品文档 → 写入 `product_spec.md` → 通知 Manager

**确认节点3 — Human 确认交付物**

```bash
# Terminal 2
python3 m4l27/human_cli.py          # 确认产品文档
```

**Step 5 — Manager 验收**

```bash
# Terminal 1
python3 m4l27/main.py "设计已确认，请审核产品文档并出具验收报告"
```

Manager 读取产品文档 → 验收 → 写入 `workspace/manager/review_result.md`

---

## 预期产出

| 产出 | 路径 | 写入者 |
|------|------|--------|
| 需求文档 | `workspace/shared/needs/requirements.md` | Manager |
| 活跃 SOP | `workspace/shared/sop/active_sop.md` | Manager |
| 产品规格 | `workspace/shared/design/product_spec.md` | PM |
| 验收报告 | `workspace/manager/review_result.md` | Manager |

---

## 清理环境（支持重跑）

```bash
cd /root/course/code/crewai_mas_demo/m4l27

# 清空邮箱
echo "[]" > workspace/shared/mailboxes/manager.json
echo "[]" > workspace/shared/mailboxes/pm.json
echo "[]" > workspace/shared/mailboxes/human.json

# 清理产出
rm -f workspace/shared/needs/requirements.md
rm -f workspace/shared/design/product_spec.md
rm -f workspace/shared/sop/active_sop.md
rm -f workspace/manager/review_result.md

# 清理 session 历史
rm -rf workspace/manager/sessions workspace/pm/sessions
```

---

## 停止沙盒

```bash
cd /root/course/code/crewai_mas_demo/m4l27
docker compose -f sandbox-docker-compose.yaml down
```

---

## 运行测试

```bash
cd /root/course/code/crewai_mas_demo

# 单元测试（18个，不需要沙盒/API）
python3 -m pytest m4l27/test_m4l27.py -v -k "not needs_llm"

# 集成测试（需要沙盒 + LLM API）
python3 -m pytest m4l27/test_m4l27.py -v -k "needs_llm" -s
```

### 测试用例一览

| 类名 | 说明 | 需要LLM |
|------|------|---------|
| `TestHumanInboxEmpty` | human.json 为空/类型不匹配 | ✗ |
| `TestSinglePointOfContact` | PM 不可写 human.json / Manager 可以 | ✗ |
| `TestWaitForHuman` | 确认/拒绝/反馈收集 | ✗ |
| `TestBuildClarificationInputs` | 多轮澄清输入构造 + 花括号转义 | ✗ |
| `TestThreeStateMachine` | 三态状态机：send/read/mark_done/reset_stale | ✗ |
| `TestCheckSopExists` | active_sop.md 存在检查 | ✗ |
| `TestGenericFramework` | DigitalWorkerCrew 导入/常量/backstory/端口/模板/兼容性 | ✗ |

---

## 三态状态机

```
发送  → status: unread
取走  → status: in_progress + processing_since 时间戳
完成  → status: done（编排器调用 mark_done）
崩溃  → reset_stale() 恢复为 unread
```

human 邮箱使用简化的 `read` 字段（同步确认，不需要三态）。

---

## 常见问题

**Q：运行到确认节点卡住不动？**
正常行为——需要在 Terminal 2 运行 `human_cli.py` 确认。

**Q：没有 SOP 库怎么办？**
课程目录自带示例 SOP，可直接运行。

**Q：报 `ModuleNotFoundError`？**
确认从 `crewai_mas_demo/` 目录运行：
```bash
cd /root/course/code/crewai_mas_demo
python3 m4l27/main.py
```

**Q：自定义项目需求？**
```bash
python3 m4l27/main.py "你的自定义需求描述"
```

---

## 课堂代码演示学习指南

本节帮你按课程教学顺序阅读代码，建立完整的理解链路。

### 整体架构一览

```
┌────────────────────────────────────────────────────────────────────┐
│                        三终端协作                                   │
│                                                                    │
│  Terminal 1          Terminal 2           Terminal 3                │
│  main.py             human_cli.py         start_pm.py              │
│  (Manager)           (Human)              (PM)                     │
│                                                                    │
│  Step1: 需求澄清 ──→ 确认节点1 ──→                                 │
│  Step2: SOP 选择 ──→ 确认节点2 ──→                                 │
│  Step3: 任务分配 ─────────────────→ Step4: PM 执行                  │
│                   ←── 确认节点3 ←── PM 产出通知                     │
│  Step5: 验收     ←──────────────── task_done                       │
└────────────────────────────────────────────────────────────────────┘
                            │
                      ┌─────▼──────┐
                      │ human.json │ ← 只有 Manager 可写
                      │ pm.json    │ ← Manager 可写
                      │ manager.json│ ← PM 可写
                      └────────────┘
```

### 学习路线（建议按顺序阅读）

---

#### 第一步：理解单一接口原则——代码级强制

**对应课文**：第二节"单一接口原则"

**阅读文件**：`workspace/manager/skills/mailbox/scripts/mailbox_cli.py`

| 重点区域 | 看什么 |
|---------|--------|
| `_is_human_inbox()` 判断 | 检测目标邮箱是否是 `human.json` |
| 权限校验（约第77行） | `if _is_human_inbox(args.to) and args.from_ != "manager"` → 返回 errcode=1 |
| `check-human` 子命令 | Manager 轮询 Human 确认状态的接口 |

**理解要点**：PM 不能给 Human 发邮件，这不是 prompt 级约束（"请不要直接联系甲方"），而是代码级强制（CLI 直接拒绝）。这就是课文说的"把决策权交给代码，不交给 LLM"。

**验证**：`python3 -m pytest test_m4l27.py -v -k "TestSingleInterface"` 看权限校验测试。

---

#### 第二步：理解 Human 邮箱的简化设计

**对应课文**：第三节"两态 vs 三态"

**对比阅读**：

| 邮箱 | Schema | 为什么 |
|------|--------|--------|
| `manager.json` / `pm.json` | 三态：`unread → in_progress → done` | Agent 可能崩溃，需要 `reset_stale` 恢复 |
| `human.json` | 二态：`read: false/true` | Human 不会崩溃，不需要 `in_progress` 中间态 |

**阅读文件**：看 `mailbox_cli.py` 中 `human.json` 的读写逻辑，对比 Agent 邮箱的三态机制。

---

#### 第三步：看 Human CLI——异步交互的实现

**对应课文**：第四节"异步 Human"

**阅读文件**：`human_cli.py`

| 重点区域 | 看什么 |
|---------|--------|
| 交互模式 | 轮询 `human.json` → 显示未读消息 → Human 输入 y/n |
| `check` 子命令 | 非交互式查看未读消息（脚本化测试用） |
| `respond` 子命令 | 非交互式确认/拒绝（`respond <msg_id> y/n [feedback]`） |
| 拒绝 + 反馈 | `n` 时可以附带文字反馈，写入 `human_feedback` 字段 |

**理解要点**：Manager 发完确认请求就退出（不阻塞等待），Human 随时用 `human_cli.py` 处理。下次 `main.py` 启动时，Manager 通过 `check-human` 命令检查确认状态。

---

#### 第四步：看 Manager 的四个场景如何驱动

**对应课文**：第三节"三个工程介入点"

**阅读文件**：`workspace/manager/agent.md`

找到四个场景的工作规范：

| 场景 | 触发条件 | Manager 做什么 | 结束动作 |
|------|---------|---------------|---------|
| 场景1 | 新项目请求 | init_project → 需求澄清 → 写 requirements.md → 通知 Human | 发 `needs_confirm` 到 human.json |
| 场景2 | Human 确认需求后 | 选择 SOP → 通知 Human 确认 | 发 `sop_confirm` 到 human.json |
| 场景3 | SOP 确认后 | 给 PM 发 `task_assign` | 发邮件到 pm.json |
| 场景4 | PM 完成后 | 读产品文档 → 验收 → 写报告 | 标记 task_done |

**理解要点**：每个场景都以"发消息然后退出"结束——Manager 从不阻塞等待。人类介入是自然的"断点"，不是人为的阻塞。

---

#### 第五步：看新增的 Skills

**对应课文**：第三节对应的工程实现

**阅读文件**：`workspace/manager/skills/` 下的新增 Skill

| Skill | 类型 | 作用 |
|-------|------|------|
| `requirements_discovery/SKILL.md` | reference | 四维需求框架（目标/边界/约束/风险） |
| `sop_creator/SKILL.md` | reference | SOP 模板设计（角色/步骤/检查点/质量标准） |
| `sop_selector/SKILL.md` | task | 从 SOP 库选最佳匹配 → 复制为 `active_sop.md` |
| `notify_human/SKILL.md` | reference | 通知 Human 的规则和 5 种消息类型 |

**理解要点**：需求澄清和 SOP 选择不是硬编码在代码里的流程，而是通过 Skill 文件描述的思考框架。Agent 按 Skill 指引决定何时需要通知 Human。

---

#### 第六步：理解 SOP 共创——时点 A

**对应课文**：第三节"SOP 制定"

**阅读文件**：`sop_setup.py`（~20行）

| 重点 | 看什么 |
|------|--------|
| 入口逻辑 | 启动 Manager，指定加载 `sop_creator` Skill |
| SOP 产出 | 写入 `workspace/shared/sop/` 目录 |
| Human 确认 | Manager 发确认请求到 `human.json`，Human 通过 `human_cli.py` 审核 |

**理解要点**：SOP 共创（时点A）和任务执行（时点B）完全解耦——SOP 是提前准备好的"作战计划"，不需要每次执行任务时重新制定。

---

#### 第七步：验证——跑测试看效果

```bash
cd /path/to/crewai_mas_demo

# 1. 单元测试（18个，不需要沙盒/API）
python3 -m pytest m4l27/test_m4l27.py -v -k "not needs_llm"

# 2. 重点关注这些测试：
#    - TestSinglePointOfContact：PM 写 human.json 被拒绝
#    - TestHumanJsonSchema：human.json 的二态 schema 验证
#    - TestHumanCli：交互式和命令式确认/拒绝
#    - TestCheckHumanCommand：Manager 轮询 Human 确认状态
```

---

### 学习检查清单

完成以上七步后，你应该能回答：

- [ ] 为什么 PM 不能直接给 Human 发邮件？（单一接口原则，代码级强制）
- [ ] Agent 邮箱用三态，Human 邮箱用二态，为什么设计不同？
- [ ] Manager 如何知道 Human 已经确认了？（`check-human` 命令轮询 `human.json`）
- [ ] 如果 Human 拒绝了需求文档，反馈信息存在哪里？（`human.json` 的 `human_feedback` 字段）
- [ ] 为什么 SOP 制定（时点A）和任务执行（时点B）要解耦？
- [ ] 如果去掉所有 Human 确认节点，代码需要改多少？（零行——只是不再往 human.json 发消息）
