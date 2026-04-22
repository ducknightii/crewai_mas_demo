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
