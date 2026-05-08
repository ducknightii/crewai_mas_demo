# 第25课：团队角色体系——分工设计与行为规范

本课演示**通用数字员工框架**（`DigitalWorkerCrew`）：同一个类，换 workspace 就变成不同角色。
Manager 和 Dev 各自独立运行，互不通信（通信在第26课实现）。

> **核心教学点**：代码层面零角色特异性。角色身份完全由 workspace 下的四个文件决定。

---

## 目录结构

```
m4l25/
├── run_manager.py              # Manager 启动器（~30行，零角色逻辑）
├── run_dev.py                  # Dev 启动器（同上）
├── sandbox-docker-compose.yaml # 双沙盒配置
├── test_m4l25.py               # 集成测试（63个用例）
├── demo_input/
│   ├── project_requirement.md  # Manager 输入
│   └── feature_requirement.md  # Dev 输入
├── workspace/
│   ├── manager/                # Manager 的 workspace（soul/agent/user/memory）
│   └── dev/                    # Dev 的 workspace
```

---

## 快速开始

### 1. 清理环境（支持重跑）

```bash
cd /path/to/crewai_mas_demo/m4l25

# 清理 session 历史（可选，重跑时建议执行）
rm -rf workspace/manager/sessions workspace/dev/sessions

# 清理上次产出（可选）
rm -f workspace/manager/task_breakdown.md workspace/dev/tech_design.md
```

### 2. 启动沙盒

```bash
cd /path/to/crewai_mas_demo/m4l25

# 按需启动（--profile manager | dev | 同时两个）
docker compose -f sandbox-docker-compose.yaml --profile manager up -d
docker compose -f sandbox-docker-compose.yaml --profile dev up -d

# 或同时启动两个
docker compose -f sandbox-docker-compose.yaml --profile manager --profile dev up -d

# 验证（healthy 状态才可用）
docker ps | grep sandbox
curl -s http://localhost:8023/mcp | head -1  # Manager
curl -s http://localhost:8024/mcp | head -1  # Dev
```

### 3. 运行演示

从 `crewai_mas_demo/` 目录执行：

```bash
cd /path/to/crewai_mas_demo

# 演示 1：Manager — 任务拆解
python3 m4l25/run_manager.py 1>m4l25/agent_manager.log 2>m4l25/agent_manager.log.wf

# 演示 2：Dev — 技术设计
python3 m4l25/run_dev.py 1>m4l25/agent_dev.log 2>m4l25/agent_dev.log.wf
```

### 4. 运行测试

```bash
cd /path/to/crewai_mas_demo

# 离线测试（62个离线 + 1个 e2e，不需要沙盒/API）
python3 -m pytest m4l25/test_m4l25.py -v -m "not e2e"

# E2E 测试（需要沙盒 + LLM API）
python3 -m pytest m4l25/test_m4l25.py -v -m e2e -s
```

---

## 演示说明

### 演示 1：Manager

| 项目 | 说明 |
|------|------|
| 入口 | `run_manager.py` |
| Workspace | `workspace/manager/`（soul + agent含团队名册 + user + memory） |
| Skills | `workspace/manager/skills/`（sop_manager + memory-save，workspace 内置） |
| 沙盒端口 | 8023 |
| 输出 | `workspace/manager/task_breakdown.md` |

**观察重点**：Manager 的 NEVER 清单阻止它做技术设计，agent.md 中的团队名册让它知道把任务分给谁。

### 演示 2：Dev

| 项目 | 说明 |
|------|------|
| 入口 | `run_dev.py` |
| Workspace | `workspace/dev/`（soul + agent含职责边界 + user + memory） |
| Skills | `workspace/dev/skills/`（sop_dev + memory-save，workspace 内置） |
| 沙盒端口 | 8024 |
| 输出 | `workspace/dev/tech_design.md` |

**观察重点**：Dev 的 soul.md 声明「技术是唯一权威」，如果输入缺验收标准，触发 NEVER 规则退回。

---

## 常见问题

**Q：报 `ConnectionError`？**
→ 检查沙盒：`docker ps | grep sandbox`

**Q：报 `ModuleNotFoundError`？**
→ 确认从 `crewai_mas_demo/` 目录运行。

**Q：想清空重来？**
```bash
rm -rf workspace/manager/sessions workspace/dev/sessions
rm -f workspace/manager/task_breakdown.md workspace/dev/tech_design.md
```

**Q：停止沙盒？**
```bash
docker compose -f sandbox-docker-compose.yaml down
```

---

## 课堂代码演示学习指南

本节帮你按课程教学顺序阅读代码，建立完整的理解链路。

### 整体架构一览

```
┌──────────────────────────────────────────────────────────┐
│  run_manager.py / run_dev.py（~30行薄启动器）             │
│  唯一区别：workspace 目录路径 + 沙盒端口号                 │
└─────────────────────┬────────────────────────────────────┘
                      │ 实例化
                      ▼
┌──────────────────────────────────────────────────────────┐
│  shared/digital_worker.py                                │
│  DigitalWorkerCrew：通用框架，零角色特异性代码              │
│  build_bootstrap_prompt() → 加载 workspace 四件套         │
│  SkillLoaderTool → 加载 workspace-local Skills            │
└─────────────────────┬────────────────────────────────────┘
                      │ 读取
                      ▼
┌──────────────────────────────────────────────────────────┐
│  workspace/manager/ 或 workspace/dev/                    │
│  soul.md  → "我是谁"（身份 + NEVER 清单）                 │
│  agent.md → "我做什么"（职责 + 工作流程 + 团队名册）       │
│  user.md  → "我服务谁"                                   │
│  memory.md → "我记住了什么"                               │
│  skills/  → "我会什么"（SOP + 工具）                      │
└──────────────────────────────────────────────────────────┘
```

### 学习路线（建议按顺序阅读）

---

#### 第一步：理解通用框架——DigitalWorkerCrew

**对应课文**：第四节"四层框架的代码实现"

**阅读文件**：`shared/digital_worker.py`

| 重点区域 | 看什么 |
|---------|--------|
| `DigitalWorkerCrew.__init__()` | 只接收 workspace 路径和沙盒端口，没有 role 参数 |
| `worker_agent()` | `build_bootstrap_prompt(workspace/)` 加载四件套 → 注入 backstory |
| `worker_task()` | `UNIVERSAL_TASK_TEMPLATE`：通用任务模板，指示 Agent 读自己的 soul/agent_rules |
| `SkillLoaderTool` 配置 | skills_dir 指向 workspace-local 目录，不是全局目录 |

**理解要点**：这个类的 `role` 永远是 "数字员工"，`goal` 永远是通用描述。所有角色特征来自 backstory 中注入的 workspace 文件内容。换 workspace 目录 = 换角色。

---

#### 第二步：对比两个角色的 Workspace

**对应课文**：第三节"三维隔离判断法"、第四节"四层框架"

**2a. Manager workspace**

| 文件 | 重点 |
|------|------|
| `workspace/manager/soul.md` | 找到 NEVER 清单——Manager 永远不写代码、不修改需求、不跳过验收标准 |
| `workspace/manager/agent.md` | 找到团队名册（PM/Dev/QA 职责表）和任务拆解工作流 |
| `workspace/manager/skills/sop_manager/SKILL.md` | 任务拆解 SOP：5步流程 → 产出 `task_breakdown.md` |

**2b. Dev workspace**

| 文件 | 重点 |
|------|------|
| `workspace/dev/soul.md` | 找到 NEVER 清单——Dev 永远不修改需求、不跳过技术设计、不接受无验收标准的任务 |
| `workspace/dev/agent.md` | 找到职责边界表（"我负责 / 我不负责"）和上下游关系 |
| `workspace/dev/skills/sop_dev/SKILL.md` | 技术设计 SOP：7步流程 → 产出 `tech_design.md` |

**对比理解**：两个角色的文件结构完全相同（soul/agent/user/memory/skills），但内容完全不同。这就是"四层框架"的核心——框架统一，内容分化。

---

#### 第三步：看启动器如何"选角色"

**对应课文**：第四节"静态工作流 vs 动态 LLM 的分离"

**阅读文件**：`run_manager.py` 和 `run_dev.py`

两个文件各 ~30 行，核心只有一行不同：

```python
# run_manager.py
worker = DigitalWorkerCrew(workspace_dir=..., sandbox_port=8023)

# run_dev.py
worker = DigitalWorkerCrew(workspace_dir=..., sandbox_port=8024)
```

**理解要点**：零角色逻辑。如果要加一个 QA 角色，只需要创建 `workspace/qa/`（四件套 + skills），再写一个 `run_qa.py`（改路径和端口）。框架代码一行不改。

---

#### 第四步：看输入如何驱动行为

**对应课文**：演示环节

**阅读文件**：`demo_input/`

| 文件 | 给谁 | 内容 |
|------|------|------|
| `project_requirement.md` | Manager | 一个"智能日程助手"的项目需求 |
| `feature_requirement.md` | Dev | 一个"NLP 日程解析"的功能需求（假设由 Manager 分配） |

**理解要点**：Manager 收到项目级需求做拆解，Dev 收到功能级需求做技术设计。输入的粒度不同，但框架处理方式完全相同：读输入 → 加载 Skill → 按 SOP 执行 → 写产出到沙盒。

---

#### 第五步：验证——跑测试看效果

```bash
cd /path/to/crewai_mas_demo

# 1. 离线测试（62个，不需要沙盒/API）
python3 -m pytest m4l25/test_m4l25.py -v -m "not e2e"

# 2. 关注这些测试理解设计意图：
#    - TestBootstrapPromptInjection：验证四件套是否正确注入
#    - TestNeverListPresence：验证 NEVER 清单在 backstory 中
#    - TestGenericFramework：验证同一个类可以服务不同角色
```

---

### 学习检查清单

完成以上五步后，你应该能回答：

- [ ] `DigitalWorkerCrew` 的代码中有没有任何一行提到 "manager" 或 "dev"？（答案：没有）
- [ ] 如果要新增一个 QA 角色，需要改哪些文件？（答案：只需创建 `workspace/qa/` + `run_qa.py`）
- [ ] Manager 的 NEVER 清单和 Dev 的 NEVER 清单各有什么作用？（防止角色越界）
- [ ] `build_bootstrap_prompt()` 读取哪四个文件？它们分别对应四层框架的哪一层？
- [ ] soul.md 和 agent.md 的区别是什么？（soul = 决策偏好 + 身份，agent = 职责 + 工作流程）
