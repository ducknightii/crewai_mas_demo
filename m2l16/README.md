# 第16课：Skill 生态——渐进式披露与 Sub-Crew 编排

本课演示 SkillLoaderTool 的设计：通过"两阶段加载"实现渐进式披露，用 Sub-Crew 工厂模式在沙盒中执行技能任务。

> **核心教学点**：渐进式披露（Progressive Disclosure）、SkillLoaderTool 元工具、Sub-Crew 工厂模式、Reference vs Task 技能类型、MCP 最小权限

---

## 目录结构

```
m2l16/
├── m2l16_skills.py                # 合并后的单文件演示（含 SkillLoaderTool + Crew）
├── crews/                         # 原始模块化版本（源码在 git 历史中）
│   └── __pycache__/               # skill_crew.py, main_crew.py 的编译缓存
├── tools/                         # 原始模块化版本
│   └── __pycache__/               # skill_loader_tool.py 的编译缓存
├── tests/                         # 原始模块化版本
│   └── __pycache__/               # 测试文件的编译缓存
└── agent.log                      # 运行日志
```

> 💡 原始模块化版本在 git 历史 commit `680822a` 中保留完整，建议对照学习。

---

## 快速开始

```bash
# 1. 启动 AIO-Sandbox
docker run -d --security-opt seccomp=unconfined --rm -it -p 8022:8080 ghcr.io/agent-infra/sandbox:latest

# 2. 运行演示
cd /path/to/crewai_mas_demo
python3 m2l16/m2l16_skills.py
```

---

## 课堂代码演示学习指南

### 整体架构一览

```
┌──────────────────────────────────────────────┐
│  Orchestrator Agent                           │
│  tools: [SkillLoaderTool]                     │
│                                              │
│  Phase 1: 看"菜单"                             │
│  SkillLoaderTool.description 中注入：           │
│  <available_skills>                           │
│    <skill name="pdf" type="task">             │
│      将 PDF 转换为结构化 Markdown...            │
│    </skill>                                   │
│    <skill name="docx" type="task">            │
│      生成格式化 Word 文档...                    │
│    </skill>                                   │
│  </available_skills>                          │
└──────────────────┬───────────────────────────┘
                   │
                   │  Phase 2: 选菜并执行
                   │  _run(skill_name="pdf", task_context={...})
                   ▼
         ┌─────────────────────┐
         │ 判断 skill.type      │
         ├─────────┬───────────┤
         │         │           │
    type=reference │      type=task
         │         │           │
    返回指令文本    │     build_skill_crew()
    （无代码执行）   │           │
                   │           ▼
                   │  ┌─────────────────────┐
                   │  │  Sub-Crew（独立实例）  │
                   │  │  Agent.backstory =    │
                   │  │    SKILL.md 指令      │
                   │  │  mcps: Sandbox MCP    │
                   │  │  tool_filter: 4个工具  │
                   │  └─────────────────────┘
                   │           │
                   │           ▼
                   │     AIO-Sandbox（Docker）
                   │     sandbox_execute_code
                   │     sandbox_execute_bash
                   │     sandbox_file_operations
                   │     sandbox_str_replace_editor
```

### 学习路线

---

#### 第一步：理解渐进式披露

**阅读文件**：`m2l16_skills.py`（搜索 `_build_description`）

| 阶段 | 时机 | 信息量 | 来源 |
|------|------|--------|------|
| Phase 1 | 工具初始化时 | 仅 YAML frontmatter（名称+简介） | `load_skills.yaml` + `SKILL.md` 前缀 |
| Phase 2 | 工具被调用时 | 完整 SKILL.md 指令 | `_get_skill_instructions()` |

**理解要点**：核心设计理念——Main Agent 的 context 是稀缺资源。Phase 1 只注入"菜单"（几十个字的 XML 描述），不浪费 token 在完整指令上。Phase 2 按需加载，且结果会被缓存（`is` 检查证明返回同一个对象）。

---

#### 第二步：看 SkillLoaderInput 的输入约束

**阅读文件**：`m2l16_skills.py`（搜索 `SkillLoaderInput`）

| 字段 | 约束 |
|------|------|
| `skill_name` | 必须是 `<available_skills>` 中列出的名称 |
| `task_context` | 必须包含 4 项：输入文件路径、期望输出结构、输出文件路径、特殊格式要求 |

**理解要点**：约束定义在 `Field(description=...)` 中，而不是 Agent 的 backstory 中。这样 LLM 在生成工具调用参数时就能看到具体要求，比 backstory 中的模糊描述更有效。

---

#### 第三步：看 Sub-Crew 工厂模式

**阅读文件**：`m2l16_skills.py`（搜索 `build_skill_crew`）

| 设计决策 | 原因 |
|---------|------|
| 每次调用创建新 Crew | 防止状态污染（上次执行的残留不影响下次） |
| MCP 连接 Sandbox | Sub-Crew 在 Docker 容器内执行代码 |
| 静态工具过滤 | 只暴露 4 个工具，排除所有 browser_* |
| Skill 指令注入 backstory | 完整 SKILL.md 成为 Agent 的"技能手册" |

**理解要点**：工厂模式是关键——`build_skill_crew()` 返回全新实例，不复用。这与 m4l23 的 `_run_one_sub_crew()` 是同一个模式：上下文隔离。

---

#### 第四步：看 Skill 注册表

**阅读文件**：`skills/load_skills.yaml`

```yaml
skills:
  - name: pdf
    path: skills/pdf
    type: task      # 需要 Sub-Crew 执行
    enabled: true
  - name: docx
    path: skills/docx
    type: task
    enabled: true
```

**理解要点**：`type` 决定执行方式——`task` 类型启动独立 Sub-Crew（有代码执行能力），`reference` 类型直接返回指令文本（无代码执行，用于知识注入）。

---

#### 第五步：看异步双通道

**阅读文件**：`m2l16_skills.py`（搜索 `_arun` 和 `_run`）

| 通道 | 场景 | 实现 |
|------|------|------|
| `_arun()` | FastAPI / `akickoff()` | 原生 `await` |
| `_run()` | 同步调用 / CLI | `ThreadPoolExecutor` 包装 |

**理解要点**：Python 的"嵌套事件循环"问题——在已有事件循环中不能再 `asyncio.run()`。`_run()` 通过线程池绕过这个限制。

---

### 学习检查清单

- [ ] 渐进式披露解决了什么问题？（避免完整 SKILL.md 指令占满 Main Agent 的 context window）
- [ ] Sub-Crew 为什么每次都创建新实例？（防止状态污染——上次执行的文件、变量不影响下次）
- [ ] Reference 和 Task 两种 Skill 类型的区别？（Reference 返回文本，Task 启动 Sub-Crew 在沙盒执行）
- [ ] 为什么用 `Field(description=...)` 而不是 backstory 定义约束？（LLM 生成工具参数时能直接看到约束，比 backstory 更精准）
- [ ] 工具过滤白名单有哪 4 个工具？（`sandbox_execute_bash`, `sandbox_execute_code`, `sandbox_file_operations`, `sandbox_str_replace_editor`）
