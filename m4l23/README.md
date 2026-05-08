# 第23课：编排者模式——动态角色创建与并行 Sub-Crew

本课演示 Orchestrator 编排者模式：运行时动态创建 Sub-Agent 角色，通过 `SpawnSubAgentTool` 串行执行和 `SpawnParallelTool` 并行执行，完成从需求到交付的完整软件开发流程。

> **核心教学点**：Orchestrator 模式 vs Workflow 模式、运行时角色创建、`SpawnSubAgentTool`（串行）、`SpawnParallelTool`（并行）、文件引用传递、SOP Skill 注入

---

## 目录结构

```
m4l23/
├── m4l23_orchestrator.py          # 核心实现：编排者 + Spawn 工具
├── test_orchestrator.py           # 8 个单元测试
└── workspace/
    ├── requirements.md            # 示例需求：员工请假管理系统
    ├── design/                    # 编排者产出：架构文档 + API 规格
    ├── mock/                      # 编排者产出：Mock 服务器代码
    └── tests/                     # 编排者产出：测试文件
```

---

## 快速开始

```bash
cd /path/to/crewai_mas_demo
python3 m4l23/m4l23_orchestrator.py
```

---

## 课堂代码演示学习指南

### 整体架构一览

```
需求文档（requirements.md）
         │
         ▼
┌─────────────────────────────────┐
│  Orchestrator Agent              │
│  tools:                          │
│  ├─ SpawnSubAgentTool（串行）     │
│  ├─ SpawnParallelTool（并行）     │
│  └─ FileReadTool（读取产出）      │
│                                 │
│  backstory 注入 SOP Skill：       │
│  设计 → Mock/测试 → 开发 →       │
│  Review → 修复 → 交付            │
│                                 │
│  运行时决策：                     │
│  • 创建什么角色？                 │
│  • 串行还是并行？                 │
│  • 传递什么上下文？               │
│  • 产出质量是否合格？             │
└──────────┬──────────────────────┘
           │
     ┌─────┴──────┐
     ▼            ▼
  串行执行       并行执行
  SpawnSub      SpawnParallel
     │            │
     ▼            ▼
  ┌──────┐    ┌──────┬──────┐
  │Sub   │    │Sub   │Sub   │
  │Crew  │    │Crew  │Crew  │
  │  A   │    │  B   │  C   │
  └──┬───┘    └──┬───┘──┬───┘
     │           │      │
     ▼           ▼      ▼
  output_A.md  output_B.md  output_C.md
  （文件引用传递，不是内容传递）
```

### 学习路线

---

#### 第一步：看工具注册表

**阅读文件**：`m4l23_orchestrator.py`（搜索 `TOOL_REGISTRY`）

| 工具名 | 类 | 作用 |
|--------|-----|------|
| `FileReadTool` | WorkspaceFileReadTool | 读取工作空间文件（带路径修复） |
| `FileWriterTool` | WorkspaceFileWriterTool | 写入工作空间文件（带路径修复） |
| `BashTool` | BashTool | 在工作空间执行 Shell 命令 |

**理解要点**：这是 Sub-Agent 的"工具池"——Orchestrator 在创建 Sub-Agent 时指定 `tool_names`，从池中挑选工具。不是每个 Sub-Agent 都能用所有工具。

---

#### 第二步：看 SpawnSubAgentTool（串行）

**阅读文件**：`m4l23_orchestrator.py`（搜索 `SpawnSubAgentTool`）

| 参数 | 类型 | 含义 |
|------|------|------|
| `role` | str | 运行时创建的角色名（如"前端开发者"） |
| `goal` | str | 本次任务的具体目标 |
| `context` | str | 传递给 Sub-Agent 的上下文（通常是文件路径） |
| `task_description` | str | 详细的任务描述 |
| `output_file` | str | 产出文件路径 |
| `tool_names` | List[str] | 从 TOOL_REGISTRY 中选取 |

**理解要点**：角色不是预定义的——Orchestrator 根据当前任务需要，在运行时"发明"角色。同一个 Orchestrator 可以创建"架构师"、"测试工程师"、"调试专家"等任意角色。

---

#### 第三步：看 SpawnParallelTool（并行）

**阅读文件**：`m4l23_orchestrator.py`（搜索 `SpawnParallelTool`）

```python
with ThreadPoolExecutor(max_workers=len(sub_tasks)) as pool:
    futures = {pool.submit(_run_one_sub_crew, ...): st for st in sub_tasks}
```

**理解要点**：并行执行使用 `ThreadPoolExecutor`——每个 Sub-Crew 在独立线程中运行，互相看不到对方的上下文。单个 Sub-Crew 失败不会阻塞其他任务（错误被捕获为 `"error: ..."`）。

---

#### 第四步：理解文件引用传递

**阅读文件**：`m4l23_orchestrator.py`（搜索 `_run_one_sub_crew`）

```
错误做法：把文件内容塞进 context
  → context 膨胀 → 超出 token 限制

正确做法：传递文件路径
  → Sub-Agent 用 FileReadTool 按需读取
  → context 保持精简
  → 产出持久化为文件供下游引用
```

**理解要点**：这是 Orchestrator 模式的核心设计——Sub-Agent 之间通过文件系统间接通信，不通过 context 传递大量内容。Orchestrator 只传递"读哪个文件"和"写到哪个文件"。

---

#### 第五步：看 LLM 路径幻觉防御

**阅读文件**：`m4l23_orchestrator.py`（搜索 `_resolve_workspace_path`）

| 问题 | 修复方法 |
|------|---------|
| LLM 生成 `Users/xiao/.../workspace/` | 去掉无效前缀 |
| LLM 生成 `workspace/workspace/file.md` | 折叠重复段 |
| 路径没有 `workspace/` 前缀 | 自动补上 |

**理解要点**：LLM 经常"幻觉"出不存在的路径（尤其是包含用户名的路径）。这三个修复函数是工程实践中的经验积累——在代码中防御，不靠 prompt 约束。

---

#### 第六步：看 SOP Skill 注入

**阅读文件**：`m4l23_orchestrator.py`（搜索 `load_sop_skill`）

**理解要点**：Orchestrator 的工作流程不是硬编码的——它从外部 `SKILL.md` 文件读取 SOP（设计→Mock/测试→开发→Review→修复→交付），注入 backstory。更换 SOP 文件就能改变 Orchestrator 的工作方式，无需改代码。

---

#### 第七步：运行测试

```bash
cd /path/to/crewai_mas_demo
python3 -m pytest m4l23/test_orchestrator.py -v
```

| 测试 | 验证点 |
|------|--------|
| T1 | spawn_sub_agent 创建输出文件 |
| T2 | 并行执行快于串行（wall-time 验证） |
| T3 | 未知工具名被安全过滤 |
| T4-T5 | 异常被捕获，不影响其他任务 |
| T6-T7 | SOP 文件存在且注入 backstory |
| T8 | 并行 Sub-Crew 上下文完全隔离 |

---

### 学习检查清单

- [ ] Orchestrator 模式和 Workflow 模式的核心区别？（Orchestrator 没有预定义任务图，运行时动态决定）
- [ ] 为什么传文件路径而不是文件内容？（避免 context 膨胀，产出持久化供多个下游引用）
- [ ] `SpawnParallelTool` 中单个任务失败会怎样？（错误被捕获，其他任务继续执行）
- [ ] 为什么每次 spawn 都创建新 Crew？（上下文隔离——Sub-Agent 不能看到 Orchestrator 或其他 Sub-Agent 的内部状态）
- [ ] LLM 路径幻觉的三种常见模式？（无效用户路径前缀、workspace 重复、缺少 workspace 前缀）
- [ ] SOP Skill 如何影响 Orchestrator 行为？（注入 backstory 提供工作流模板，更换 SKILL.md 即可改变流程）
