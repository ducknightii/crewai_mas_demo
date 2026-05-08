# 第3课：Multi-Agent 系统——Agent、Task、Process 的协作美学

本课演示四个 Agent 协作完成一份深度调研报告：Researcher 拆解任务 → Writer 委派执行 → Searcher 搜索信息 → Editor 审核质量。

> **核心教学点**：Multi-Agent 协作、Task 依赖（`context`）、`Process.sequential`、Agent 委派（`allow_delegation`）

---

## 目录结构

```
m1l3/
├── m1l3_multi_agent.py             # 四 Agent 协作演示
└── 极客时间平台全面深度调研报告-*.md  # 示例产出（大纲 + 7步 + 最终报告）
```

---

## 快速开始

```bash
cd /path/to/crewai_mas_demo
python3 m1l3/m1l3_multi_agent.py
```

---

## 课堂代码演示学习指南

### 整体架构一览

```
┌─────────────────────────────────────────────────────────┐
│  Process.sequential                                      │
│                                                         │
│  Task 1: task_plan ──→ Task 2: task_write               │
│  (Researcher)          (Writer)                         │
│                          │                              │
│                    allow_delegation                      │
│                    ┌─────┼─────┐                        │
│                    ▼           ▼                         │
│                 Searcher    Editor                       │
│                 (搜索信息)   (审核报告)                    │
└─────────────────────────────────────────────────────────┘
```

### 学习路线（建议按顺序阅读）

---

#### 第一步：看四个 Agent 的角色设计

**阅读文件**：`m1l3_multi_agent.py`

| Agent | 角色 | 核心能力 | 能否委派 |
|-------|------|---------|---------|
| Researcher | 深度研究专家 | 分析任务、设计大纲 | 不能 |
| Writer | 报告撰写研究员 | 写报告、协调他人 | **可以**（核心） |
| Searcher | 网络搜索专家 | 搜索 + 抓取 | 不能 |
| Editor | 报告审核编辑 | 审核 + 反馈 | 不能 |

**理解要点**：只有 Writer 可以委派（`allow_delegation=True`）。它是"导演"角色，Searcher 和 Editor 是它的"助手"。

---

#### 第二步：理解 Task 依赖——context 参数

| Task | Agent | context | 含义 |
|------|-------|---------|------|
| `task_plan` | Researcher | 无 | 第一个任务，自主完成 |
| `task_write` | Writer | `[task_plan]` | 拿到 Researcher 的大纲后再开始 |

**理解要点**：`context=[task_plan]` 让 Writer 可以读到 Researcher 的产出。这是 Task 间的"数据传递"。

---

#### 第三步：观察委派行为

运行后观察日志，Writer 会：
1. 收到 Researcher 的大纲
2. 委派 Searcher 搜索每个子话题
3. 自己整合搜索结果写报告
4. 委派 Editor 审核
5. 根据 Editor 反馈修改

---

#### 第四步：看产出文件理解协作结果

7 个步骤文件（`步骤1.md` ~ `步骤7.md`）展示了逐步撰写过程，最终合并为`最终报告.md`。

---

### 学习检查清单

- [ ] `context` 参数的作用是什么？（让下游 Task 读到上游 Task 的产出）
- [ ] `Process.sequential` 保证了什么？（Task 按顺序执行）
- [ ] `allow_delegation=True` 让 Agent 可以做什么？（运行时将子任务委派给其他 Agent）
- [ ] 四个 Agent 中谁是"协调者"？（Writer，唯一可以委派的角色）
