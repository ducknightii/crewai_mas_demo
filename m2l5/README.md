# 第9课：定义 Process——任务调度与信息传递

本课构建完整的三 Agent 流水线：策略 → 文案 → SEO 优化，演示 `Process.sequential` 和 Task 链式数据传递。

> **核心教学点**：`Process.sequential`、Task 依赖链（`context`）、多阶段 Pydantic 输出、Pipeline 模式

---

## 目录结构

```
m2l5/
└── m2l5_crew.py    # 三 Agent 流水线演示
```

---

## 快速开始

```bash
cd /path/to/crewai_mas_demo
python3 m2l5/m2l5_crew.py
```

---

## 课堂代码演示学习指南

### 整体架构一览

```
策略专家                   文案编辑                    SEO 优化师
  │                         │                           │
  ▼                         ▼                           ▼
Task 1: 策略              Task 2: 文案                Task 3: SEO
output_pydantic:          output_pydantic:            output_pydantic:
ContentStrategyBrief      CopywritingOutput           SEOOptimizedNoteReport
  │                         │                           │
  └──── context ────────────┘                           │
                            └──────── context ──────────┘
```

### 学习路线

---

#### 第一步：看 Pipeline 的三个阶段

**阅读文件**：`m2l5_crew.py`（365 行）

| 阶段 | Agent | 输入 | 输出 |
|------|-------|------|------|
| 策略 | content_strategist | `{visual_report}` 动态注入 | `ContentStrategyBrief` |
| 文案 | content_writer | 策略产出（via context） | `CopywritingOutput` |
| SEO | seo_optimizer | 策略 + 文案产出（via context） | `SEOOptimizedNoteReport` |

---

#### 第二步：理解 context 链

```python
task_copywriting = Task(context=[task_content_strategy])       # 拿到策略
task_seo = Task(context=[task_content_strategy, task_copywriting])  # 拿到策略+文案
```

**理解要点**：`context` 是数据传递的核心机制。SEO 任务同时依赖策略和文案两个上游，可以对照修改。

---

#### 第三步：看结果的访问方式

```python
result = crew.kickoff(inputs={...})
result.pydantic              # 最后一个 Task 的结构化输出
result.tasks_output          # 每个 Task 的独立输出
```

---

### 学习检查清单

- [ ] `Process.sequential` 保证了什么？（三个 Task 严格按顺序执行）
- [ ] `context` 参数可以传多个 Task 吗？（可以，如 SEO 同时依赖策略和文案）
- [ ] 每个 Task 可以有不同的 `output_pydantic` 吗？（可以，各自独立的契约）
- [ ] `result.tasks_output` 和 `result.pydantic` 的区别？（前者是所有 Task 的输出列表，后者是最后一个 Task 的结构化输出）
