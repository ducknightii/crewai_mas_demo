# 第8课：定义 Task——从"步骤控制"到"契约驱动"

本课演示用 Pydantic 模型定义 Task 的输出"契约"，让 Agent 产出结构化、类型安全的数据。

> **核心教学点**：`output_pydantic` 契约驱动、Mock 数据模式、`kickoff(inputs={...})` 动态注入

---

## 目录结构

```
m2l4/
└── m2l4_task.py    # 契约驱动 Task 演示
```

---

## 快速开始

```bash
cd /path/to/crewai_mas_demo
python3 m2l4/m2l4_task.py
```

---

## 课堂代码演示学习指南

### 学习路线

**阅读文件**：`m2l4_task.py`（200 行）

---

#### 第一步：看 Pydantic 契约定义

| 模型 | 角色 |
|------|------|
| `ImageAnalysis` | 单张图片分析结构（主体、氛围、质量分） |
| `VisualAnalysisReport` | 视觉分析聚合报告 |
| `ContentStrategyBrief` | 最终交付"契约"——8 个具体字段 |

**理解要点**：`ContentStrategyBrief` 就是 Task 的"契约"——Agent 的输出必须严格符合这个结构，不能自由发挥。

---

#### 第二步：看 Mock 数据模式

```python
mock_report = VisualAnalysisReport(...)  # 模拟上游 Task 的输出
```

**理解要点**：上游 Task（图片分析）还没实现，用 Mock 数据先开发下游。这是开发阶段的标准做法。

---

#### 第三步：看动态注入和结构化输出

| 机制 | 代码 |
|------|------|
| 动态注入 | `crew.kickoff(inputs={"visual_report": mock_report})` |
| 契约绑定 | `Task(output_pydantic=ContentStrategyBrief)` |
| 结构化访问 | `task.output.pydantic.suggested_title` |

---

### 学习检查清单

- [ ] `output_pydantic` 的作用是什么？（强制 Agent 输出符合 Pydantic 模型的结构化数据）
- [ ] Mock 数据模式解决什么问题？（上游未实现时先开发下游）
- [ ] `kickoff(inputs={...})` 如何传递数据？（通过模板变量 `{visual_report}` 注入 Task 描述）
