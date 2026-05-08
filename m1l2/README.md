# 第2课：解构智能体——Agent 的解剖学与 ReAct 范式

本课从零构建一个 Agent，理解 ReAct 循环的本质，然后用 CrewAI 框架实现同样的功能。

> **核心教学点**：Agent 三要素（Role/Goal/Backstory）、ReAct 循环（Thought→Action→Observation→Final Answer）、手写 Agent vs 框架 Agent 的对比

---

## 目录结构

```
m1l2/
├── m1l2_raw_agent.py          # 手写 ReAct Agent（不用框架）
├── m1l2_agent.py              # CrewAI 框架 Agent
├── agent_system_prompt.txt    # ReAct 系统提示词模板
├── agent_user_prompt.txt      # 用户提示词模板
└── 极客时间-最终报告.md        # 示例产出
```

---

## 快速开始

```bash
cd /path/to/crewai_mas_demo

# 手写 Agent（理解原理）
python3 m1l2/m1l2_raw_agent.py

# CrewAI Agent（框架封装）
python3 m1l2/m1l2_agent.py
```

---

## 课堂代码演示学习指南

### 整体架构一览

```
┌──────────────────────────────────────────────────────┐
│  m1l2_raw_agent.py（手写 ReAct）                      │
│                                                      │
│  system_prompt + user_prompt                         │
│       ↓                                              │
│  while True:                                         │
│    LLM.call(stop=["Observation:"])                   │
│       ↓                                              │
│    解析 Action / Action Input                         │
│       ↓                                              │
│    执行工具 → 拼接 Observation                        │
│       ↓                                              │
│    检测 Final Answer → 退出                           │
└──────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────┐
│  m1l2_agent.py（CrewAI 封装）                         │
│                                                      │
│  Agent(role, goal, backstory, tools)                 │
│       ↓                                              │
│  Task(description, expected_output)                  │
│       ↓                                              │
│  Crew(agents, tasks).kickoff()                       │
│  → 框架内部自动完成 ReAct 循环                        │
└──────────────────────────────────────────────────────┘
```

### 学习路线（建议按顺序阅读）

---

#### 第一步：理解 ReAct 提示词模板

**阅读文件**：`agent_system_prompt.txt` + `agent_user_prompt.txt`

| 重点 | 看什么 |
|------|--------|
| 系统提示词 | `{role}`, `{goal}`, `{backstory}` 占位符——Agent 三要素 |
| ReAct 格式 | `Thought:` → `Action:` → `Action Input:` → `Observation:` 循环 |
| 终止条件 | `Final Answer:` 标记结束 |
| 工具注入 | `{tools_map}` 和 `{tools_name}` 告诉 Agent 可用工具 |

---

#### 第二步：看手写 Agent——理解 ReAct 循环的本质

**对应课文**：第二节"ReAct 范式"

**阅读文件**：`m1l2_raw_agent.py`

| 重点区域 | 看什么 |
|---------|--------|
| `RawAgent.__init__()` | 加载 prompt 模板 + 注册工具 |
| `_generate_prompt()` | 将 Role/Goal/Backstory + 工具列表填入模板 |
| `run()` 主循环 | `while True` → 调用 LLM（`stop=["Observation:"]`） → 解析输出 |
| Action 解析 | 正则匹配 `Action:` 和 `Action Input:` |
| 工具执行 | 根据 Action 名称查表 → 调用工具 → 结果拼接为 `Observation:` |
| Final Answer 检测 | 匹配到 `Final Answer:` 就退出循环 |

**理解要点**：`stop=["Observation:"]` 是关键——让 LLM 在输出 Action 后暂停，等待外部工具执行结果。这就是 ReAct 的"交替执行"机制。

---

#### 第三步：看 CrewAI Agent——框架如何封装 ReAct

**对应课文**：第三节"框架封装"

**阅读文件**：`m1l2_agent.py`

| 重点区域 | 看什么 |
|---------|--------|
| Agent 定义 | `role`, `goal`, `backstory`——和手写版的三要素一一对应 |
| 工具挂载 | `tools=[ScrapeWebsiteTool(), BaiduSearchTool(), FileWriterTool()]` |
| Task 定义 | `description` + `expected_output`——对应用户提示词模板 |
| Crew 执行 | `crew.kickoff()` 一行代码完成整个 ReAct 循环 |

**理解要点**：CrewAI 把手写版的 ~280 行代码封装成了 ~50 行。ReAct 循环、工具调用、结果拼接全部在框架内部完成。

---

#### 第四步：对比两个版本

| 维度 | 手写版 | CrewAI 版 |
|------|--------|----------|
| ReAct 循环 | 显式 while 循环 | 框架内部 |
| 工具执行 | 手动解析+查表 | 声明式注册 |
| 提示词 | 手写模板 | 框架自动生成 |
| 代码量 | ~280 行 | ~50 行 |
| 灵活性 | 完全可控 | 受框架约束 |

---

### 学习检查清单

- [ ] ReAct 的四个步骤是什么？（Thought → Action → Action Input → Observation）
- [ ] `stop=["Observation:"]` 的作用是什么？（让 LLM 暂停等待工具结果）
- [ ] Agent 的三要素是什么？（Role / Goal / Backstory）
- [ ] 手写版和框架版的核心区别是什么？（ReAct 循环的显式 vs 隐式实现）
