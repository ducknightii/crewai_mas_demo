# 第15课：王牌超能力——代码解释器与无头浏览器

本课演示如何通过 AIO-Sandbox（Docker 容器）为 Agent 赋予代码执行和浏览器操控能力，实现 Manager-Worker 委派模式完成金融数据分析任务。

> **核心教学点**：AIO-Sandbox Docker 部署、MCP 工具集成、Manager-Worker 委派模式、Pydantic 复杂嵌套输出、IntermediateTool 中间产物

---

## 目录结构

```
m2l10/
├── m2l10_sandbox.py       # 主演示：双 Agent + Sandbox
├── m2l10_mcp_tools.md     # AIO-Sandbox 暴露的 33 个 MCP 工具文档
└── agent.log              # 运行日志
```

---

## 快速开始

```bash
# 1. 先启动 AIO-Sandbox Docker 容器
docker run -d --security-opt seccomp=unconfined --rm -it -p 8022:8080 ghcr.io/agent-infra/sandbox:latest

# 2. 运行演示
cd /path/to/crewai_mas_demo
python3 m2l10/m2l10_sandbox.py
```

---

## 课堂代码演示学习指南

### 整体架构一览

```
用户需求："阿里港股早报"
         │
         ▼
┌─────────────────────────────┐
│  assistant_agent（总管）       │
│  allow_delegation = True     │
│  tools: IntermediateTool     │
│                             │
│  五步工作法：                  │
│  1. 理解需求                  │
│  2. 规划步骤                  │
│  3. 委派子任务（JSON Schema） │
│  4. 管理进度                  │
│  5. 返回最终结果               │
└──────────┬──────────────────┘
           │ delegate_work_to_coworker
           ▼
┌─────────────────────────────┐
│  sandbox_agent（工人）        │
│  allow_delegation = False    │
│  mcps: MCPServerHTTP         │
│                             │
│  三大超能力：                  │
│  ├─ sandbox_execute_code     │  ← Python/JS 代码执行
│  ├─ browser_navigate         │  ← 无头浏览器
│  └─ sandbox_file_operations  │  ← 文件读写
└──────────┬──────────────────┘
           │ MCP HTTP
           ▼
┌─────────────────────────────┐
│  AIO-Sandbox（Docker）       │
│  localhost:8022/mcp          │
│  33 个 MCP 工具              │
└─────────────────────────────┘
```

### 学习路线

---

#### 第一步：理解双 Agent 架构

**阅读文件**：`m2l10_sandbox.py`（52-125 行）

| Agent | 角色 | delegation | 核心工具 |
|-------|------|-----------|---------|
| `sandbox_agent` | 万能沙盒工人 | `False`（只执行） | MCPServerHTTP → 33 个 Sandbox 工具 |
| `assistant_agent` | 总管 | `True`（只委派） | IntermediateTool（保存中间思考） |

**理解要点**：总管永远不直接执行任务，工人永远不主动委派——这是 Manager-Worker 模式的核心约束。总管的 backstory 明确禁止"自己编造数据"。

---

#### 第二步：看 MCP 连接与 Bug 修复

**阅读文件**：`m2l10_sandbox.py`（85-95 行）

```python
mcps=[MCPServerHTTP(
    url="http://localhost:8022/mcp",
    streamable=True,
    cache_tools_list=True,
)]
```

**理解要点**：注意 92-95 行的手动工具加载——这是一个 CrewAI Bug 的 workaround。当使用 `allow_delegation` 时，`execute_task` 不会调用 `_prepare_kickoff`，导致 MCP 工具未自动加载。代码通过 `get_mcp_tools()` 手动初始化解决。

---

#### 第三步：看复杂 Pydantic 嵌套输出

**阅读文件**：`m2l10_sandbox.py`（130-211 行）

```
AlibabaMorningReport
  ├── today: str
  ├── latest_data: LatestData
  │     ├── price, volume, market_cap, pe_ratio
  │     └── kline_30d: List[KlineData]
  │           └── date, open, high, low, close, volume
  ├── quantitative_analysis: str
  ├── sentiment_analysis: str
  └── final_report: str
```

**理解要点**：三层嵌套的 Pydantic 模型——`KlineData` 嵌入 `LatestData`，`LatestData` 嵌入 `AlibabaMorningReport`。`output_pydantic` 确保无论 Agent 内部流程多复杂，最终输出都符合这个结构。

---

#### 第四步：看任务描述中的行为约束

**阅读文件**：`m2l10_sandbox.py`（221-238 行）

| 约束 | 作用 |
|------|------|
| "所有数据必须为真实数据" | 禁止 LLM 编造金融数据 |
| "使用 Yahoo Finance" | 指定数据源 |
| "使用百度搜索" | 指定搜索引擎（中文互联网） |
| `max_iter=100` | 允许足够多的迭代完成复杂任务 |

**理解要点**：金融分析是真实场景，`max_iter=100` 反映了现实——涉及代码执行、浏览器操作、数据分析的任务需要大量 Agent-Tool 交互轮次。

---

#### 第五步：浏览 MCP 工具文档

**阅读文件**：`m2l10_mcp_tools.md`

AIO-Sandbox 暴露的 33 个工具分三类：

| 类别 | 工具数 | 典型工具 |
|------|--------|---------|
| 沙盒执行 | 9 | `sandbox_execute_code`, `sandbox_execute_bash`, `sandbox_file_operations` |
| 浏览器交互 | 14+ | `browser_navigate`, `browser_click`, `browser_form_input_fill` |
| 视觉分析 | 3 | `browser_vision_screen_capture`, `browser_vision_screen_click` |

---

### 学习检查清单

- [ ] Manager-Worker 模式的核心约束是什么？（Manager 只委派不执行，Worker 只执行不委派）
- [ ] 为什么需要手动调用 `get_mcp_tools()`？（CrewAI Bug：delegation 路径下 MCP 工具不自动加载）
- [ ] `max_iter=100` 的含义？（Agent 最多执行 100 轮 Thought-Action-Observation 循环）
- [ ] `IntermediateTool` 的作用？（给 Manager 一个"暂存"中间推理结果的工具，支持多步规划）
- [ ] AIO-Sandbox 提供了哪三大能力？（代码执行、无头浏览器、文件系统操作）
