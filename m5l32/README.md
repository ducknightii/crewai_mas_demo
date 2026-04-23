# 第30课：可观测性——Hook 骨架 + Langfuse 全链路追踪

本课构建 Hook 框架骨架，将前序课程的零散 Hook 统一到 5+2 事件体系，并接入 Langfuse 实现全链路追踪。

> **核心教学点**：5+2 事件类型（对齐 Agent Turn 周期）、两层 Hook 配置（全局+Workspace）、HookRegistry 分发机制、CrewAI 适配层映射、Langfuse Docker 自托管追踪

---

## 运行演示前（重要）

### 1. 确保 Langfuse 运行中

本课使用 Docker 自托管的 Langfuse（6 容器）。如果尚未搭建：

```bash
# 使用课程提供的 docker-compose（或你自己的 Langfuse 实例）
cd /path/to/langfuse
docker compose up -d
# 等待 2-3 分钟，访问 http://localhost:3000
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 DashScope API Key 和 Langfuse 密钥
```

---

## 目录结构

```
m5l30/
├── hook_framework/                     # Hook 框架核心
│   ├── __init__.py                     # 导出公共接口
│   ├── registry.py                     # F1-F2: EventType(5+2) + HookContext + HookRegistry
│   ├── loader.py                       # F3-F4: hooks.yaml 解析 + importlib 两层自动加载
│   └── crew_adapter.py                 # F5: CrewAI 4种机制 → HookRegistry 7种事件
├── shared_hooks/                       # 全局 Hook（所有 Agent 共享）
│   ├── hooks.yaml                      # 全局配置：事件 → handler 映射
│   ├── structured_log.py               # F6: 结构化 JSON 日志（stderr）
│   └── langfuse_trace.py              # F7: Langfuse 追踪（trace + generation + span）
├── workspace/                          # 演示 Workspace
│   └── demo_agent/
│       └── hooks/
│           ├── hooks.yaml              # Workspace 配置
│           └── task_audit.py           # F8: 任务审计日志
├── demo.py                             # F10: 单 Agent 端到端演示
├── agents.yaml                         # Agent 定义
├── tasks.yaml                          # Task 定义
├── .env.example                        # 环境变量模板
├── tests/
│   ├── test_registry.py                # T1-T5 + T_extra1: HookRegistry 单元测试
│   ├── test_loader.py                  # T6-T8 + T_extra3: HookLoader 单元测试
│   ├── test_handlers.py                # T9-T11: handler 输出测试
│   ├── test_adapter.py                 # T12-T14 + T_extra2/4: 适配层测试
│   └── conftest.py                     # pytest fixtures
└── README.md
```

---

## 核心设计：5+2 事件类型

对齐 Agent Turn 周期，将 CrewAI 的 4 种 Hook 机制统一映射到 7 种事件：

```
BEFORE_TURN ──→ BEFORE_LLM ──→ [LLM] ──→ BEFORE_TOOL_CALL ──→ [工具] ──→ AFTER_TOOL_CALL ──→ AFTER_TURN
                                           （无工具调用时直接 → AFTER_TURN）
```

| 事件 | CrewAI 实现机制 | 说明 |
|------|---------------|------|
| BEFORE_TURN | `@before_llm_call` + 轮次计数 | 每轮首次 LLM 调用时触发 |
| BEFORE_LLM | `@before_llm_call` | 每次 LLM 调用 |
| BEFORE_TOOL_CALL | `@before_tool_call` | 工具执行前 |
| AFTER_TOOL_CALL | `@after_tool_call` | 工具执行后 |
| AFTER_TURN | `step_callback` | 一步推理完成 |
| TASK_COMPLETE | `task_callback` | 任务完成 |
| SESSION_END | 手动调用 | 清理 + flush |

---

## 两层 Hook 架构

```
shared_hooks/                    ← 全局（基线可观测，所有 Agent 共享）
  hooks.yaml                     # 结构化日志 + Langfuse 追踪
  structured_log.py
  langfuse_trace.py

workspace/demo_agent/hooks/      ← Workspace（业务定制，仅本 Agent）
  hooks.yaml                     # 任务审计
  task_audit.py
```

- **全局层**：日志 + Langfuse，是每个 Agent 都应该有的基线保障
- **Workspace 层**：特定 Agent 的业务需求（如审计、告警）
- **加载顺序**：全局先加载 → Workspace 追加

---

## 快速开始

```bash
cd /path/to/crewai_mas_demo/m5l30

# 1. 配置环境变量
cp .env.example .env
# 编辑 .env 填入密钥

# 2. 运行测试（无需 LLM/Langfuse）
python3 -m pytest tests/ -v

# 3. 运行端到端演示（需 LLM + Langfuse）
python3 demo.py
python3 demo.py "你自己选一个AI领域的话题"
```

---

## 运行效果

```
🔗 Session: sess_20260422_150145
📦 HookRegistry: 11 handlers loaded
   [global] structured_log.before_turn_handler → before_turn
   [global] structured_log.before_llm_handler → before_llm
   [global] langfuse_trace.before_llm_handler → before_llm
   ...
   [workspace] task_audit.write_audit_entry → task_complete
   [global] langfuse_trace.flush_and_close → session_end

🚀 Starting crew for topic: AI Agent 可观测性

{"timestamp":"...","event":"before_turn","session_id":"...","turn":1,"agent_id":"Research Analyst"}
{"timestamp":"...","event":"before_llm","session_id":"...","turn":1,"agent_id":"Research Analyst"}
{"timestamp":"...","event":"before_tool_call","session_id":"...","turn":1,"tool":"knowledge_search"}
{"timestamp":"...","event":"after_tool_call","session_id":"...","turn":1,"tool":"knowledge_search"}
{"timestamp":"...","event":"before_llm","session_id":"...","turn":1,"agent_id":"Research Analyst"}
{"timestamp":"...","event":"after_turn","session_id":"...","turn":1}

📊 Result: ...
🔗 Langfuse: http://localhost:3000
📝 Audit log: workspace/demo_agent/audit.log
```

运行后可在以下位置查看结果：
1. **终端 stderr**：每个事件的结构化 JSON 日志
2. **Langfuse Dashboard**（http://localhost:3000）：Trace 树——tool span + generation + task-complete
3. **workspace/demo_agent/audit.log**：任务审计 JSON 条目

---

## 测试（20 个用例）

```bash
# 单元测试（无需 LLM/Langfuse）
python3 -m pytest tests/ -v --ignore=tests/test_e2e_hooks.py

# 全部测试（需 LLM API + Langfuse）
python3 -m pytest tests/ -v
```

| 文件 | 编号 | 测试内容 |
|------|------|---------|
| test_registry.py | T1-T5 | 注册/分发/多handler/无handler/summary/count |
| test_registry.py | T_extra1 | handler 异常不中断后续 handler |
| test_loader.py | T6-T8 | yaml 加载/两层合并/缺 yaml |
| test_loader.py | T_extra3 | 不存在的模块跳过不报错 |
| test_handlers.py | T9-T11 | 日志 JSON schema/全事件覆盖/审计写文件 |
| test_adapter.py | T12-T14 | BEFORE_TURN 计数/step→AFTER_TURN/轮次重置 |
| test_adapter.py | T_extra2 | cleanup 清理全局 hooks |
| test_adapter.py | T_extra4 | tool call 事件映射 |
| test_e2e_hooks.py | T15 | **全链路**：真实 Crew → 7种事件×2层 hook 全部触发 |
| test_e2e_hooks.py | T16 | **Langfuse**：真实 Crew → trace 含 TOOL + GENERATION observations |
