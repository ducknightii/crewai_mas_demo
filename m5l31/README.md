# 第31课：可靠性——重试、循环控制与成本围栏

在第30课 Hook 骨架 + Langfuse 全链路追踪的基础上，增量升级三个可靠性策略，让 Agent 不仅"看得见"，还能"自动刹车"。

> **核心教学点**：`dispatch_gate` 拦截分发、`pending_deny` 延迟抛出、三策略（RetryTracker / CostGuard / LoopDetector）、hooks.yaml `strategies` 段声明式加载

---

## 运行演示前（重要）

### 1. 确保 Langfuse + 沙盒运行中

```bash
# Langfuse（6 容器）——第30课已搭建
cd /path/to/langfuse && docker compose up -d
# 等待 2-3 分钟，访问 http://localhost:3000

# 沙盒（代码执行环境）
cd m5l31 && docker compose -f sandbox-docker-compose.yaml up -d
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入你的 DashScope API Key 和 Langfuse 密钥
```

---

## 目录结构

```
m5l31/
├── hook_framework/                         # Hook 框架核心（30课骨架 + 31课升级）
│   ├── __init__.py                         # 导出公共接口
│   ├── registry.py                         # EventType(7) + HookContext + HookRegistry
│   │                                       # 31课新增：GuardrailDeny + dispatch_gate
│   ├── loader.py                           # hooks.yaml 解析 + importlib 动态加载
│   │                                       # 31课新增：strategies 段——有状态类实例化
│   └── crew_adapter.py                     # CrewAI 机制 → 7种事件映射
│                                           # 31课新增：pending_deny 延迟抛出
├── shared_hooks/                           # 全局 Hook（观测 + 可靠性策略）
│   ├── hooks.yaml                          # 声明式配置：hooks 段 + strategies 段
│   ├── structured_log.py                   # 结构化 JSON 日志（stderr）
│   ├── langfuse_trace.py                   # Langfuse 追踪（trace + generation + span）
│   ├── retry_tracker.py                    # 31课：重试追踪（纯观测）
│   ├── cost_guard.py                       # 31课：成本围栏（超预算 deny）
│   └── loop_detector.py                    # 31课：循环检测（重复状态 deny）
├── workspace/                              # 演示 Workspace（复用25课四件套）
│   └── demo_agent/
│       ├── soul.md / user.md / agent.md / memory.md
│       ├── hooks/
│       │   ├── hooks.yaml                  # Workspace 层配置
│       │   └── task_audit.py               # 任务审计日志
│       ├── skills/
│       │   ├── load_skills.yaml            # Skill 注册
│       │   └── sop_design/SKILL.md         # 技术设计文档 SOP
│       └── output/                         # 产出物目录
├── demo.py                                 # 端到端演示（Bootstrap + SkillLoader + 沙盒）
├── sandbox-docker-compose.yaml             # 沙盒 Docker Compose
├── .env.example                            # 环境变量模板
└── tests/                                  # 51 个单元测试 + 5 个集成测试
    ├── test_registry.py                    # HookRegistry 基础测试
    ├── test_dispatch_gate.py               # dispatch_gate 拦截测试
    ├── test_loader.py                      # HookLoader + strategies 加载测试
    ├── test_adapter.py                     # CrewAI 适配层测试
    ├── test_handlers.py                    # 日志 + 审计 handler 测试
    ├── test_cost_guard.py                  # 成本围栏单元测试
    ├── test_loop_detector.py               # 循环检测单元测试
    ├── test_retry_tracker.py               # 重试追踪单元测试
    ├── test_install_hooks.py               # 策略 YAML 加载集成测试
    ├── test_e2e_hooks.py                   # 全链路 Hook 集成测试（需 LLM）
    └── test_e2e_reliability.py             # 可靠性策略 E2E 测试（需 LLM）
```

---

## 30课 vs 31课：增量关系

| 维度 | 30课（可观测性） | 31课（可靠性） |
|------|----------------|---------------|
| HookRegistry | `dispatch()`：fire-and-forget | 新增 `dispatch_gate()`：传播 GuardrailDeny |
| HookLoader | hooks 段：无状态函数 | 新增 strategies 段：有状态类实例化 |
| CrewAdapter | 事件映射 + token 估算 | 新增 `_pending_deny` 延迟抛出 |
| shared_hooks | structured_log + langfuse_trace | 新增 retry_tracker + cost_guard + loop_detector |
| hooks.yaml | hooks 段 7 种事件 | 新增 strategies 段（顺序敏感） |

---

## 核心设计

### dispatch vs dispatch_gate

```
dispatch()      →  异常吞掉，继续执行下一个 handler（"报警器"）
dispatch_gate() →  GuardrailDeny 向上传播，其他异常吞掉（"断路器"）
```

### pending_deny 模式

CrewAI 的 `@before_tool_call` / `@after_tool_call` 装饰器内部有 `except Exception: pass`，会吞掉所有异常。解决方案：

```
@before_tool_call → GuardrailDeny → 存入 _pending_deny + return False（阻止工具）
                                      ↓
step_callback    → 检查 _pending_deny → 重新 raise（step_callback 不在 try/except 保护内）
```

### 三个可靠性策略

| 策略 | 类 | 事件 | 行为 |
|------|-----|------|------|
| 重试追踪 | RetryTracker | AFTER_TOOL_CALL | 纯观测：记录连续失败次数和成功率 |
| 成本围栏 | CostGuard | AFTER_TURN + BEFORE_TOOL_CALL | 实时累计 token 成本，超预算 deny |
| 循环检测 | LoopDetector | AFTER_TOOL_CALL + AFTER_TURN | 状态哈希去重，连续重复 deny |

### hooks.yaml strategies 段

```yaml
# 顺序敏感：列表顺序 = 执行顺序
# cost_guard 必须在 loop_detector 之前——已发生的成本必须先记录
strategies:
  - class: retry_tracker.RetryTracker
    config:
      max_retries: 3
    hooks:
      AFTER_TOOL_CALL: after_tool_handler

  - class: cost_guard.CostGuard
    config:
      budget_usd: 1.0
    hooks:
      AFTER_TURN: after_turn_handler
      BEFORE_TOOL_CALL: before_tool_handler

  - class: loop_detector.LoopDetector
    config:
      threshold: 3
    hooks:
      AFTER_TOOL_CALL: after_tool_handler
      AFTER_TURN: after_turn_handler
```

---

## 快速开始

```bash
cd /path/to/crewai_mas_demo/m5l31

# 1. 配置
cp .env.example .env
# 编辑 .env 填入密钥

# 2. 单元测试（无需 LLM/Langfuse，51 个用例）
python3 -m pytest tests/ -v --ignore=tests/test_e2e_reliability.py --ignore=tests/test_e2e_hooks.py

# 3. 端到端演示（需 LLM + Langfuse + 沙盒）
python3 demo.py
python3 demo.py "为一个短链接服务产出技术设计文档"

# 4. 低预算模式（触发成本围栏）
COST_GUARD_BUDGET=0.001 python3 demo.py
```

---

## 运行效果

```
Session: sess_20260424_120000
HookRegistry: 17 handlers loaded
   [global] structured_log.before_turn_handler -> before_turn
   [global] langfuse_trace.before_llm_handler -> before_llm
   [global] retry_tracker.RetryTracker.after_tool_handler -> after_tool_call
   [global] cost_guard.CostGuard.after_turn_handler -> after_turn
   [global] cost_guard.CostGuard.before_tool_handler -> before_tool_call
   [global] loop_detector.LoopDetector.after_tool_handler -> after_tool_call
   [global] loop_detector.LoopDetector.after_turn_handler -> after_turn
   ...
Strategies: ['retry_tracker', 'cost_guard', 'loop_detector']

Task: 为一个用户注册功能产出技术设计文档
...

============================================================
Guardrail Metrics:

  [retry_tracker]
    total_tool_calls: 3
    retry_success_rate: 1.0

  [cost_guard]
    model: qwen-plus
    estimated_cost_usd: 0.001234
    budget_usd: 1.0
    budget_utilization: 0.0

  [loop_detector]
    loop_detections: 0
    total_turns: 2
```

运行后查看结果：
1. **终端 stderr**：每个事件的结构化 JSON 日志 + 策略 cost/deny 记录
2. **Langfuse**（http://localhost:3000）：Trace 树含 TOOL span + GENERATION + task-complete
3. **workspace/demo_agent/audit.log**：任务审计 JSON 条目
4. **终端末尾**：三个策略的 Metrics 汇总

---

## 测试（51 单元 + 5 集成）

```bash
# 单元测试（无需 LLM/Langfuse）
python3 -m pytest tests/ -v --ignore=tests/test_e2e_reliability.py --ignore=tests/test_e2e_hooks.py

# 集成测试（需 LLM API + Langfuse）
python3 -m pytest tests/test_e2e_reliability.py -v -s
python3 -m pytest tests/test_e2e_hooks.py -v -s
```

| 文件 | 数量 | 测试内容 |
|------|------|---------|
| test_registry.py | 6 | 注册/分发/多handler/无handler/summary/异常隔离 |
| test_dispatch_gate.py | 5 | 正常分发/GuardrailDeny传播/非deny吞掉/首次deny停止/reason属性 |
| test_loader.py | 10 | yaml加载/两层合并/缺yaml/缺模块/策略实例化/多事件/共享实例/坏类名/坏方法名/属性拷贝 |
| test_adapter.py | 5 | BEFORE_TURN计数/step→AFTER_TURN/轮次重置/cleanup/tool映射 |
| test_handlers.py | 3 | 日志JSON schema/全事件覆盖/审计写文件 |
| test_cost_guard.py | 7 | token累加/超预算deny/预算内通过/metrics精度/边界值/AFTER_TURN检查/deny计数 |
| test_loop_detector.py | 7 | 不同状态/连续重复/threshold参数/非连续/metrics/工具路径/双路径独立 |
| test_retry_tracker.py | 5 | 连续失败/成功重置/成功率/独立工具/空工具名 |
| test_install_hooks.py | 3 | 策略全注册/cost先于loop顺序/预算deny端到端 |
| test_e2e_hooks.py | 2 | 全链路7事件×2层触发 / Langfuse trace验证 |
| test_e2e_reliability.py | 3 | 正常执行/循环检测/成本围栏 |

---

## 课堂代码演示学习指南

本节帮你按课程教学顺序阅读代码，建立完整的理解链路。

### 整体架构一览

```
┌─────────────────────────────────────────────────────────────────┐
│                         demo.py                                 │
│  HookRegistry + HookLoader → 加载 hooks.yaml（观测 + 策略）     │
│  CrewObservabilityAdapter → 桥接 CrewAI 原生 Hook                │
│  Agent + SkillLoader + Crew → 执行任务                           │
│  GuardrailDeny → 可靠性策略触发时中断                             │
└─────────────┬───────────────────────────────┬───────────────────┘
              │                               │
    ┌─────────▼──────────┐         ┌──────────▼──────────┐
    │   hook_framework/  │         │   shared_hooks/     │
    │                    │         │                     │
    │  registry.py       │         │  hooks.yaml         │
    │   EventType(7)     │         │   hooks 段 → 观测    │
    │   dispatch_gate()  │◄────────│   strategies 段 → 策略│
    │   GuardrailDeny    │         │                     │
    │                    │         │  structured_log.py   │
    │  loader.py         │         │  langfuse_trace.py   │
    │   strategies 段解析 │         │  retry_tracker.py    │
    │                    │         │  cost_guard.py       │
    │  crew_adapter.py   │         │  loop_detector.py    │
    │   pending_deny     │         └─────────────────────┘
    └────────────────────┘
```

### 学习路线（建议按顺序阅读）

---

#### 第一步：理解"断路器"——dispatch_gate 与 GuardrailDeny

**对应课文**：第五节"代码实战：从策略到 Hook 的工程落地"开头

**阅读文件**：`hook_framework/registry.py`

| 重点区域 | 行号 | 看什么 |
|---------|------|--------|
| `GuardrailDeny` 类 | 25-30 | 自定义异常，携带 `reason` 字段 |
| `dispatch()` 方法 | 61-70 | 30课已有：吞掉所有异常，继续执行——"报警器" |
| `dispatch_gate()` 方法 | 72-84 | 31课新增：`GuardrailDeny` 向上传播，其他异常吞掉——"断路器" |

**理解要点**：`dispatch` 和 `dispatch_gate` 只差一行——`except GuardrailDeny: raise`。这一行让 Hook 从"只能看"变成"能拦截"。

**验证**：运行 `python3 -m pytest tests/test_dispatch_gate.py -v`，看 5 个测试覆盖的场景。

---

#### 第二步：认识三个策略类

**对应课文**：第二、三、四节分别讲 RetryTracker、LoopDetector、CostGuard

**阅读顺序和重点**：

**2a. `shared_hooks/retry_tracker.py`**（最简单，纯观测）

| 重点区域 | 看什么 |
|---------|--------|
| `after_tool_handler()` | 失败 → 计数加1；成功 → 计数归零 + `_successful_retries` 加1 |
| `get_metrics()` | `retry_success_rate` = 成功重试 / 总重试 |

关键理解：RetryTracker **不做** retry，它只记录"重试发生了"。实际的重试是 CrewAI 自己的行为（工具失败后 LLM 会自动决定是否再调用）。

**2b. `shared_hooks/loop_detector.py`**（状态哈希去重）

| 重点区域 | 看什么 |
|---------|--------|
| `_compute_hash()` | 取 `tool_name:output前200字符` 算 SHA256 |
| `_check_loop()` | 取最近 N 个 hash，全部相同则 deny |
| 双路径 | `_tool_hashes` vs `_turn_hashes`：工具循环和推理循环独立检测 |

关键理解：和 `max_iter` 的区别——`max_iter=200` 要跑满 200 轮才停；哈希去重在第 3 次重复就能发现。

**2c. `shared_hooks/cost_guard.py`**（成本围栏，有 deny 能力）

| 重点区域 | 看什么 |
|---------|--------|
| `__init__()` | `COST_GUARD_BUDGET` 环境变量覆盖 |
| `after_turn_handler()` | 每轮结束累加 token → 算成本 → 超预算 raise GuardrailDeny |
| `before_tool_handler()` | 工具执行前再检一次预算 |
| `MODEL_PRICES` | 每百万 token 定价表 |

关键理解：双检查点——`AFTER_TURN` 记录已发生成本，`BEFORE_TOOL_CALL` 阻止新的开销。

**验证**：分别运行各策略的单元测试：
```bash
python3 -m pytest tests/test_retry_tracker.py tests/test_loop_detector.py tests/test_cost_guard.py -v
```

---

#### 第三步：策略如何声明式加载——hooks.yaml + HookLoader

**对应课文**：第五节中关于"注册顺序"的讨论

**阅读文件**：

**3a. `shared_hooks/hooks.yaml`**——声明式配置

```yaml
# 上半部分：30课的观测 hooks（无状态函数）
hooks:
  BEFORE_TURN:
    - handler: structured_log.before_turn_handler
  ...

# 下半部分：31课的可靠性策略（有状态类）
strategies:
  - class: cost_guard.CostGuard     # ← 类名
    config:                          # ← 构造函数参数
      budget_usd: 1.0
    hooks:                           # ← 实例方法 → 事件映射
      AFTER_TURN: after_turn_handler
      BEFORE_TOOL_CALL: before_tool_handler
```

关键理解：`hooks` 段注册无状态函数（每次调用独立）；`strategies` 段实例化有状态对象（同一个实例跨多个事件共享状态，比如 CostGuard 需要在 AFTER_TURN 累加、BEFORE_TOOL_CALL 检查，两个 handler 访问同一个 `_estimated_cost`）。

**3b. `hook_framework/loader.py`**——加载引擎

| 重点区域 | 行号 | 看什么 |
|---------|------|--------|
| `_load_module()` | 20-34 | importlib 动态导入 + 路径穿越防护 |
| hooks 段处理 | 43-61 | 30课已有：解析 `module.function` → 注册 |
| strategies 段处理 | 65-111 | 31课新增：解析 `module.Class` → 实例化 → 方法注册 |
| `load_two_layers()` | 113-117 | 全局 → Workspace 两层合并 |
| `strategies` 属性 | 119-121 | 返回副本，外部可读不可改 |

关键理解：strategies 列表的**顺序就是执行顺序**。cost_guard 排在 loop_detector 前面，确保成本先记录再判断循环。

**验证**：
```bash
python3 -m pytest tests/test_loader.py -v  # 10 个测试，含 7 个策略加载测试
python3 -m pytest tests/test_install_hooks.py -v  # 3 个策略注册集成测试
```

---

#### 第四步：pending_deny——CrewAI 的异常吞噬问题

**对应课文**：第五节"pending_deny"段落

**阅读文件**：`hook_framework/crew_adapter.py`

| 重点区域 | 行号 | 看什么 |
|---------|------|--------|
| `_pending_deny` 字段 | 54 | 存储被延迟的 GuardrailDeny |
| `_before_tool()` | 114-132 | 捕获 GuardrailDeny → 存入 `_pending_deny` → return False |
| `_after_tool()` | 134-157 | 同理，捕获并暂存 |
| `callback()` (step) | 163-207 | 执行完正常逻辑后，检查 `_pending_deny`，有则 raise |

关键理解：CrewAI 的装饰器 Hook 内部有 `except Exception: pass`，直接 raise 会被吞掉。所以：

1. 在 `@before_tool_call` 中 catch 并存储，return False 阻止工具
2. 在 `step_callback` 中取出并重新 raise（step_callback 不在 CrewAI 的 try/except 内）

这是一个**框架限制的工程绕路**，课文中有详细解释。

---

#### 第五步：端到端串联——demo.py

**对应课文**：整课的代码演示部分

**阅读文件**：`demo.py`

| 阶段 | 行号 | 做了什么 |
|------|------|---------|
| Hook 框架初始化 | 73-89 | `HookRegistry()` → `HookLoader.load_two_layers()` → 打印 handler 和 strategies |
| CrewAI 适配 | 92-95 | `CrewObservabilityAdapter` 安装全局 hooks + atexit 注册 cleanup |
| Bootstrap + Skill | 97-104 | `build_bootstrap_prompt()`（25课）+ `SkillLoaderTool`（25课） |
| 构建 Crew | 113-138 | Agent + Task + Crew，step_callback 和 task_callback 由 adapter 生成 |
| 执行 + 拦截 | 145-152 | `crew.kickoff()`，try/except `GuardrailDeny` |
| 度量输出 | 154-173 | cleanup + 打印三策略 metrics |

关键理解：demo.py 和 30 课是同一个 Agent（Bootstrap + SkillLoader + 沙盒执行），唯一的区别是 `hooks.yaml` 里多了 `strategies` 段，以及 `try/except GuardrailDeny`。这就是"在不改 Agent 代码的前提下，通过 Hook 声明式增加可靠性保障"。

---

#### 第六步：验证——跑测试看效果

```bash
# 1. 全部单元测试（51 个，无需外部依赖）
python3 -m pytest tests/ -v \
  --ignore=tests/test_e2e_reliability.py \
  --ignore=tests/test_e2e_hooks.py

# 2. 可靠性策略 E2E（需 LLM API + Langfuse）
python3 -m pytest tests/test_e2e_reliability.py -v -s

# 3. 低预算演示（触发成本围栏）
COST_GUARD_BUDGET=0.001 python3 demo.py

# 4. 查看 Langfuse Dashboard
open http://localhost:3000
# 检查：Trace 树中 TOOL span + GENERATION + task-complete 是否完整
```

---

### 学习检查清单

完成以上六步后，你应该能回答：

- [ ] `dispatch()` 和 `dispatch_gate()` 的区别是什么？（一行代码的差异）
- [ ] 为什么 `pending_deny` 要在 `step_callback` 里 raise，而不是直接在 `@before_tool_call` 里 raise？
- [ ] hooks.yaml 里 `hooks` 段和 `strategies` 段的区别是什么？（无状态 vs 有状态）
- [ ] 为什么 cost_guard 必须排在 loop_detector 前面？
- [ ] LoopDetector 的哈希去重和 `max_iter` 有什么区别？
- [ ] 如果要新增一个"超时围栏"策略，需要改哪些文件？（答案：只需写一个 .py + 在 hooks.yaml strategies 段声明）
