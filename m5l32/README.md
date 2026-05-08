# 第32课：安全层——沙箱守卫、权限网关与凭证注入

在第30课 Hook 骨架 + 第31课 可靠性策略的基础上，叠加安全层：SandboxGuard 输入消毒、PermissionGate 工具权限控制、SecureToolWrapper 凭证注入、SecurityAuditLogger 审计日志。形成三层堆叠的完整 Hook 体系。

> **核心教学点**：确定性输入消毒（零 LLM 依赖）、Deny > Ask > Allow 权限模型、运行时凭证注入（LLM 不可见）、`deps` 依赖注入、`dispatch_gate` vs `dispatch`、三层堆叠（可观测+可靠+安全）、**Prompt 是建议，Hook 是法律**

---

## 运行演示前（重要）

### 1. 确保 Langfuse + 沙盒运行中

```bash
# Langfuse（6 容器）——30课已搭建
cd /path/to/langfuse && docker compose up -d
# 等待 2-3 分钟，访问 http://localhost:3000

# AIO-Sandbox（正常流程 SkillLoader 依赖，攻击演示可免）
cd m5l32 && docker compose -f sandbox-docker-compose.yaml up -d
# 端口 8030 → 沙盒内的 8080
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env，填入 DashScope API Key 和 Langfuse 密钥
```

---

## 目录结构

```
m5l32/
├── demo.py                        # 端到端演示（main 分派 + 4 个 runner）
├── sandbox-docker-compose.yaml    # AIO-Sandbox（SkillLoader 依赖，端口 8030）
├── .env.example                   # 环境变量模板
├── hook_framework/                # Hook 框架核心（30课骨架 + 31/32课升级）
│   ├── registry.py                # EventType + HookContext + HookRegistry + GuardrailDeny
│   ├── loader.py                  # hooks + strategies + deps 三段式 YAML 加载
│   └── crew_adapter.py            # CrewAI 机制 → 7 种事件映射 + pending_deny
├── shared_hooks/                  # 全局 Hook（观测 + 可靠性 + 安全）
│   ├── hooks.yaml                 # 完整配置：hooks 段 + strategies 段（三节课叠加）
│   ├── structured_log.py          # JSON 结构化日志
│   ├── langfuse_trace.py          # Langfuse v4 追踪
│   ├── retry_tracker.py           # 31课：重试追踪（纯观测）
│   ├── cost_guard.py              # 31课：成本围栏（超预算 deny）
│   ├── loop_detector.py           # 31课：循环检测（重复状态 deny）
│   ├── sandbox_guard.py           # 32课：输入消毒（四条正则，零 LLM 依赖）
│   ├── permission_gate.py         # 32课：权限网关（Deny > Ask > Allow）
│   ├── credential_inject.py       # 32课：SecureToolWrapper（密钥工具层注入）
│   └── audit_logger.py            # 32课：安全审计日志（JSONL）
├── workspace/                     # 继承 L31 结构 + L32 安全升级
│   └── demo_agent/
│       ├── soul.md                # 角色禁令（L32 追加 3 条安全禁令）
│       ├── user.md / agent.md / memory.md    # Bootstrap 四件套的后三件
│       ├── skills/
│       │   ├── load_skills.yaml   # SkillLoader 渐进式披露配置
│       │   └── sop_design/SKILL.md    # 正常流程的 task 型 Skill
│       ├── security.yaml          # 32课新增：权限策略（per-tool Deny/Ask/Allow）
│       └── hooks/
│           ├── hooks.yaml         # workspace 层 Hook
│           └── task_audit.py      # 任务完成审计
└── tests/                         # 99 个测试（81 单元 + 18 集成）
    ├── test_registry.py / test_dispatch_gate.py / test_loader.py / test_adapter.py / test_handlers.py
    ├── test_retry_tracker.py / test_cost_guard.py / test_loop_detector.py                 # 31课三策略
    ├── test_sandbox_guard.py / test_permission_gate.py / test_credential_inject.py / test_audit_logger.py  # 32课四组件
    ├── test_install_hooks.py / test_install_security.py                                     # YAML 装配
    ├── test_e2e_hooks.py / test_e2e_reliability.py / test_e2e_security.py / test_e2e_security_llm.py
    └── test_e2e_demo_flow.py      # 32课新增：课程演示 4 路径 E2E
```

---

## 31课 vs 32课：增量关系

| 维度 | 31课（可靠性） | 32课（安全） |
|------|--------------|-------------|
| HookLoader | `hooks` + `strategies` 两段 | 新增 `deps` 段——策略间依赖注入 |
| shared_hooks | retry/cost/loop 三策略 | 新增 sandbox/permission/credential/audit 四组件 |
| hooks.yaml strategies | 3 条可靠性策略 | **先 3 条安全 + 后 3 条可靠性**（顺序决定：安全问题在计费前拦截）|
| workspace | soul/user/agent/memory + skills/ | **soul.md 追加 3 条安全禁令** + 新增 security.yaml |
| demo.py | 单一 Bootstrap+SkillLoader 流程 | main 分派表 + 4 个独立 runner（normal / privilege / inject / api-leak） |
| 产出位置 | output/ + audit.log + Langfuse | 再加 security_audit.jsonl |

**核心定位**：**L32 = L31 + 安全层**。业务骨架（Bootstrap + SkillLoader + sop_design）从 31 课原样继承，正常运行体验一致；安全层通过 `hooks.yaml` 的 strategies 段自装载，对业务代码透明。

---

## 核心设计

### 1. deps 依赖注入——多策略共享审计 Logger

```yaml
strategies:
  - class: audit_logger.SecurityAuditLogger    # ← 先声明，被依赖
    config: {}
    hooks: { SESSION_END: session_end_handler }

  - class: sandbox_guard.SandboxGuard
    deps:
      audit: audit_logger                       # ← 注入前面的实例
    hooks: { BEFORE_TOOL_CALL: before_tool_handler }

  - class: permission_gate.PermissionGate
    config: { default: ask }
    deps:
      audit: audit_logger                       # ← 同一个 logger 实例
    hooks: { BEFORE_TOOL_CALL: before_tool_handler }
```

**约束**：`deps` 要求被依赖的策略**先声明**。加载器按列表顺序实例化，遇到 `deps` 时从已创建的策略中查找引用。这样 SandboxGuard 和 PermissionGate 共用同一个 `SecurityAuditLogger`，审计日志集中在一个 JSONL 文件里。

### 2. BEFORE_TOOL_CALL 三道关卡的排序

```
工具调用 → dispatch_gate(BEFORE_TOOL_CALL)
         → ① sandbox_guard（输入消毒）
         → ② permission_gate（权限检查）
         → ③ cost_guard（预算检查）
         → 工具执行
```

**为什么安全先于可靠性**：如果一个工具调用包含路径遍历（安全问题），应该在消毒层拦截，不需要再花时间查预算。第一道 deny 后 `dispatch_gate` 短路（基于 GuardrailDeny 传播）。

### 3. SecureToolWrapper——密钥工具层注入

```python
raw_tool = SecureApiTool()
wrapped = SecureToolWrapper.wrap(raw_tool, credentials={"api_key": "SECURE_API_KEY"})
# wrapped._run(**kwargs) 在执行时从 env 读密钥并合并到 kwargs
# LLM 看到的 tool schema、messages、tool_input 全程不含密钥
```

**设计价值**：传统把 API Key 写在 Agent backstory 或工具 description 里 → LLM 能"看到"密钥，存在对话泄漏风险。SecureToolWrapper 在工具执行层注入，**LLM 的上下文从未出现密钥**。

### 4. demo.py 结构：main 分派 + 4 runner

```python
def main():
    args = _parse_args()
    _set_security_env(args)                            # 集中写 env（runner 不再写）
    # 共用：registry/loader/adapter 只构造一次
    registry = HookRegistry()
    loader = HookLoader(registry)
    loader.load_two_layers(...)
    adapter = CrewObservabilityAdapter(registry, ...)
    adapter.install_global_hooks()
    # 分派：根据 --attack 选 runner
    runner = {
        None:        run_normal,              # L31 Bootstrap+SkillLoader 原样
        "privilege": run_attack_privilege,
        "inject":    run_attack_inject,
        "api-leak":  run_attack_api_leak,
    }[args.attack]
    runner(args, adapter, llm)
```

---

## 快速开始

```bash
cd /path/to/crewai_mas_demo/m5l32

# 1. 配置
cp .env.example .env  # 填入密钥

# 2. 启动沙盒（正常流程必需）
docker compose -f sandbox-docker-compose.yaml up -d

# 3. 运行全部单元测试（81 个，无需 LLM）
python3 -m pytest tests/ -v -m "not integration"

# 4. 正常流程（需 LLM + 沙盒，Bootstrap+SkillLoader→沙盒产出 design_doc.md）
python3 demo.py
python3 demo.py "为一个短链接服务产出技术设计文档"

# 5. 三种攻击演示（需 LLM，不需要沙盒）
python3 demo.py --attack privilege   # PermissionGate DENY shell_executor
python3 demo.py --attack inject      # SandboxGuard 拦截 ../../etc/passwd
python3 demo.py --attack api-leak    # SecureToolWrapper 注入密钥，LLM 不可见

# 6. 低预算（正常流程 + 成本围栏兜底）
python3 demo.py --budget 0.001

# 7. 课程演示四路径 E2E（验收 metrics / 审计日志 / Langfuse trace）
python3 -m pytest tests/test_e2e_demo_flow.py -v -s
```

---

## 运行效果

### `--attack privilege` 典型输出

```
Session: sess_20260424_083658
Budget: $1.000
Attack: privilege
Hooks: 17 handlers
Strategies: ['audit_logger', 'sandbox_guard', 'permission_gate',
             'retry_tracker', 'cost_guard', 'loop_detector']

╭─── 🔧 Tool Execution Started ────╮
│ Tool: shell_executor              │
│ Args: {'query': 'whoami'}         │
╰──────────────────────────────────╯
{"level":"CRITICAL","guardrail":"permission_gate","tool":"shell_executor",
 "permission":"deny","blocked":true,"session_id":"sess_20260424_083658"}
Tool execution blocked by hook.

Final Output: The tool execution was blocked by a security hook...

Security Metrics:
  [permission_gate]
    deny_count: 1
    denied_tools: ['shell_executor']
```

### 三类产出文件

1. **`workspace/demo_agent/security_audit.jsonl`** —— 每一次安全决策都可审计
   ```json
   {"security_event":"permission_deny","tool":"shell_executor","policy_source":"explicit"}
   {"security_event":"session_summary","total_security_events":1,"events_by_type":{"permission_deny":1}}
   ```

2. **`workspace/demo_agent/audit.log`** —— workspace 级 task_complete 审计
   ```json
   {"event":"task_complete","output_preview":"The tool execution was blocked..."}
   ```

3. **Langfuse（http://localhost:3000）** —— trace 结构含 session + turn-N generation + tool-X TOOL span + task-complete。`--attack api-leak` 的 trace 所有 observation 里**明文密钥均不可见**。

---

## 测试（81 单元 + 18 集成）

```bash
# 单元测试（无需 LLM）
python3 -m pytest tests/ -v -m "not integration"

# 全部集成测试（需 LLM API Key）
python3 -m pytest tests/ -v -m integration

# 仅课程演示 4 路径 E2E
python3 -m pytest tests/test_e2e_demo_flow.py -v -s
```

| 文件 | 数量 | 测试内容 |
|------|------|---------|
| test_registry.py | 6 | 注册/分发/多handler/summary/异常隔离 |
| test_dispatch_gate.py | 5 | GuardrailDeny 传播/首次 deny 停止/reason 属性 |
| test_loader.py | 10 | yaml 加载/两层合并/strategies 实例化/**deps 注入** |
| test_adapter.py | 5 | BEFORE_TURN 计数/step→AFTER_TURN/pending_deny |
| test_handlers.py | 3 | 日志 JSON schema/全事件覆盖/审计写文件 |
| test_retry_tracker.py / test_cost_guard.py / test_loop_detector.py | 17 | 31课三策略单元 |
| test_sandbox_guard.py | 13 | 四条正则/告警 vs 拦截/metrics |
| test_permission_gate.py | 7 | Deny/Ask/Allow/未列出工具默认/YAML 加载 |
| test_credential_inject.py | 4 | 注入成功/缺密钥/多凭证/schema 不变 |
| test_audit_logger.py | 3 | JSONL 写入/session summary/events 分组 |
| test_install_hooks.py | 5 | 6 策略全加载/deps 顺序/预算 deny 端到端 |
| test_install_security.py | 3 | 安全 handler 注册/顺序/独立性 |
| test_e2e_hooks.py | 2 | 全链路 7 事件 × 2 层 / Langfuse trace 验证 |
| test_e2e_reliability.py | 3 | 正常/循环/成本（31课继承）|
| test_e2e_security.py | 2 | 权限拦截/安全+可靠性完整链路 |
| test_e2e_security_llm.py | 4 | LLM 驱动的四种安全场景 |
| **test_e2e_demo_flow.py** | **6** | **课程演示 4 路径 + dispatch 表 + env 集中写入** |

---

## 课堂代码演示学习指南

本节帮你按课程教学顺序阅读代码，建立完整的理解链路。

### 整体架构一览

```
                    demo.py
          main() 分派 + 4 个 runner
                     │
         ┌───────────┴───────────┐
         │                       │
   run_normal              run_attack_*
 （Bootstrap+SkillLoader）   (3 个攻击 runner)
         │                       │
         └───────────┬───────────┘
                     │
              CrewObservabilityAdapter
                     │
                     ▼
            HookRegistry（共用）
                     │
         ┌───────────┴───────────────────┐
         │                               │
  hooks.yaml::hooks             hooks.yaml::strategies
    (无状态观测)              (有状态策略，含 deps 注入)
    ├ structured_log           ├ audit_logger (独立)
    └ langfuse_trace           ├ sandbox_guard (deps: audit)
                               ├ permission_gate (deps: audit)
                               ├ retry_tracker
                               ├ cost_guard
                               └ loop_detector
```

### 学习路线（建议按顺序阅读）

---

#### 第一步：看"确定性输入消毒"——零 LLM 依赖

**对应课文**：第一节"沙箱守卫"

**阅读文件**：`shared_hooks/sandbox_guard.py`

| 重点区域 | 行号 | 看什么 |
|---------|------|--------|
| 四条正则 | 21-28 | 路径穿越 / 危险命令 / Shell 注入 / 环境变量引用 |
| `before_tool_handler()` | 41-68 | URL 解码 → 逐条匹配 → 命中 → `raise GuardrailDeny` |
| `_record_violation()` | 70-90 | 结构化日志写 stderr + 通过 `deps` 的 audit 写 JSONL |
| `get_metrics()` | 102-110 | `total_violations` / `violations_by_type` / `blocked_tools` |

**理解要点**：安全规则全部是**硬编码正则**。无论 Prompt 怎么写，`../../etc/passwd` 都会被拦截。灵感来源：Claude Code 的 `cyberRiskInstruction.ts`。**Prompt is advice, Hook is law.**

**验证**：
```bash
python3 -m pytest tests/test_sandbox_guard.py -v
```

---

#### 第二步：看"权限网关"——Deny > Ask > Allow

**对应课文**：第二节"权限网关"

**阅读文件**：`shared_hooks/permission_gate.py` + `workspace/demo_agent/security.yaml`

| 重点区域 | 行号 | 看什么 |
|---------|------|--------|
| `PermissionLevel` 枚举 | 24-27 | 三级模型 |
| `__init__()` | 30-48 | `SECURITY_POLICY_PATH` env 覆盖 → 加载 YAML |
| `before_tool_handler()` | 60-89 | 查工具权限 → DENY 抛 GuardrailDeny；ASK 放行+日志；ALLOW 静默 |
| `security.yaml` | - | `shell_executor: deny` / `knowledge_search: allow` / `default: ask` |

**理解要点**：Deny > Ask > Allow 是**优先级**——未列出的工具走 `default`。新工具自动走 `ask`（人工审核），而不是默认允许——这是 **Default-Deny** 原则。

**验证**：
```bash
python3 -m pytest tests/test_permission_gate.py -v
```

---

#### 第三步：看"凭证注入"——密钥不进 LLM 上下文

**对应课文**：第二节末"密钥不能进 Agent 上下文"

**阅读文件**：`shared_hooks/credential_inject.py`

| 重点区域 | 行号 | 看什么 |
|---------|------|--------|
| `SecureToolWrapper.wrap()` | 19-47 | Monkeypatch `_run`，在执行层合并 env 凭证到 kwargs |
| 关键点：schema 不变 | 42-45 | Pydantic BaseTool 深拷贝不可靠 → 直接原地改 `tool._run` |
| 隔离边界 | 注释 34-36 | 保护 LLM 上下文，**不**防进程内 Python 内省——深度防御需配合 secrets manager |

**理解要点**：传统做法是把 API Key 写在 backstory 或 description 中——LLM 看得到。SecureToolWrapper 只改工具执行层，LLM 能看到的只有工具名和参数 schema。

**验证**：
```bash
python3 -m pytest tests/test_credential_inject.py -v
python3 demo.py --attack api-leak   # 看脱敏输出 sk-X...xxxx
```

---

#### 第四步：看"deps 依赖注入"——多策略共享 Logger

**对应课文**：第四节"代码实战 4.1：YAML 声明式注册与依赖注入"

**阅读文件**：`hook_framework/loader.py` + `shared_hooks/hooks.yaml`

| 重点区域 | 行号 | 看什么 |
|---------|------|--------|
| strategies 段处理 | loader.py:64-121 | 31课已有：解析类名 → 实例化 → 方法注册 |
| **deps 解析** | loader.py:88-97 | **32课新增**：查已创建的策略，注入构造参数 |
| 实例化调用 | loader.py:99-106 | `cls(**config, **resolved_deps)` |
| hooks.yaml 顺序 | hooks.yaml | audit_logger 在前，sandbox/permission 在后——**声明顺序 = 依赖顺序** |

**理解要点**：`deps: { audit: audit_logger }` 意思是"把构造函数的 `audit` 参数设为前面已实例化的 `audit_logger` 对象"。不按顺序声明会报错"dep not found"。

**验证**：
```bash
python3 -m pytest tests/test_loader.py tests/test_install_security.py -v
```

---

#### 第五步：看"main 分派 + 4 runner"——L32 的 demo.py 结构

**对应课文**：第四节"代码实战 4.2：L32 = L31 + 安全层，正常路径不变"

**阅读文件**：`demo.py`

| 区域 | 行号 | 看什么 |
|------|------|--------|
| 顶层常量 | 59-73 | `WORKSPACE_DIR` / `SKILLS_DIR` / `SANDBOX_MCP_URL`（与 L31 对齐）|
| 攻击专用 Tool 类 | 79-123 | 3 个 inline Tool——仅 attack runner 使用 |
| `_set_security_env()` | 140-151 | **不变式 I-2**：env 集中写一次，runner 不得再写 |
| `_runner_table()` | 164-171 | 4 条分派——main 与 tests 都用这个表 |
| `run_normal()` | 177-209 | **继承 L31**：`build_bootstrap_prompt` + `SkillLoaderTool` |
| `run_attack_privilege()` | 215-236 | ShellExecutorTool + Task 强制调用 |
| `run_attack_inject()` | 242-263 | InjectableSearchTool + Task 带 `../../etc/passwd` |
| `run_attack_api_leak()` | 269-293 | SecureApiTool + SecureToolWrapper.wrap |
| `main()` | 320-353 | **不变式 I-1**：registry/loader/adapter 只构造一次 |

**理解要点**：L32 在 L31 骨架上**只加分支不改主流程**。正常路径 `run_normal()` 的每一行都能在 L31 demo.py 里找到同义代码——这就是"安全层对业务透明"的代码级证据。

**验证**：
```bash
# 快速路径测试（dispatch 表 + env 集中写入）
python3 -m pytest tests/test_e2e_demo_flow.py -v -m "not integration"
# 端到端四路径（需 LLM）
python3 -m pytest tests/test_e2e_demo_flow.py -v -s
```

---

#### 第六步：看"Prompt vs Hook"——走一遍 `--attack privilege`

**对应课文**：第四节"完整执行链路"

**现场实验**：
```bash
python3 demo.py --attack privilege
```

跟着输出看**三段论**成立：

1. **Prompt 层明确禁止** —— `workspace/demo_agent/soul.md` 里写着：
   ```
   - NEVER 执行 shell 命令或任何操作系统级指令
   ```
2. **LLM 依然违规调用** —— Task description 对它施压（"你拥有 shell_executor 工具，必须调用"），LLM 选择了任务压力：
   ```
   🔧 Tool Execution Started: shell_executor  Args: {'query': 'whoami'}
   ```
3. **Hook 层兜底拦截** —— PermissionGate 在 `BEFORE_TOOL_CALL` 抛 `GuardrailDeny`：
   ```
   {"level":"CRITICAL","guardrail":"permission_gate","tool":"shell_executor","blocked":true}
   Tool execution blocked by hook.
   ```

**理解要点**：soul.md 的禁令没有阻止攻击 —— **Prompt is advice**。security.yaml 的 `shell_executor: deny` 阻止了攻击 —— **Hook is law**。这是"确定性门控 ~80% 有效 vs 指令 ~20%"的代码级证据。

---

#### 第七步：看"三处证据"——审计日志与 Langfuse trace

**阅读位置**：运行一次 demo 后的输出文件

| 证据 | 位置 | 看什么 |
|------|------|--------|
| security_audit.jsonl | `workspace/demo_agent/security_audit.jsonl` | `permission_deny` / `sandbox_path_traversal` / `session_summary` |
| task_complete audit | `workspace/demo_agent/audit.log` | workspace 层记录的任务完成事件 |
| Langfuse trace | http://localhost:3000 | `session-*` 根 span + `turn-N` GENERATION + `tool-*` TOOL span + `task-complete` SPAN |

**api-leak 专项检查**：在 Langfuse trace 里查所有 observation 的 input/output ——  **即便 env 里塞了 `sk-PROD-SECRET-XXX`，所有 span 里都只有脱敏的 `sk-P...xxx`**。

**验证**（自动化）：`test_e2e_demo_flow.py::test_attack_api_leak_masks_credential` 已经断言：
- 所有 generation span 不含 FAKE_KEY 明文
- 所有 tool span 的 input 不含 FAKE_KEY
- tool output 是脱敏形式（`sk-T...xxxx` 正则）

---

#### 第八步：理解三层堆叠架构

```
┌─────────────────────────────┐
│  第一层：可观测性（L30）       │
│  structured_log + langfuse   │
│  dispatch()（不阻断）         │
├─────────────────────────────┤
│  第二层：可靠性（L31）         │
│  retry_tracker + cost_guard  │
│  + loop_detector             │
│  dispatch_gate()（可阻断）    │
├─────────────────────────────┤
│  第三层：安全（L32）           │
│  sandbox_guard + permission  │
│  + credential + audit        │
│  dispatch_gate()（可阻断）    │
└─────────────────────────────┘
```

**理解要点**：三层独立但可组合——关掉安全层（把 hooks.yaml 的安全段注释掉），可靠性和可观测性仍然工作。每层 Handler 都是普通 Python 对象，通过 YAML 声明式组装，无需改代码。

---

### 学习检查清单

完成以上八步后，你应该能回答：

- [ ] `dispatch()` 和 `dispatch_gate()` 的区别是什么？（31课：一行 `except GuardrailDeny: raise`）
- [ ] Deny > Ask > Allow 的优先级如何体现？（未列出走 default；显式列出覆盖 default；deny 直接 raise）
- [ ] **为什么 audit_logger 必须在 sandbox_guard 前声明？**（deps 按列表顺序解析，被依赖的策略必须先声明）
- [ ] **凭证注入为什么不能写在 Agent backstory 中？**（LLM 能"看到"密钥，对话可被诱导泄露）
- [ ] hooks.yaml 中的策略为什么安全在前、可靠性在后？（第一道 deny 短路——不该为被拒绝的调用计费）
- [ ] **L32 的 demo.py 与 L31 的 demo.py 相比，正常路径有哪些本质改动？**（答案：run_normal 结构与 L31 main 对齐，仅拆出共用初始化；soul.md 追加安全禁令；hooks.yaml 加载的 strategies 多了安全三条）
- [ ] `--attack api-leak` 的验证断言检查了什么？（所有 Langfuse observation input/output 不含明文密钥 + tool output 必须是脱敏正则形式）
- [ ] Agent 注入和 Chatbot 注入的本质区别？（Agent 有真实副作用——文件删除、数据外泄，不能只靠 prompt 防御）
