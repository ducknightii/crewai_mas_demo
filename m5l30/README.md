# 第30课：可观测性——Hook 骨架 + Langfuse 全链路追踪

本课构建 Hook 框架骨架，将前序课程的零散 Hook 统一到 5+2 事件体系，并接入 Langfuse 实现全链路追踪。

> **核心教学点**：5+2 事件类型（对齐 Agent Turn 周期）、两层 Hook 配置（全局+Workspace）、HookRegistry 分发机制、CrewAI 适配层映射、Langfuse Docker 自托管追踪、全局 Hook 自动传播到 Sub-Crew

---

## 环境搭建（重要，请按顺序完成）

### Step 1：启动 Langfuse（Docker 自托管，6 容器）

本课使用 Docker 自托管的 Langfuse v3。如果已经有运行中的 Langfuse 实例，跳到 Step 2。

```bash
# 克隆 Langfuse 官方仓库
git clone https://github.com/langfuse/langfuse.git
cd langfuse

# 生成必需的密钥（写入 .env）
echo "ENCRYPTION_KEY=$(openssl rand -hex 32)" >> .env
echo "SALT=$(openssl rand -hex 32)" >> .env
echo "NEXTAUTH_SECRET=$(openssl rand -hex 32)" >> .env
echo "NEXTAUTH_URL=http://localhost:3000" >> .env
echo "POSTGRES_PASSWORD=$(openssl rand -hex 16)" >> .env
echo "REDIS_AUTH=$(openssl rand -hex 16)" >> .env

# 启动所有容器（Web + Worker + Postgres + Redis + ClickHouse + MinIO）
docker compose up -d

# ⚠️ ClickHouse 冷启动慢，请等待 5-8 分钟再检查健康状态
# 如果 langfuse-web-1 一直在重启，通常是 ClickHouse 还没就绪，继续等待即可
# 可以用以下命令查看各容器状态：
docker compose ps
docker compose logs langfuse-web-1 --tail=20      # 查看 web 容器报错
docker compose logs langfuse-clickhouse-1 --tail=10  # 确认 ClickHouse 已就绪

# 确认健康状态（ClickHouse 就绪后才会返回 OK）
curl http://localhost:3000/api/public/health
# 预期输出：{"status":"OK","version":"3.x.x"}
```

启动后访问 http://localhost:3000，首次登录需注册账号（自托管模式，注册即管理员）。

> **常见问题**：`langfuse-web-1` 反复重启 → 99% 是 ClickHouse 尚未就绪，等待即可，不需要重建容器。

### Step 2：创建 Project 和 API Key

登录 Langfuse Dashboard 后：

1. **创建 Organization**（如 `my-org`）
2. **创建 Project**（如 `course-demo`）
3. 进入 Project → **Settings → API Keys → Create API Key**
4. 记录 `Public Key`（`pk-lf-...`）和 `Secret Key`（`sk-lf-...`）

> 课程演示使用的测试 Key：`pk-lf-course-demo` / `sk-lf-course-demo`，仅限本机使用。

### Step 3：启动 AIO-Sandbox（演示用沙盒）

```bash
cd /path/to/crewai_mas_demo/m5l30
docker compose -f sandbox-docker-compose.yaml up -d

# 等待健康检查通过
docker inspect m5l30-aio-sandbox-1 --format '{{.State.Health.Status}}'
# 预期输出：healthy
```

沙盒端口 `8030`，挂载：
- `workspace/demo_agent/skills → /mnt/skills:ro`（Skill 资源）
- `workspace/demo_agent/output → /workspace/output:rw`（产出物）

### Step 4：配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，填入真实值：

```bash
# === LLM 配置（阿里云 DashScope） ===
OPENAI_API_KEY=your-dashscope-api-key          # 替换为你的 DashScope API Key
OPENAI_API_BASE=https://dashscope.aliyuncs.com/compatible-mode/v1
AGENT_MODEL=qwen-plus

# === Langfuse（Docker 自托管） ===
LANGFUSE_PUBLIC_KEY=pk-lf-course-demo           # 替换为你的 Public Key
LANGFUSE_SECRET_KEY=sk-lf-course-demo           # 替换为你的 Secret Key
LANGFUSE_BASE_URL=http://localhost:3000          # ⚠️ 必须设置，否则 SDK 会发到 cloud.langfuse.com

# === CrewAI 遥测（必须关闭）===
CREWAI_DISABLE_TELEMETRY=true                   # ⚠️ 不设置会在 Langfuse 产生两条相同 trace
```

**关键注意**：

1. `LANGFUSE_BASE_URL` 必须显式设置为 `http://localhost:3000`。如果不设置，Langfuse Python SDK 默认连接 `https://cloud.langfuse.com`，trace 数据不会出现在你的本地实例中。

2. `CREWAI_DISABLE_TELEMETRY=true` 必须设置。CrewAI 启动时会将自己的 OTel TracerProvider 注册为全局 provider，导致 Langfuse SDK 只能复用该 provider 而无法独立接管，最终使同一次 Demo 在 Langfuse 中产生两条内容相同的 trace 记录。

### Step 5：验证连接

```bash
# 验证 Langfuse API Key 有效
curl -s -u "pk-lf-course-demo:sk-lf-course-demo" \
     "http://localhost:3000/api/public/traces?limit=1"
# 预期输出：{"data":[],"meta":{"page":1,...}}（空列表表示连接正常）
```

---

## 目录结构

```
m5l30/
├── hook_framework/                     # Hook 框架核心
│   ├── __init__.py                     # 导出公共接口
│   ├── registry.py                     # F1-F2: EventType(5+2) + HookContext + HookRegistry
│   ├── loader.py                       # F3-F4: hooks.yaml 解析 + importlib 两层自动加载 + 模块缓存
│   └── crew_adapter.py                 # F5: CrewAI 4种机制 → HookRegistry 7种事件
├── shared_hooks/                       # 全局 Hook（所有 Agent 共享）
│   ├── hooks.yaml                      # 全局配置：事件 → handler 映射
│   ├── structured_log.py               # F6: 结构化 JSON 日志（stderr）
│   └── langfuse_trace.py              # F7: Langfuse 追踪（树状 trace + span 栈嵌套）
├── workspace/                          # 演示 Workspace（25 课 Bootstrap 架构）
│   └── demo_agent/
│       ├── soul.md                     # 💡 Agent 身份
│       ├── agent.md                    # 💡 Agent 职责
│       ├── user.md                     # 💡 用户画像
│       ├── memory.md                   # 💡 记忆索引
│       ├── hooks/
│       │   ├── hooks.yaml              # Workspace 配置
│       │   └── task_audit.py           # F8: 任务审计日志
│       ├── skills/
│       │   ├── load_skills.yaml        # 💡 Skill 清单（task 型）
│       │   └── sop_design/SKILL.md     # 💡 技术设计 SOP（写入文件到沙盒）
│       └── output/                     # 沙盒产出物（设计文档写入此处）
├── sandbox-docker-compose.yaml         # AIO-Sandbox 配置（端口 8030）
├── demo.py                             # F10: Bootstrap + SkillLoader + Hook 全链路演示
├── .env.example                        # 环境变量模板
├── tests/
│   ├── test_registry.py                # T1-T5 + T_extra1: HookRegistry 单元测试
│   ├── test_loader.py                  # T6-T8 + T_extra3: HookLoader 单元测试
│   ├── test_handlers.py                # T9-T11: handler 输出测试
│   ├── test_adapter.py                 # T12-T14 + T_extra2/4: 适配层测试
│   ├── test_e2e_hooks.py               # T15-T16: 全链路端到端测试
│   └── conftest.py                     # pytest fixtures
└── README.md
```

---

## 演示架构（复用 25 课框架）

```
┌─────────────────────────────────────────────────────────────────────┐
│  demo.py                                                            │
│                                                                     │
│  1. build_bootstrap_prompt(workspace/)                              │
│     → 加载 soul.md + agent.md + user.md + memory.md → backstory    │
│                                                                     │
│  2. SkillLoaderTool(skills_dir=workspace/skills/)                   │
│     → 渐进式披露：YAML frontmatter 构建 XML description            │
│     → task 型 Skill：启动 Sub-Crew 在 AIO-Sandbox 中执行           │
│                                                                     │
│  3. Hook 框架（两层加载）                                            │
│     → shared_hooks/：结构化日志 + Langfuse 追踪                     │
│     → workspace/hooks/：任务审计                                    │
│                                                                     │
│  4. 💡 全局 Hook 自动传播到 Sub-Crew                                │
│     → @before_tool_call / @after_tool_call 捕获沙盒工具调用         │
│     → Langfuse trace 包含 Sub-Crew 的 sandbox 操作                  │
└─────────────────────────────────────────────────────────────────────┘
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

# 1. 完成上方「环境搭建」五步（Langfuse + Sandbox + .env）

# 2. 运行单元测试（无需 LLM/Langfuse/Sandbox）
python3 -m pytest tests/ -v --ignore=tests/test_e2e_hooks.py

# 3. 运行端到端测试（需 LLM API + Langfuse）
python3 -m pytest tests/ -v

# 4. 运行演示（需 LLM API + Langfuse + Sandbox）
python3 demo.py
python3 demo.py "为一个短链接服务产出技术设计文档"

# 5. 查看结果
#    - 终端 stderr：结构化 JSON 日志
#    - Langfuse Dashboard（http://localhost:3000）→ 选择 course-demo 项目 → Traces
#    - workspace/demo_agent/output/：Sub-Crew 在沙盒中产出的设计文档
#    - workspace/demo_agent/audit.log：任务审计条目
```

---

## 运行效果

```
🔗 Session: sess_20260423_110456
📦 HookRegistry: 12 handlers loaded
   [global] structured_log.before_turn_handler → before_turn
   [global] structured_log.before_llm_handler → before_llm
   [global] langfuse_trace.before_llm_handler → before_llm
   [global] structured_log.before_tool_handler → before_tool_call
   [global] langfuse_trace.before_tool_handler → before_tool_call
   ...
   [workspace] task_audit.write_audit_entry → task_complete
   [global] langfuse_trace.flush_and_close → session_end

🚀 Task: 为一个用户注册功能产出技术设计文档

📋 Bootstrap: 613 chars from workspace/demo_agent/
🔧 Skills: ['sop_design']

... (Sub-Crew 在沙盒中执行 SOP，写入设计文档)

📊 Result: # 用户注册功能技术设计文档 ...
🔗 Langfuse: http://localhost:3000
📄 Design doc: workspace/demo_agent/output/user_registration_design.md
📝 Audit log: workspace/demo_agent/audit.log
```

### Langfuse Trace（树状结构）

```
session-{id}                              ← CHAIN（root span）
├── tool-skill_loader                     ← TOOL（主 Agent 调用 Skill）
│   ├── tool-sandbox_file_operations      ← TOOL（Sub-Crew 沙盒操作，嵌套在 skill_loader 下）
│   ├── tool-sandbox_str_replace_editor   ← TOOL
│   ├── tool-sandbox_execute_code         ← TOOL
│   └── tool-sandbox_execute_bash         ← TOOL
├── turn-1                                ← GENERATION（始终挂在 root 下）
└── task-complete                         ← SPAN（始终挂在 root 下）
```

> 💡 span 栈机制：`before_tool_handler` 将 span 压栈，`after_tool_handler` 出栈。Sub-Crew 的沙盒工具调用以栈顶 span（即 `skill_loader`）为父节点，自动形成树状嵌套。模块缓存确保 langfuse_trace 全局状态在所有 handler 间共享。

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

---

## 课堂代码演示学习指南

本节帮你按课程教学顺序阅读代码，建立完整的理解链路。

### 整体架构一览

```
┌─────────────────────────────────────────────────────────────────┐
│                         demo.py                                 │
│  HookRegistry + HookLoader → 加载 hooks.yaml（两层合并）         │
│  CrewObservabilityAdapter → 桥接 CrewAI 原生 Hook                │
│  Agent + SkillLoader + Crew → 执行任务                           │
└─────────────┬───────────────────────────────┬───────────────────┘
              │                               │
    ┌─────────▼──────────┐         ┌──────────▼──────────┐
    │   hook_framework/  │         │   shared_hooks/     │
    │                    │         │                     │
    │  registry.py       │         │  hooks.yaml         │
    │   EventType(7)     │         │   7种事件 → 12个    │
    │   HookContext      │◄────────│     handler 映射    │
    │   HookRegistry     │         │                     │
    │                    │         │  structured_log.py   │
    │  loader.py         │         │   → stderr JSON      │
    │   YAML解析+动态导入 │         │                     │
    │                    │         │  langfuse_trace.py   │
    │  crew_adapter.py   │         │   → Langfuse trace   │
    │   CrewAI 4种机制   │         └─────────────────────┘
    │   → 7种事件映射    │
    └────────────────────┘
```

### 学习路线（建议按顺序阅读）

---

#### 第一步：理解事件模型——EventType + HookContext + HookRegistry

**对应课文**：第二节"5+2 事件体系"、第三节"HookRegistry"

**阅读文件**：`hook_framework/registry.py`

| 重点区域 | 行号 | 看什么 |
|---------|------|--------|
| `EventType` 枚举 | 16-23 | 7 个事件对齐 Agent Turn 生命周期 |
| `HookContext` 数据类 | 33-49 | `frozen=True`——不可变，防止 handler 互相污染 |
| `register()` | 57-59 | handler 按事件类型存入列表，支持多个 handler |
| `dispatch()` | 61-70 | try-except 包裹每个 handler——一个崩溃不影响其他 |
| `summary()` | 89-93 | 自省接口——打印注册的所有 handler |

**理解要点**：`dispatch()` 的异常隔离是核心设计。如果 Langfuse 网络超时，结构化日志照常输出——这就是课文说的"观测系统不能成为生产系统的单点故障"。

**验证**：`python3 -m pytest tests/test_registry.py -v` 看 6 个测试覆盖的场景。

---

#### 第二步：看声明式配置——hooks.yaml

**对应课文**：第三节"声明式配置"

**阅读文件**：`shared_hooks/hooks.yaml`

```yaml
hooks:
  BEFORE_TURN:
    - handler: structured_log.before_turn_handler  # ← 模块名.函数名
  BEFORE_LLM:
    - handler: structured_log.before_llm_handler
    - handler: langfuse_trace.before_llm_handler   # ← 同一个事件可以有多个 handler
  ...
```

**理解要点**：配置格式是 `事件名 → handler 列表`。同一个事件可以挂多个 handler（如 `BEFORE_LLM` 同时触发日志和 Langfuse）。添加新的观测能力只需在 YAML 里加一行，不改框架代码。

---

#### 第三步：看最简单的 handler——structured_log

**对应课文**：第四节"结构化日志"

**阅读文件**：`shared_hooks/structured_log.py`

| 重点 | 看什么 |
|------|--------|
| `_emit(ctx)` | 从 HookContext 提取字段 → 构建 JSON → `print(json, file=sys.stderr)` |
| 各 handler 函数 | 每个只是调用 `_emit(ctx)` + 少量字段定制 |

**理解要点**：这是最简单的 handler 实现——纯函数，无状态，只输出。理解了这个，就理解了 handler 的接口约定：接收 `HookContext`，做想做的事，不返回值。

---

#### 第四步：看 Langfuse handler——从"能看"到"能追踪"

**对应课文**：第四节"Langfuse 全链路追踪"

**阅读文件**：`shared_hooks/langfuse_trace.py`

| 重点区域 | 看什么 |
|---------|--------|
| `_ensure_client()` | 懒初始化——首次事件触发时才创建 Langfuse 客户端 |
| `_ensure_trace()` | `create_trace_id(seed=session_id)` 保证幂等性——同一 session 同一 trace |
| `_get_otel_span_id()` | 从 OTel span 对象中提取 span ID，用于构建父子关系 |
| `_get_parent_context()` | 栈顶 span 为父；栈空则 root span 为父 |
| `_get_root_context()` | 始终以 root span 为父（用于 GENERATION / TASK_COMPLETE） |
| `before_tool_handler()` | 开启 TOOL span → 存入 `_pending_spans` + 压入 `_span_stack` |
| `after_tool_handler()` | 从 `_pending_spans` 取出 span → 写入 output → `end()` → 出栈 |
| `after_turn_handler()` | 创建 GENERATION，始终挂在 root span 下 |
| `flush_and_close()` | 清空 span 栈 + 关闭孤儿 span + 结束根 span + flush 客户端 |

**理解要点**：span 栈是实现树状追踪的核心。`before_tool_handler` 将新 span 压栈，Sub-Crew 的工具调用以栈顶为父节点自动嵌套。`after_tool_handler` 出栈恢复上层。GENERATION 和 TASK_COMPLETE 始终用 `_get_root_context()` 挂在根 span 下，避免被嵌套到工具 span 内。

---

#### 第五步：看加载引擎——HookLoader

**对应课文**：第三节"两层加载"

**阅读文件**：`hook_framework/loader.py`

| 重点区域 | 看什么 |
|---------|--------|
| `load_from_directory()` | 解析 hooks.yaml → `importlib.util.spec_from_file_location` 动态导入 |
| `_module_cache` | 模块缓存——同一模块只加载一次，确保全局状态共享 |
| 路径穿越防护 | `module_path.is_relative_to(hooks_dir.resolve())` 阻止 `../../` 攻击 |
| `load_two_layers()` | 先加载 `global_dir` → 再加载 `workspace_dir/hooks/`，追加不覆盖 |

**理解要点**：模块缓存是关键——`importlib.util.module_from_spec()` 每次都会创建新的模块实例，如果同一个 YAML 中多次引用同一模块（如 `langfuse_trace` 被多个事件引用），不缓存会导致每个 handler 持有独立的全局状态（`_span_stack`、`_pending_spans` 等），span 栈机制会失效。

---

#### 第六步：看适配层——CrewAI 4种机制 → 7种事件

**对应课文**：第三节"CrewAI 适配层"

**阅读文件**：`hook_framework/crew_adapter.py`

| 重点区域 | 看什么 |
|---------|--------|
| `_before_llm()` | `_current_turn_has_llm` flag：首次 LLM 调用触发 BEFORE_TURN + BEFORE_LLM，后续只触发 BEFORE_LLM |
| `_before_tool()` / `_after_tool()` | 直接映射 + 工具输入/输出截断 |
| `make_step_callback()` | step_callback → AFTER_TURN，重置 `_current_turn_has_llm` 开始新轮 |
| `cleanup()` | `_cleaned` flag 防止双重触发 + `clear_*_hooks()` 清理全局注册 |
| 不使用 `@after_llm_call` | 注释说明：注册该 hook 会干扰 CrewAI 的 function calling 工具调度 |

**理解要点**：适配层的核心价值是**框架无关性**——如果将来换成 LangChain 或 AutoGen，只需重写适配层，所有 handler（日志、Langfuse、审计）一行不改。

---

#### 第七步：看 Workspace 层定制——task_audit

**对应课文**：第三节"两层架构"的 Workspace 层

**阅读文件**：`workspace/demo_agent/hooks/`

- `hooks.yaml`：只注册了一个 handler——`TASK_COMPLETE → task_audit.write_audit_entry`
- `task_audit.py`：在任务完成时往 `audit.log` 写一条 JSON（session_id + 时间戳 + 产出预览）

**理解要点**：这个 handler 只有 demo_agent 有，其他 Agent 不会被审计。这就是 Workspace 层的意义——业务定制，而非全局行为。

---

#### 第八步：端到端串联——demo.py

**对应课文**：整课的代码演示部分

**阅读文件**：`demo.py`

| 阶段 | 行号范围 | 做了什么 |
|------|---------|---------|
| Hook 初始化 | 前半段 | `HookRegistry()` → `HookLoader.load_two_layers()` → 打印 handler 注册情况 |
| CrewAI 适配 | 中段 | `CrewObservabilityAdapter` 安装全局 hooks + atexit 注册 cleanup |
| Agent 构建 | 中段 | `build_bootstrap_prompt()`（25课复用）+ `SkillLoaderTool` + `LLM` + `Agent` |
| 执行 | 后段 | `crew.kickoff()` → Hook 自动拦截所有事件 → 日志 + Langfuse |
| 清理 | 末段 | `adapter.cleanup()` → SESSION_END → flush Langfuse |

**理解要点**：demo.py 的 Agent 构建部分和 25 课完全相同（Bootstrap + SkillLoader + 沙盒）。唯一的新增是 Hook 框架的初始化和适配——Agent 代码本身不感知任何观测逻辑。

---

#### 第九步：验证——跑测试看效果

```bash
cd /path/to/crewai_mas_demo/m5l30

# 1. 单元测试（18个，无需 LLM/Langfuse）
python3 -m pytest tests/ -v --ignore=tests/test_e2e_hooks.py

# 2. 端到端演示（需 LLM + Langfuse + 沙盒）
python3 demo.py

# 3. 查看 Langfuse Dashboard
#    → http://localhost:3000 → 选择 course-demo 项目 → Traces
#    → 检查 Trace 树：TOOL span + GENERATION + task-complete
```

---

### 学习检查清单

完成以上九步后，你应该能回答：

- [ ] `HookContext` 为什么是 `frozen=True`？（防止 handler 互相污染）
- [ ] `dispatch()` 的异常隔离有什么作用？（Langfuse 崩溃不影响日志输出）
- [ ] 为什么不能用 `@after_llm_call`？（会干扰 CrewAI 的 function calling 调度）
- [ ] 两层加载中，全局层和 Workspace 层各负责什么？
- [ ] Langfuse handler 为什么用 `_pending_spans` dict 追踪 TOOL span？（配对开启/关闭，捕获真实耗时）
- [ ] 如果要给另一个 Agent 加一个"工具使用统计"hook，需要改哪些文件？（只需在那个 Agent 的 workspace/hooks/ 下加 yaml + py）
- [ ] 适配层的框架无关性价值是什么？（换框架只需重写适配层，所有 handler 不变）
