# 第17课：项目实战2——XiaoPaw 飞书 AI 助理

本课构建一个完整的飞书（Lark）AI 助理：从消息接收、Session 管理、Main Crew + Sub-Crew 双层编排、到 Skills 生态系统，形成可运行的生产级 Agent 应用。

> **核心教学点**：飞书 WebSocket 集成、Runner 并发模型（per-routing_key 串行队列）、Main Crew + Sub-Crew 双层架构、SkillLoaderTool 渐进式披露、9 个内置 Skill、凭证隔离、CronService 定时任务

---

## 代码位置

```
xiaopow/                               # 完整项目代码（独立仓库）
├── xiaopaw/                           # 主包
│   ├── main.py                        # 进程入口
│   ├── runner.py                      # 执行引擎（per-key 串行队列）
│   ├── models.py                      # InboundMessage + SenderProtocol
│   ├── agents/
│   │   ├── main_crew.py               # Main Crew 工厂
│   │   ├── skill_crew.py              # Sub-Crew 工厂（MCP + Sandbox）
│   │   └── config/                    # agents.yaml + tasks.yaml
│   ├── tools/
│   │   ├── skill_loader.py            # SkillLoaderTool（渐进式披露）
│   │   ├── add_image_tool_local.py    # 本地图片 → Base64
│   │   ├── baidu_search_tool.py       # 百度千帆搜索
│   │   └── intermediate_tool.py       # 中间产物保存
│   ├── feishu/
│   │   ├── listener.py                # WebSocket 事件监听
│   │   ├── sender.py                  # 卡片消息发送
│   │   ├── downloader.py              # 附件下载
│   │   └── session_key.py             # routing_key 解析
│   ├── session/manager.py             # Session 管理（index.json + JSONL）
│   ├── cron/service.py                # 定时任务（at/every/cron 三模式）
│   ├── cleanup/service.py             # 存储清理 + 凭证写入
│   ├── llm/aliyun_llm.py              # AliyunLLM 适配器
│   ├── observability/                 # structlog + Prometheus
│   ├── api/test_server.py             # 本地测试 HTTP API
│   └── skills/                        # 9 个内置 Skill
│       ├── load_skills.yaml           # Skill 注册表
│       ├── pdf/ docx/ pptx/ xlsx/     # 文件处理
│       ├── feishu_ops/                # 飞书 API（16 个脚本）
│       ├── scheduler_mgr/            # 定时任务管理
│       ├── baidu_search/              # 百度搜索
│       ├── web_browse/                # 浏览器操控
│       └── history_reader/            # 对话历史（reference 类型）
├── data/workspace/                    # 运行时工作空间
├── tests/                             # 562 单元测试 + 29 集成测试
├── config.yaml.template               # 配置模板
└── docs/                              # 设计文档
```

---

## 快速开始

```bash
cd /path/to/xiaopow

# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置
cp config.yaml.template config.yaml
# 编辑 config.yaml 填入飞书 App ID/Secret、QWEN_API_KEY 等

# 3. 启动 Sandbox
docker run -d --security-opt seccomp=unconfined --rm -it -p 8022:8080 ghcr.io/agent-infra/sandbox:latest

# 4. 运行
python -m xiaopaw
```

---

## 课堂代码演示学习指南

### 整体架构一览

```
飞书消息
   │  WebSocket
   ▼
FeishuListener
   │  解析 → InboundMessage
   ▼
Runner（per-routing_key 串行队列）
   │
   ├─ 斜杠命令？ /new /verbose /help
   │     → 直接处理
   │
   ├─ 有附件？
   │     → FeishuDownloader 下载到 session workspace
   │
   ├─ SessionManager 加载/创建 Session
   │
   ├─ FeishuSender.send_thinking()  ← "正在思考" 卡片
   │
   ▼
Main Crew（单 Agent + SkillLoaderTool）
   │
   │  Phase 1: 看"菜单"
   │  <available_skills> XML 描述
   │
   │  Phase 2: 调用 Skill
   │  skill_loader(skill_name="pdf", task_context={...})
   │
   ▼
Sub-Crew（独立实例 + MCP Sandbox）
   │
   │  sandbox_execute_code / sandbox_execute_bash
   │  sandbox_file_operations
   │
   ▼
FeishuSender.update_card()  ← 更新回复卡片
   │
   ▼
SessionManager.append()  ← 持久化对话历史
```

### 学习路线

---

#### 第一步：看消息接收链路

**阅读文件**：`xiaopaw/feishu/listener.py`

| 事件类型 | 处理 |
|---------|------|
| `im.message.receive_v1` | 解析消息 → InboundMessage → Runner |
| `im.chat.member.bot.added_v1` | 机器人入群通知 |

**理解要点**：飞书使用 WebSocket 长连接（不需要公网 IP），通过 `allowed_chats` 白名单控制哪些群聊可以触发 Agent。

---

#### 第二步：看 Runner 并发模型

**阅读文件**：`xiaopaw/runner.py`

```
routing_key_1（用户A）: ──msg1──msg2──msg3──  串行
routing_key_2（群B）:   ──msg1──msg2──        串行
                        ↕ 并行 ↕
```

| 设计决策 | 原因 |
|---------|------|
| 同一 routing_key 串行 | 同一对话的消息必须按顺序处理 |
| 不同 routing_key 并行 | 不同用户/群互不阻塞 |
| Worker 空闲超时退出 | 释放内存，按需创建 |

**理解要点**：每个 routing_key 有独立的 `asyncio.Queue` + worker 协程。这是"per-key 串行队列"模式——在并发和一致性之间取得平衡。

---

#### 第三步：看双层 Crew 架构

**阅读文件**：`xiaopaw/agents/main_crew.py` + `skill_crew.py`

| 层 | 角色 | LLM | 工具 |
|----|------|-----|------|
| Main Crew | 意图理解 + 任务规划 | qwen3.6-max-preview | SkillLoaderTool |
| Sub-Crew | 具体任务执行 | qwen3-max | MCP Sandbox 工具 |

**理解要点**：
- Main Crew 只有一个 Tool（SkillLoaderTool）——"单工具原则"，所有能力通过 Skill 提供
- Sub-Crew 每次调用都创建新实例（工厂模式），防止状态污染
- Main Crew 的历史不进入 Sub-Crew，Sub-Crew 的执行细节不进入 Main Crew——上下文隔离

---

#### 第四步：看 SkillLoaderTool 渐进式披露

**阅读文件**：`xiaopaw/tools/skill_loader.py`

| 阶段 | 时机 | 加载内容 |
|------|------|---------|
| Phase 1 | 工具初始化 | YAML frontmatter → XML "菜单"（十几个字） |
| Phase 2 | 被调用时 | 完整 SKILL.md 指令（可能几百行） |

**理解要点**：Main Agent 的 context 是稀缺资源。Phase 1 只注入轻量"菜单"，Phase 2 按需加载且缓存。`{var}` 会被转义为 `{{var}}`，防止 CrewAI 模板引擎误解析。

---

#### 第五步：看 9 个内置 Skill

**阅读文件**：`xiaopaw/skills/load_skills.yaml` + 各 Skill 目录

| Skill | 类型 | 能力 |
|-------|------|------|
| `pdf` | task | PDF → Markdown |
| `docx` | task | 生成 Word 文档 |
| `pptx` | task | 生成 PPT |
| `xlsx` | task | 生成 Excel |
| `feishu_ops` | task | 飞书 API（16 个脚本：发消息/读文档/管日历/写表格...） |
| `scheduler_mgr` | task | 定时任务 CRUD |
| `baidu_search` | task | 百度搜索 + 内容抓取 |
| `web_browse` | task | 无头浏览器操控 |
| `history_reader` | reference | 分页读取对话历史（不需要 Sub-Crew） |

---

#### 第六步：看凭证隔离

**阅读文件**：`xiaopaw/cleanup/service.py`（搜索 `_write_credentials`）

```
启动时：CleanupService 写入凭证
  → data/workspace/.config/feishu.json
  → data/workspace/.config/baidu.json
  权限 0600，原子写入（write-then-rename）

Skill 脚本直接读取 .config/ 文件
  → LLM 上下文中没有任何凭证
```

**理解要点**：凭证不通过 Agent 的 backstory 或 Task 描述传递——LLM 永远看不到 API Key。脚本在 Sandbox 内直接读取 `.config/` 文件。

---

#### 第七步：看 Session 管理

**阅读文件**：`xiaopaw/session/manager.py`

| 文件 | 格式 | 作用 |
|------|------|------|
| `index.json` | JSON | routing_key → active_session_id 映射 |
| `{session_id}.jsonl` | JSONL | 对话历史（meta 行 + user/assistant 对） |

**理解要点**：并发安全通过 `asyncio.Lock` + 原子写入（`write-then-rename`）保证。`/new` 斜杠命令创建新 Session，历史清零。

---

#### 第八步：看定时任务

**阅读文件**：`xiaopaw/cron/service.py`

| 模式 | 示例 | 场景 |
|------|------|------|
| `at` | `"2026-04-25T09:00"` | 一次性提醒 |
| `every` | `"30m"` | 固定间隔 |
| `cron` | `"0 9 * * 1-5"` | 标准 cron 表达式 |

**理解要点**：定时任务通过构造 fake `InboundMessage` 走正常 Runner 管线——不需要单独的执行路径。`tasks.json` 支持热加载（mtime + size 双检测）。

---

#### 第九步：看测试和本地调试

**阅读文件**：`xiaopaw/api/test_server.py`

```bash
# 本地测试（不需要飞书环境）
curl -X POST http://localhost:8080/api/test/message \
  -H "Content-Type: application/json" \
  -d '{"routing_key": "test", "content": "帮我搜索 Qwen3 最新动态"}'
```

**理解要点**：`TestAPI` 通过 `CaptureSender`（实现 `SenderProtocol`）拦截 Agent 回复，将异步飞书消息流转化为同步 HTTP 响应——方便本地开发调试。

---

### 学习检查清单

- [ ] per-routing_key 串行队列解决了什么问题？（同一对话按序处理，不同对话并行——兼顾一致性和并发）
- [ ] Main Crew 为什么只有一个 Tool？（单工具原则——所有能力通过 Skill 提供，降低 Agent 决策复杂度）
- [ ] Sub-Crew 为什么每次创建新实例？（工厂模式防状态污染——CrewAI 内部有运行时状态）
- [ ] 凭证为什么不写在 backstory 中？（LLM 能看到 backstory，凭证写在 Sandbox 文件中更安全）
- [ ] reference 和 task 两种 Skill 的区别？（reference 返回文本给 Main Agent，task 启动独立 Sub-Crew 在 Sandbox 执行）
- [ ] 定时任务如何触发 Agent？（构造 fake InboundMessage 走正常 Runner 管线）
- [ ] `CaptureSender` 的作用？（实现 SenderProtocol，将异步消息流转为同步 HTTP 响应，用于本地测试）
