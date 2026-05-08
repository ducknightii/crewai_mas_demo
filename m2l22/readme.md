# 第22课：项目实战3——XiaoPaw 三层记忆系统

本课在 XiaoPaw（L17）基础上增加完整的三层记忆架构：上下文生命周期管理（L19）、文件系统记忆（L20）、pgvector 搜索式记忆（L21），让 Agent 具备跨 Session 持久记忆能力。

> **核心教学点**：三层记忆架构（上下文/文件/搜索）、Bootstrap 导航骨架、剪枝+压缩、memory-save/skill-creator/memory-governance Skill、pgvector 混合检索、异步后台索引、Onboarding SOP 自删除

---

## 代码位置

```
xiaopaw-with-memory/                   # 完整项目代码（独立仓库）
├── xiaopaw/
│   ├── memory/                        # 三层记忆核心模块
│   │   ├── bootstrap.py               # L1: 四文件骨架 → Agent backstory
│   │   ├── context_mgmt.py            # L1: 剪枝 + 压缩 + Session 持久化
│   │   └── indexer.py                 # L3: 异步 pgvector 索引流水线
│   ├── agents/
│   │   ├── main_crew.py               # MemoryAwareCrew（@before_llm_call Hook）
│   │   ├── skill_crew.py              # Sub-Crew 工厂
│   │   └── config/                    # agents.yaml + tasks.yaml
│   ├── tools/
│   │   ├── skill_loader.py            # SkillLoaderTool
│   │   └── ...                        # 其他工具（同 L17）
│   ├── skills/                        # 18 个 Skill（L17 的 9 个 + L20-22 新增）
│   │   ├── memory-save/               # L2: 写入用户偏好/事实
│   │   ├── skill-creator/             # L2: 固化 SOP 为 SKILL.md
│   │   ├── memory-governance/         # L2: 审计+清理记忆
│   │   ├── search_memory/             # L3: pgvector 语义检索
│   │   └── ...                        # 其他 Skill（同 L17）
│   └── ...                            # 其他模块（同 L17）
├── workspace-init/                    # Bootstrap 模板文件
│   ├── soul.md                        # Agent 身份/人格
│   ├── user.md                        # 用户画像模板
│   ├── agent.md                       # Agent 规则 + Onboarding SOP
│   └── memory.md                      # 记忆索引（初始为空）
├── data/ctx/                          # 上下文持久化
│   ├── {routing_key}_ctx.json         # 压缩后的上下文快照
│   └── {routing_key}_raw.jsonl        # 追加式完整历史
├── sandbox-docker-compose.yaml        # AIO-Sandbox Docker
├── tests/                             # 642 单元测试 + 29 集成测试
└── docs/test-design-course22.md       # L22 集成测试设计文档
```

---

## 快速开始

```bash
cd /path/to/xiaopaw-with-memory

# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置
cp config.yaml.template config.yaml
# 编辑 config.yaml

# 3. 启动外部依赖
docker compose -f sandbox-docker-compose.yaml up -d  # AIO-Sandbox
# pgvector（如需 L3 搜索记忆）参考 crewai_mas_demo/m3l21/pgvector-docker-compose.yaml

# 4. 运行
python -m xiaopaw
```

---

## 课堂代码演示学习指南

### 整体架构一览

```
┌─────────────────────────────────────────────────────┐
│  三层记忆架构                                         │
│                                                     │
│  Layer 1（L19）: 上下文生命周期                        │
│  ┌─────────────────────────────────────────────┐    │
│  │ Bootstrap        ← soul/user/agent/memory.md │    │
│  │ → Agent backstory 注入                        │    │
│  │                                              │    │
│  │ @before_llm_call Hook:                       │    │
│  │   首次: 从 ctx.json 恢复历史                   │    │
│  │   每次: prune(工具结果) → compress(超阈值)     │    │
│  │                                              │    │
│  │ Session 持久化:                               │    │
│  │   ctx.json（覆写）+ raw.jsonl（追加）          │    │
│  └─────────────────────────────────────────────┘    │
│                                                     │
│  Layer 2（L20）: 文件系统记忆                         │
│  ┌─────────────────────────────────────────────┐    │
│  │ memory-save    → 写入偏好/事实到 workspace/    │    │
│  │ skill-creator  → 固化 SOP 为 SKILL.md         │    │
│  │ memory-governance → 审计清理（8 项健康检查）     │    │
│  │                                              │    │
│  │ memory.md 是导航索引（≤200 行）                │    │
│  │ 实际内容在 memory_{topic}.md 中               │    │
│  └─────────────────────────────────────────────┘    │
│                                                     │
│  Layer 3（L21）: 搜索式记忆                          │
│  ┌─────────────────────────────────────────────┐    │
│  │ 写入: asyncio.create_task(async_index_turn)  │    │
│  │   parse → extract_summary → embed → upsert   │    │
│  │                                              │    │
│  │ 读取: search_memory Skill                    │    │
│  │   hybrid: 0.7×向量 + 0.3×BM25               │    │
│  │   过滤: tags / days / routing_key            │    │
│  └─────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────┘
```

### 学习路线

---

#### 第一步：理解三层记忆的分工

| 层 | 存什么 | 什么时候写 | 什么时候读 | 时间尺度 |
|----|--------|-----------|-----------|---------|
| L1 上下文 | LLM 对话状态 | 每次 LLM 调用（剪枝/压缩）+ 每轮结束（ctx.json） | Session 恢复时 | 跨 Session |
| L2 文件 | 稳定事实（偏好、SOP、持仓） | 用户表达偏好或确认工作流时 | Bootstrap 注入（每次 Session 启动） | 永久 |
| L3 搜索 | 瞬态对话产出（分析结论、临时判断） | 每轮自动后台索引 | 按需语义检索（search_memory Skill） | 永久 |

**理解要点**：L2 存"稳定事实"（始终在 context 中），L3 存"瞬态产出"（按需检索）——这是两层的核心区别。

---

#### 第二步：看 Bootstrap 导航骨架

**阅读文件**：`xiaopaw/memory/bootstrap.py` + `workspace-init/` 四个模板文件

| 文件 | XML 标签 | 内容 |
|------|----------|------|
| `soul.md` | `<soul>` | 不可变的人格/风格 |
| `user.md` | `<user_profile>` | 用户画像（记忆写入更新） |
| `agent.md` | `<agent_rules>` | 工作规则 + Onboarding SOP |
| `memory.md` | `<memory_index>` | 导航索引（≤200 行，指向 topic 文件） |

**理解要点**：Bootstrap 只注入"骨架"——告诉 Agent 信息在哪，不全量加载。memory.md 是目录，不是内容存储。

---

#### 第三步：看 MemoryAwareCrew 的 Hook 集成

**阅读文件**：`xiaopaw/agents/main_crew.py`（搜索 `MemoryAwareCrew`）

| Hook 时机 | 行为 |
|----------|------|
| 首次 LLM 调用 | 从 ctx.json 恢复历史 + 合并 Bootstrap 系统消息 |
| 每次 LLM 调用 | 剪枝（清除旧工具结果）→ 压缩（超 45% 阈值时分块摘要） |
| 返回值 | `None`（继续执行，不阻断） |

**理解要点**：为什么用 `@before_llm_call` 而不是 `@after_llm_call`？因为 CrewAI 的 after hook 会 `str()` 返回值，破坏工具调度的类型检查。

---

#### 第四步：看文件记忆的三个 Skill

**阅读文件**：`xiaopaw/skills/memory-save/SKILL.md` + `skill-creator/SKILL.md` + `memory-governance/SKILL.md`

| Skill | 触发场景 | 写入位置 |
|-------|---------|---------|
| memory-save | 用户表达偏好（"我喜欢..."） | workspace/ 下的 md 文件 |
| skill-creator | 用户描述重复工作流 | skills/ 下的 SKILL.md |
| memory-governance | 记忆积累过多 / 定期审计 | 清理 workspace/ 和 skills/ |

**理解要点**：memory-save 有严格的四步准入协议——准入控制（三个月价值？）→ 目标选择（写哪个文件？）→ 阈值门控（memory.md < 150 行？）→ 写入执行（str_replace + 回读验证）。

---

#### 第五步：看 pgvector 搜索记忆

**阅读文件**：`xiaopaw/memory/indexer.py` + `xiaopaw/skills/search_memory/`

写入路径（每轮自动触发）：
```
对话结束 → asyncio.create_task → ThreadPoolExecutor
  → parse_turns → extract_summary_and_tags(LLM)
  → embed_texts(text-embedding-v3, 1024维)
  → upsert_memory(ON CONFLICT DO NOTHING)
```

读取路径（按需触发）：
```
用户: "之前讨论的那个方案..."
  → search_memory Skill → Sub-Crew → scripts/search.py
  → hybrid: 0.7×cosine + 0.3×BM25
  → 返回匹配记录
```

**理解要点**：search_memory 的 SKILL.md 被设计为"主动激活"——当用户消息暗示依赖历史信息时（"根据之前的讨论"），Agent 应主动调用，不需要用户明确要求。

---

#### 第六步：看 Onboarding SOP 自删除

**阅读文件**：`workspace-init/agent.md`（搜索 "onboarding"）

六步自引导流程：命名 → 用途 → 风格 → 用户信息 → 禁忌 → SOP 训练。每一步都使用 memory-save 持久化进度。全部完成后，agent.md 中的 Onboarding SOP 章节自动删除——"任务完成即自毁"。

**理解要点**：这是"自修改 Agent 规则"的演示——Agent 不仅能写记忆文件，还能修改自己的行为规则文件。Onboarding 完成后不再需要 SOP，删除它释放 context 空间。

---

#### 第七步：看集成测试设计

**阅读文件**：`docs/test-design-course22.md` + `tests/integration/test_course22_cases.py`

| 测试组 | 场景 | 验证层 |
|--------|------|--------|
| U (P2) | SOP Skill 路由 | L2: 创建投资报告 Skill → 一句话触发 |
| V (P3) | 持仓决策 | L2: Bootstrap 注入持仓 → Agent 直接知道 |
| W (P4) | Onboarding 自删除 | L2: 完成后 agent.md 中 SOP 消失 |
| X (P5) | SOP 全生命周期 | L2: 描述 → 结构化 → 确认 → 创建 → 触发 |
| Y (P6) | 历史分析召回 | L3: 自动索引 → pgvector → search_memory |

---

### 学习检查清单

- [ ] L2 文件记忆和 L3 搜索记忆的核心区别？（L2 存稳定事实始终在 context 中，L3 存瞬态产出按需检索）
- [ ] Bootstrap 为什么只注入"骨架"不注入全量？（context 是稀缺资源，骨架只占几十行 token）
- [ ] memory.md 的 200 行限制是什么？（导航索引不是内容存储，超出会浪费 context）
- [ ] 为什么不用 `@after_llm_call`？（CrewAI 框架 `str()` 返回值，破坏工具调度类型检查）
- [ ] 异步索引为什么用 `create_task` + `run_in_executor`？（create_task 不阻塞用户对话，run_in_executor 将同步 DB 操作放入线程池）
- [ ] Onboarding SOP 完成后为什么要自删除？（释放 context 空间，不再需要的指令只是噪声）
- [ ] search_memory 什么时候应该"主动激活"？（用户消息暗示依赖历史信息时，如"之前的方案"、"上次的结论"）
