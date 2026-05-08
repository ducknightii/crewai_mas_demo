# 第21课：搜索式记忆——pgvector 混合检索与异步索引

本课演示基于 pgvector 的搜索式记忆系统：异步后台索引对话轮次，通过向量+全文混合检索实现跨 Session 语义召回。

> **核心教学点**：pgvector 混合检索（向量+BM25）、四步索引流水线、异步后台索引（`asyncio.create_task`）、search_memory Skill、幂等写入

---

## 目录结构

```
m3l21/
├── m3l21_search_memory.py         # 主演示：异步索引 + 搜索 Skill
├── indexer.py                     # 四步索引流水线
├── schema.sql                     # DDL：memories 表 + 索引
├── pgvector-docker-compose.yaml   # PostgreSQL + pgvector 容器
├── test_m3l21.py                  # 单元测试（14 个用例）
└── skills/
    └── search_memory/
        └── scripts/               # search.py 的编译缓存
```

> 💡 search_memory Skill 的源码在共享目录 `skills/search_memory/`

---

## 快速开始

```bash
# 1. 启动 pgvector
cd /path/to/crewai_mas_demo/m3l21
docker compose -f pgvector-docker-compose.yaml up -d

# 2. 启动 AIO-Sandbox（如未启动）
docker run -d --security-opt seccomp=unconfined --rm -it -p 8022:8080 ghcr.io/agent-infra/sandbox:latest

# 3. 运行演示
cd /path/to/crewai_mas_demo
python3 m3l21/m3l21_search_memory.py
```

---

## 课堂代码演示学习指南

### 整体架构一览

```
对话轮次
  │
  ├─ 同步路径：Crew 执行 + Session 持久化
  │
  └─ 异步路径：asyncio.create_task(async_index_turn(...))
                    │
                    ▼
              四步索引流水线
              ┌────────────────────────────┐
              │ 1. parse_turns()           │
              │    JSONL → (user, assistant)│
              │                            │
              │ 2. extract_summary_and_tags│
              │    LLM(qwen3.6-max) →      │
              │    {summary, tags}          │
              │                            │
              │ 3. embed_texts()           │
              │    text-embedding-v3 →      │
              │    [summary_vec, msg_vec]   │
              │    1024 维                  │
              │                            │
              │ 4. upsert_memory()         │
              │    ON CONFLICT DO NOTHING   │
              │    search_text = user+tags  │
              └────────────────────────────┘
                    │
                    ▼
              pgvector（memories 表）
                    │
                    ▼
              search_memory Skill
              ┌────────────────────────────┐
              │ 三种检索模式：              │
              │ • vector:  余弦相似度       │
              │ • fulltext: BM25 近似       │
              │ • hybrid:  0.7×向量+0.3×全文 │
              │                            │
              │ 过滤器：                    │
              │ • tags（数组重叠 &&）        │
              │ • days（时间范围）           │
              │ • routing_key（精确匹配）   │
              └────────────────────────────┘
```

### 学习路线

---

#### 第一步：看数据库 Schema

**阅读文件**：`schema.sql`

| 列 | 类型 | 作用 |
|----|------|------|
| `summary_vec` | `vector(1024)` | 摘要的向量表示 |
| `message_vec` | `vector(1024)` | 原始消息的向量表示 |
| `search_text` | `text` | user_message + tags 拼接，用于全文检索 |
| `search_tsv` | `tsvector` | **GENERATED ALWAYS AS STORED**——自动维护 |
| `tags` | `text[]` | 领域标签数组 |

**理解要点**：向量和全文索引共存于同一张表——不需要单独的向量数据库。`search_tsv` 使用 `GENERATED ALWAYS AS STORED` 自动更新，零维护成本。

---

#### 第二步：看四步索引流水线

**阅读文件**：`indexer.py`

| 步骤 | 函数 | 输入 | 输出 |
|------|------|------|------|
| 1 | `parse_turns()` | JSONL 文件 | `[(user_msg, assistant_reply, ts)]` |
| 2 | `extract_summary_and_tags()` | user + assistant | `{summary, tags}` |
| 3 | `embed_texts()` | `[summary, message]` | `[vec_1024, vec_1024]` |
| 4 | `upsert_memory()` | 完整记录 | INSERT ... ON CONFLICT DO NOTHING |

**理解要点**：每步独立可测。ID 生成使用 `SHA-256(session_id + turn_ts + user_message[:32])`，保证同一轮对话的 ID 稳定不变——重复索引不会产生重复数据。

---

#### 第三步：看异步后台索引

**阅读文件**：`m3l21_search_memory.py`（搜索 `run_and_index`）

```python
asyncio.create_task(async_index_turn(
    session_id, routing_key, user_message, assistant_reply, turn_start_ts
))
```

**理解要点**：`asyncio.create_task()` 是 Fire-and-Forget 模式——索引任务在后台线程池执行，不阻塞用户对话。`async_index_turn` 内部使用 `run_in_executor()` 将同步的索引操作放入 `ThreadPoolExecutor`。

---

#### 第四步：看三种检索模式

**阅读文件**：`skills/search_memory/scripts/search.py`

| 模式 | 算法 | 适用场景 |
|------|------|---------|
| `vector` | 余弦相似度 `1 - cosine_distance` | 语义/模糊查询（"那个飞行的事"） |
| `fulltext` | `ts_rank` + `plainto_tsquery` | 精确关键词（"PDF 转换"） |
| `hybrid` | `0.7 × vector + 0.3 × fulltext` | **推荐**：兼顾语义和关键词 |

**理解要点**：混合检索的权重 `0.7:0.3` 偏向向量——大多数记忆查询是语义性的（"之前讨论过的那个方案"），但关键词补充能捕获向量遗漏的精确匹配。

---

#### 第五步：看渐进式放松策略

**阅读文件**：`skills/search_memory/SKILL.md`（搜索"重试策略"）

```
结果为空时的放松顺序：
1. 移除 --days 限制
2. 移除 --tags 过滤
3. 切换为纯 vector 模式
```

**理解要点**：这是 SKILL.md 中的"行为指令"——Sub-Crew 在 Sandbox 中执行搜索脚本，如果结果为空，按策略逐步放宽条件重试。

---

#### 第六步：看三轮演示

**阅读文件**：`m3l21_search_memory.py`（搜索 `main`）

| 轮次 | 任务 | 索引？ | 验证点 |
|------|------|--------|--------|
| 第1轮 | 搜索 Qwen3 最新动态 | ✅ 后台索引 | 正常对话 + 索引触发 |
| 第2轮 | 对比 pgvector vs Qdrant | ✅ 后台索引 | 技术讨论也被索引 |
| 第3轮 | 用 search_memory 搜索历史 | ❌ | 跨 Session 语义召回 |

---

#### 第七步：运行测试

```bash
cd /path/to/crewai_mas_demo
python3 -m pytest m3l21/test_m3l21.py -v
```

14 个测试覆盖：parse_turns（4）、extract_summary（3）、embed_texts（1）、upsert（1）、search（3）、async_index（1）。

---

### 学习检查清单

- [ ] 为什么用 pgvector 而不是单独的向量数据库？（向量是"多一列"——复用现有 PostgreSQL 基础设施）
- [ ] 混合检索公式？（`score = 0.7 × vector_similarity + 0.3 × BM25_rank`）
- [ ] 幂等索引如何实现？（SHA-256 稳定 ID + `ON CONFLICT DO NOTHING`）
- [ ] `GENERATED ALWAYS AS STORED` 解决什么问题？（tsvector 自动更新，零维护）
- [ ] 异步索引为什么用 `run_in_executor`？（索引涉及同步 DB 操作，不能直接 await）
- [ ] 渐进式放松的三步顺序？（移除时间 → 移除标签 → 切换纯向量）
