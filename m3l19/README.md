# 第19课：上下文生命周期——Bootstrap、剪枝与压缩

本课演示 Agent 上下文的完整生命周期管理：Bootstrap 导航骨架注入、工具结果剪枝、超阈值压缩、以及双文件 Session 持久化。

> **核心教学点**：`build_bootstrap_prompt`（四文件骨架）、`prune_tool_results`（工具结果剪枝）、`maybe_compress`（分块摘要压缩）、`@before_llm_call` Hook、双文件持久化（ctx.json + raw.jsonl）

---

## 目录结构

```
m3l19/
├── m3l19_context_mgmt.py          # 核心模块：六大功能区
├── test_context_mgmt.py           # 单元测试（23 个用例）
├── workspace/
│   ├── soul.md                    # Agent 身份/风格定义
│   ├── user.md                    # 用户画像（晓寒）
│   ├── agent.md                   # Agent SOP/规则
│   ├── memory.md                  # 记忆索引（200 行上限）
│   └── sessions/
│       ├── demo_ctx.json          # 压缩后的上下文快照
│       └── demo_raw.jsonl         # 追加式完整历史
└── agent.log                      # 运行日志
```

---

## 快速开始

```bash
cd /path/to/crewai_mas_demo
python3 m3l19/m3l19_context_mgmt.py
```

---

## 课堂代码演示学习指南

### 整体架构一览

```
Session 开始
   │
   ▼
build_bootstrap_prompt()
   │  加载 soul.md → <soul>
   │  加载 user.md → <user_profile>
   │  加载 agent.md → <agent_rules>
   │  加载 memory.md → <memory_index>（≤200 行）
   │
   ▼  注入 Agent.backstory
   │
   ▼
每次 LLM 调用前（@before_llm_call）
   │
   ├─ 首次调用：load_session_ctx() 恢复历史
   │
   ├─ prune_tool_results()
   │     找到倒数第 N 个 user 消息
   │     之前的 tool 消息 → content 替换为 [已剪枝]
   │     保留 tool_call_id（结构合法性）
   │
   └─ maybe_compress()
         approx_tokens / model_limit > 0.45？
         ├─ 否 → 跳过
         └─ 是 → chunk_by_tokens() 分块
                  _summarize_chunk() 用 qwen3-turbo 摘要
                  替换为 <context_summary> 系统消息
                  保留最近 10 轮原文
   │
   ▼
LLM 执行
   │
   ▼
Session 持久化（每轮结束）
   ├─ save_session_ctx() → ctx.json（覆写）
   └─ append_session_raw() → raw.jsonl（追加 + 时间戳）
```

### 学习路线

---

#### 第一步：看 Bootstrap 导航骨架

**阅读文件**：`m3l19_context_mgmt.py`（搜索 `build_bootstrap_prompt`）

| 文件 | XML 标签 | 作用 |
|------|----------|------|
| `soul.md` | `<soul>` | 身份/风格（严谨、高效、结果导向） |
| `user.md` | `<user_profile>` | 用户画像（职业、偏好、技术栈） |
| `agent.md` | `<agent_rules>` | 工作 SOP（工具用法、调研流程、周报流程） |
| `memory.md` | `<memory_index>` | 记忆导航（≤200 行，超出截断） |

**理解要点**：Bootstrap 只注入"导航骨架"——告诉 Agent "你是谁、为谁工作、怎么工作、记得什么"。实际记忆内容按需读取（通过 FileReadTool），不全量塞入 context。

---

#### 第二步：看工具结果剪枝

**阅读文件**：`m3l19_context_mgmt.py`（搜索 `prune_tool_results`）

```
消息序列示例（keep_turns=2）：

[user] 搜索 Qwen3            ← 倒数第 2 个 user
[tool] 搜索结果（3000 字）     ← 剪枝！→ [已剪枝]
[assistant] 分析结果
[user] 写周报                 ← 倒数第 1 个 user（保护区）
[tool] 文件读取结果             ← 保留
[assistant] 周报内容
```

**理解要点**：工具结果是 context 膨胀的主要来源（一次搜索可能返回几千字）。剪枝只清除 `content`，保留 `tool_call_id` 确保消息结构合法（否则 LLM API 会报错）。

---

#### 第三步：看压缩机制

**阅读文件**：`m3l19_context_mgmt.py`（搜索 `maybe_compress`）

| 参数 | 值 | 含义 |
|------|-----|------|
| `COMPRESS_THRESHOLD` | 0.45 | context 占比超过 45% 触发压缩 |
| `FRESH_KEEP_TURNS` | 10 | 最近 10 轮保持原文 |
| `chunk_tokens` | 2000 | 每块约 2000 token |
| 摘要模型 | `qwen3-turbo` | 轻量模型做摘要（不用主力模型） |

**理解要点**：压缩是分块进行的——老消息按 token 数切块，每块独立摘要，摘要替换原文成为 `<context_summary>` 系统消息。最近 10 轮的原文始终保留，确保 Agent 对"刚才发生了什么"有完整记忆。

---

#### 第四步：看 Session 持久化

**阅读文件**：`m3l19_context_mgmt.py`（搜索 `save_session_ctx` 和 `append_session_raw`）

| 文件 | 格式 | 写入方式 | 用途 |
|------|------|---------|------|
| `{session}_ctx.json` | JSON 数组 | 覆写 | Session 恢复（Agent 看到连续上下文） |
| `{session}_raw.jsonl` | JSONL（每行+ts） | 追加 | 审计/调试（完整历史，不丢失） |

**理解要点**：双文件策略解决了一个矛盾——ctx.json 保存压缩后的"当前状态"（用于恢复），raw.jsonl 保存未压缩的"完整历史"（用于事后分析）。两者互不干扰。

---

#### 第五步：看 @before_llm_call Hook

**阅读文件**：`m3l19_context_mgmt.py`（搜索 `before_llm_hook`）

| 调用时机 | 行为 |
|---------|------|
| 首次调用 | 从 ctx.json 恢复历史 + 追加当前 user 消息 |
| 每次调用 | 保存消息引用 → 剪枝 → 压缩 |
| 返回值 | `None`（继续执行 LLM） |

**理解要点**：为什么不用 `@after_llm_call`？因为 CrewAI 的 `_setup_after_llm_call_hooks` 会对 answer 调用 `str()`，破坏 `isinstance(answer, list)` 检查，导致工具执行失败。这是一个框架限制。

---

#### 第六步：看三轮演示

**阅读文件**：`m3l19_context_mgmt.py`（搜索 `main`）

| 轮次 | 任务 | 验证点 |
|------|------|--------|
| 第1轮 | 调研极客时间多智能体课程 | Bootstrap + 工具调用 + 持久化 |
| 第2轮 | 总结刚才的调研结论 | Session 恢复（能引用第1轮结果） |
| 第3轮 | 生成本周周报 | 跨轮上下文连续性 |

---

#### 第七步：运行测试

```bash
cd /path/to/crewai_mas_demo
python3 -m pytest m3l19/test_context_mgmt.py -v
```

23 个测试覆盖五大模块：Bootstrap（4）、剪枝（6）、分块（4）、持久化（5）、压缩（4）。

---

### 学习检查清单

- [ ] Bootstrap 的四个文件分别注入什么？（soul=身份、user=用户画像、agent=SOP、memory=记忆索引）
- [ ] 为什么 memory.md 限制 200 行？（防止记忆索引膨胀占满 context window）
- [ ] 剪枝为什么保留 `tool_call_id`？（LLM API 要求 tool 消息必须有对应的 tool_call_id，否则报错）
- [ ] 压缩阈值 0.45 意味着什么？（context 使用超过模型窗口的 45% 就触发压缩）
- [ ] ctx.json 和 raw.jsonl 的区别？（前者覆写保存压缩状态用于恢复，后者追加保存完整历史用于审计）
- [ ] 为什么不用 `@after_llm_call`？（CrewAI 框架会 `str()` answer，破坏工具执行的类型检查）
