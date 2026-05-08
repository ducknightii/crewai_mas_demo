# 第20课：文件系统记忆——自主写入与技能创建

本课演示 Agent 通过沙盒 Sub-Crew 自主管理记忆：写入用户偏好（memory-save）、从 SOP 创建可复用技能（skill-creator）、审计清理记忆与技能（memory-governance）。

> **核心教学点**：memory-save / skill-creator / memory-governance 三个 Skill、沙盒文件隔离（Docker 挂载）、Bootstrap 导航骨架模式、零直接文件工具

---

## 目录结构

```
m3l20/
├── m3l20_file_memory.py           # 主演示：三个 Skill 的端到端流程
├── sandbox-docker-compose.yaml    # Sandbox Docker 配置（含挂载）
├── workspace/
│   ├── soul.md                    # Agent 身份定义
│   ├── user.md                    # 用户画像
│   ├── agent.md                   # Agent 规则（禁止直接文件工具）
│   ├── memory.md                  # 记忆导航索引
│   └── sessions/
│       ├── demo_m3l20_ctx.json    # 压缩上下文快照
│       └── demo_m3l20_raw.jsonl   # 追加式完整历史
└── agent.log                      # 运行日志
```

---

## 快速开始

```bash
# 1. 启动 Sandbox（带挂载）
cd /path/to/crewai_mas_demo/m3l20
docker compose -f sandbox-docker-compose.yaml up -d

# 2. 运行演示
cd /path/to/crewai_mas_demo
python3 m3l20/m3l20_file_memory.py
```

---

## 课堂代码演示学习指南

### 整体架构一览

```
┌────────────────────────────────────┐
│  Main Agent（XiaoPaw）              │
│  tools: [SkillLoaderTool]           │
│  直接文件工具: ❌ 无                  │
│                                    │
│  Bootstrap 骨架:                    │
│  <soul> + <user> + <agent> + <mem> │
└──────────┬─────────────────────────┘
           │ SkillLoaderTool
           │
     ┌─────┴──────┬─────────────────┐
     ▼            ▼                 ▼
  memory-save  skill-creator  memory-governance
     │            │                 │
     ▼            ▼                 ▼
  Sub-Crew     Sub-Crew          Sub-Crew
     │            │                 │
     ▼            ▼                 ▼
  Sandbox      Sandbox           Sandbox
     │            │                 │
     ▼            ▼                 ▼
  写 workspace/  写 ../skills/    读+审计+清理
  (user.md等)   (SKILL.md)      (两个目录)

Docker 挂载：
  ./workspace → /workspace:rw
  ../skills   → /mnt/skills:rw
```

### 学习路线

---

#### 第一步：理解"零直接文件工具"设计

**阅读文件**：`m3l20_file_memory.py`（搜索 `assistant_agent`）

| 对比 | m3l19（上一课） | m3l20（本课） |
|------|----------------|--------------|
| 文件工具 | FileWriterTool, FileReadTool | ❌ 无 |
| 文件操作方式 | Agent 直接读写 | 通过 SkillLoaderTool → Sub-Crew → Sandbox |
| 安全模型 | Hook 拦截路径 | Docker 挂载隔离 |

**理解要点**：Main Agent 没有任何直接文件操作能力——所有文件读写都委托给沙盒中的 Sub-Crew。这比 Hook 拦截更安全，因为 Agent 根本没有文件工具可以被 prompt injection 利用。

---

#### 第二步：看 Docker 挂载配置

**阅读文件**：`sandbox-docker-compose.yaml` + `m3l20_file_memory.py`（搜索 `M3L20_SANDBOX_MOUNT_DESC`）

| 宿主机路径 | 容器内路径 | 权限 | 用途 |
|-----------|-----------|------|------|
| `./workspace` | `/workspace` | rw | memory-save 写入用户偏好 |
| `../skills` | `/mnt/skills` | rw | skill-creator 创建新 SKILL.md |

**理解要点**：Docker 挂载是安全边界——Sub-Crew 只能访问这两个目录，不能访问宿主机的其他文件。`rw` 权限确保可以写入。

---

#### 第三步：看三个 Skill 的分工

**阅读文件**：`m3l20_file_memory.py`（搜索三轮 `kickoff`）

| Skill | 触发场景 | 输入 | 输出 |
|-------|---------|------|------|
| `memory-save` | 用户说出偏好/事实 | 偏好内容 + 目标文件 | 更新 workspace/ 下的文件 |
| `skill-creator` | 用户描述重复性 SOP | SOP 描述 + 技能名 | 创建 skills/{name}/SKILL.md |
| `memory-governance` | 记忆/技能积累过多 | 审计范围 | 治理报告 + 清理操作 |

**理解要点**：三个 Skill 形成记忆的"写入-固化-治理"闭环——memory-save 写入碎片信息，skill-creator 将重复模式固化为可复用技能，memory-governance 定期审计防止膨胀。

---

#### 第四步：看 Bootstrap 与上下文管理

**阅读文件**：`m3l20_file_memory.py`（搜索 `build_bootstrap_prompt`、`prune_tool_results`、`maybe_compress`）

本课复用了 m3l19 的全部上下文管理机制：

| 机制 | 功能 | 与 m3l19 的差异 |
|------|------|----------------|
| Bootstrap | 四文件骨架注入 | 相同 |
| 剪枝 | 工具结果替换为 [已剪枝] | 相同 |
| 压缩 | 超阈值分块摘要 | 相同 |
| Session 持久化 | ctx.json + raw.jsonl | 相同 |

**理解要点**：上下文管理是基础设施——一旦建好就复用。本课的新增价值全在三个 Skill 和沙盒隔离模型上。

---

#### 第五步：看 agent.md 中的规则约束

**阅读文件**：`workspace/agent.md`

| 规则 | 作用 |
|------|------|
| 禁止直接文件工具 | 所有文件操作必须通过 skill_loader |
| 记忆治理触发条件 | memory.md 超过 150 行 / 发现死链 / 技能重叠 |

**理解要点**：agent.md 中的规则是"提示级"约束——靠 LLM 遵守。真正的安全保障是 Docker 挂载隔离（Main Agent 根本没有文件工具可用）。

---

### 学习检查清单

- [ ] 本课为什么移除了 FileWriterTool 和 FileReadTool？（安全：沙盒隔离比 Hook 拦截更彻底）
- [ ] memory-save 和 skill-creator 的区别？（前者写入碎片偏好，后者固化重复 SOP 为可复用技能）
- [ ] Docker 挂载的两个目录分别给哪个 Skill 使用？（workspace → memory-save，skills → skill-creator）
- [ ] memory-governance 在什么情况下触发？（记忆索引超 150 行、发现死链、技能目录重叠）
- [ ] Bootstrap 骨架中 memory.md 为什么限制 200 行？（它是导航索引不是内容存储，超出会浪费 context）
