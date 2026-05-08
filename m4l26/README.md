# 第26课：任务链与信息传递

本课演示**通用数字员工框架**在多角色协作中的应用：Manager 和 PM 通过邮箱（mailbox Skill）实现任务链。

> **核心教学点**：三态状态机（unread→in_progress→done）、路径引用传递、崩溃恢复（reset_stale）

---

## 目录结构

```
m4l26/
├── main.py                       # Manager 入口（新项目分配 / 验收）
├── start_pm.py                   # PM 入口（检查邮箱 + 完成任务）
├── tools/
│   ├── mailbox_ops.py            # Python API（用于测试，与 CLI 同逻辑）
│   └── workspace_ops.py          # create_workspace（幂等初始化）
├── sandbox-docker-compose.yaml   # 双沙盒（Manager:8025, PM:8026）
├── test_m4l26.py                 # 单元测试（含邮箱状态机 + 工作区 + Skill结构）
├── test_m4l26_integration.py     # E2E 集成测试
├── conftest.py                   # pytest fixtures（shared_dir 临时目录）
├── demo_input/
│   └── project_requirement.md   # 项目需求输入
└── workspace/
    ├── manager/                  # Manager workspace
    │   ├── soul.md               # 身份与决策偏好
    │   ├── agent.md              # 工作规范（两个场景）
    │   ├── user.md               # 甲方画像
    │   └── skills/               # workspace-local Skill Manifest
    │       ├── load_skills.yaml  # Skill 清单
    │       ├── mailbox/          # mailbox Skill
    │       ├── init_project/     # init_project Skill
    │       └── memory-save/      # memory-save Skill
    ├── pm/                       # PM workspace（同结构）
    │   ├── soul.md / agent.md / user.md
    │   └── skills/
    │       ├── mailbox/ product_design/ memory-save/
    └── shared/                   # 运行时生成（由 init_project Skill 创建）
        ├── needs/requirements.md
        ├── design/product_spec.md
        └── mailboxes/{manager,pm}.json
```

---

## 架构：v3 vs v2

| 维度 | v2（旧） | v3（当前） |
|------|---------|-----------|
| 编排 | `run_demo.py` 编排器，顺序调用三个 Crew | 无编排器，Manager / PM 各自独立启动 |
| 类 | ManagerAssignCrew + ManagerReviewCrew + PMExecuteCrew | DigitalWorkerCrew × 2 实例（同一个类） |
| 身份 | 硬编码 role/goal | 身份来自 `workspace/{role}/*.md` |
| 通信 | 直接操作 mailbox_tools.py | mailbox Skill（CLI in Docker sandbox） |
| Skills | 全局 skills 目录 | workspace-local `skills/` + 全局脚本分层 |

---

## 快速开始

### 步骤 0：清理环境（支持重跑）

```bash
cd /path/to/crewai_mas_demo/m4l26

# 重置邮箱（保留目录结构）
echo '[]' > workspace/shared/mailboxes/manager.json
echo '[]' > workspace/shared/mailboxes/pm.json

# 清理运行时产出
rm -f workspace/shared/design/product_spec.md
rm -f workspace/shared/needs/requirements.md
rm -f workspace/shared/WORKSPACE_RULES.md
rm -f workspace/manager/review_result.md

# 清理 session 历史和日志
rm -rf workspace/manager/sessions workspace/pm/sessions
rm -f agent.log agent.log.wf
```

### 步骤 1：启动沙盒

```bash
cd /path/to/crewai_mas_demo/m4l26

docker compose -f sandbox-docker-compose.yaml --profile manager --profile pm up -d

# 等待沙盒就绪（返回 200 即可）
curl -s -o /dev/null -w '%{http_code}' http://localhost:8025/   # Manager → 200
curl -s -o /dev/null -w '%{http_code}' http://localhost:8026/   # PM → 200
```

### 步骤 2：Manager 启动（分配任务）

```bash
python3 main.py
```

Manager 执行：
1. 初始化共享工作区（`/mnt/shared/needs/` + `/mnt/shared/design/` + mailboxes）
2. 将需求写入 `/mnt/shared/needs/requirements.md`
3. 给 PM 发 `task_assign` 邮件（只传路径引用）

### 步骤 3：PM 启动（执行任务）

```bash
python3 start_pm.py
```

PM 执行：
1. 读取自己的邮箱，发现 `task_assign`
2. 读取 `/mnt/shared/needs/requirements.md`
3. 按 `product_design` Skill 规范撰写产品规格文档
4. 写入 `/mnt/shared/design/product_spec.md`
5. 给 Manager 发 `task_done` 邮件
6. 标记原消息为 done

### 步骤 4：Manager 验收

```bash
python3 main.py
```

Manager 执行：
1. 读取邮箱，发现 PM 的 `task_done`
2. 对照需求验收产品文档
3. 写入验收报告 `workspace/manager/review_result.md`
4. 标记消息为 done

### 步骤 5：检查产出

```bash
# 产品规格文档（PM 产出）
cat workspace/shared/design/product_spec.md

# 验收报告（Manager 产出）
cat workspace/manager/review_result.md
```

### 步骤 6：运行测试

```bash
# 单元测试（不需要沙盒/API，61 tests）
python3 -m pytest test_m4l26.py -v

# E2E 集成测试（需要沙盒 + LLM API）
python3 -m pytest test_m4l26_integration.py -v -s
```

### 步骤 7：停止沙盒

```bash
docker compose -f sandbox-docker-compose.yaml --profile manager --profile pm down
```

---

## 三态状态机

```
send_mail → [unread]
read_inbox → [in_progress]   ← 防止重复取走
mark_done → [done]

reset_stale: [in_progress] →[unread]  (超时恢复，崩溃容错)
```

| 操作 | CLI 命令 | Python API |
|------|---------|-----------|
| 发邮件 | `mailbox_cli.py send --to pm ...` | `send_mail(mailboxes_dir, to, ...)` |
| 读邮件 | `mailbox_cli.py read --role pm` | `read_inbox(mailboxes_dir, role)` |
| 标记完成 | `mailbox_cli.py done --role pm --msg-id x` | `mark_done(mailboxes_dir, role, msg_id)` |
| 崩溃恢复 | `mailbox_cli.py reset-stale --role pm` | `reset_stale(mailboxes_dir, role, timeout)` |

---

## 邮件内容规范（核心教学点）

**只传路径引用，不传文档内容**

| ✅ 正确 | ❌ 错误 |
|--------|--------|
| 「请读 /mnt/shared/needs/requirements.md」 | 把 requirements.md 全文复制进邮件 |
| 「产品文档已写入 /mnt/shared/design/product_spec.md」 | 把 product_spec.md 全文放进 task_done |

---

## 常见问题

**Q：Manager 发完邮件后 PM 没读到？**
→ 检查 `workspace/shared/mailboxes/pm.json` 中是否有 `status="unread"` 的消息。

**Q：想清空重来？**
→ 执行步骤 0 的清理命令，然后从步骤 2 重新开始。

**Q：停止沙盒？**
```bash
docker compose -f sandbox-docker-compose.yaml --profile manager --profile pm down
```

**Q：模型切换？**
→ 修改 `main.py` 和 `start_pm.py` 中的 `model` 参数（默认 `glm-5.1`）。

---

## 课堂代码演示学习指南

本节帮你按课程教学顺序阅读代码，建立完整的理解链路。

### 整体架构一览

```
┌────────────────────────────────────────────────────────────────┐
│  main.py（Manager）           start_pm.py（PM）                │
│  自动判断模式：分配 or 验收    固定模式：检查邮箱 + 执行任务     │
└───────────┬──────────────────────────┬─────────────────────────┘
            │                          │
            ▼                          ▼
┌────────────────────────────────────────────────────────────────┐
│  DigitalWorkerCrew（通用框架，25课同一个类）                     │
│  角色身份来自 workspace/{role}/*.md                             │
└───────────┬──────────────────────────┬─────────────────────────┘
            │                          │
            ▼                          ▼
┌─────────────────────┐    ┌─────────────────────┐
│ workspace/manager/  │    │ workspace/pm/       │
│ skills/mailbox/     │    │ skills/mailbox/     │
│ skills/init_project/│    │ skills/product_     │
│                     │    │       design/       │
└─────────┬───────────┘    └──────────┬──────────┘
          │        通过邮箱通信         │
          └──────────┬─────────────────┘
                     ▼
          ┌─────────────────────┐
          │ workspace/shared/   │
          │ mailboxes/*.json    │ ← 三态状态机
          │ needs/*.md          │ ← Manager 写
          │ design/*.md         │ ← PM 写
          └─────────────────────┘
```

### 学习路线（建议按顺序阅读）

---

#### 第一步：理解三态状态机——邮箱的核心

**对应课文**：第三节"三态邮件状态机"

**阅读文件**：`tools/mailbox_ops.py`

| 重点区域 | 看什么 |
|---------|--------|
| `send_mail()` | 邮件结构：`id`, `type`, `from_`, `to`, `subject`, `body`, `status="unread"` |
| `read_inbox()` | 原子操作：读取时立刻标记为 `in_progress` + 记录 `processing_since` 时间戳 |
| `mark_done()` | 处理完成后标记为 `done` |
| `reset_stale()` | 崩溃恢复：超时的 `in_progress` 消息重置为 `unread` |
| `FileLock` 使用 | 每次读写都在锁内完成，防止并发冲突 |

**理解要点**：为什么不用简单的 `read: bool`？因为 Agent 可能在读取后、处理完成前崩溃。`in_progress` + `processing_since` 实现了类似 AWS SQS Visibility Timeout 的模式，崩溃后可以通过 `reset_stale()` 恢复。

**验证**：`python3 -m pytest test_m4l26.py -v -k "TestThreeState"` 看完整状态转换测试。

---

#### 第二步：理解共享工作区——状态如何共享

**对应课文**：第二节"共享工作区"

**阅读文件**：`tools/workspace_ops.py`

| 重点区域 | 看什么 |
|---------|--------|
| `create_workspace()` | 创建三个目录：`needs/`（Manager 写）、`design/`（PM 写）、`mailboxes/`（双方读写） |
| `WORKSPACE_RULES.md` 生成 | 每个目录有单一 Owner，权限规则写入文件供 Agent 读取 |
| 幂等性 | 目录已存在则跳过，不会覆盖 |

**理解要点**：权限边界不是硬安全限制，而是"收窄 Agent 的注意力"——让它知道该往哪写，不该动哪里。

---

#### 第三步：看 Agent 的工作规范如何驱动协作

**对应课文**：第四节"路径引用传递"

**3a. `workspace/manager/agent.md`**

找到两个场景：
- **场景1（新项目）**：init_project → 写需求文档 → 发 `task_assign` 邮件给 PM（只传路径引用！）
- **场景2（验收）**：读邮箱 → 读 PM 产出 → 写验收报告

**3b. `workspace/pm/agent.md`**

找到 7 步工作流：读邮箱 → 读需求 → 加载 product_design Skill → 写产品文档 → 发 `task_done` → 标记消息完成

**理解要点**：邮件内容只传路径（"请读 /mnt/shared/needs/requirements.md"），不传文档全文。这是核心设计原则——文件在共享工作区，邮件只是触发器。

---

#### 第四步：理解沙盒挂载——两个 Agent 如何共享文件系统

**对应课文**：第二节"状态共享"

**阅读文件**：`sandbox-docker-compose.yaml`

| 容器 | 端口 | 挂载 |
|------|------|------|
| Manager | 8025 | `workspace/manager` → `/workspace`，`workspace/shared` → `/mnt/shared` |
| PM | 8026 | `workspace/pm` → `/workspace`，`workspace/shared` → `/mnt/shared` |

**理解要点**：两个沙盒各有私有的 `/workspace`（互不可见），但共享同一个 `/mnt/shared`。Manager 写到 `/mnt/shared/needs/`，PM 从 `/mnt/shared/needs/` 读取。

---

#### 第五步：看入口如何串联三步任务链

**对应课文**：第五节"端到端演示"

**阅读文件**：`main.py`

| 重点区域 | 看什么 |
|---------|--------|
| `_has_pending_task_done()` | 检查 `manager.json` 是否有未处理的 `task_done` 消息 |
| 模式自动切换 | 有 `task_done` → 验收模式；没有 → 分配模式 |
| `_build_user_request()` | 根据模式构建不同的 prompt |

**理解要点**：同一个 `main.py` 跑两次，行为完全不同——第一次分配任务，第二次验收交付。切换依据是文件系统状态（邮箱内容），不是命令行参数或 LLM 自我判断。

---

#### 第六步：验证——跑测试看效果

```bash
cd /path/to/crewai_mas_demo

# 1. 单元测试（61个，不需要沙盒/API）
python3 -m pytest m4l26/test_m4l26.py -v

# 2. 重点关注这些测试：
#    - TestThreeStateMachine：完整状态转换 unread→in_progress→done + reset_stale
#    - TestSendMailFields：邮件结构字段验证
#    - TestReadInboxAtomicity：读取的原子性（读的同时标记 in_progress）
#    - TestConcurrentWriteSafety：FileLock 并发安全
```

---

### 学习检查清单

完成以上六步后，你应该能回答：

- [ ] 为什么邮箱要用三态（unread/in_progress/done）而不是二态（unread/done）？
- [ ] `read_inbox()` 为什么在读取的同时就标记为 `in_progress`？（防止另一个进程重复取走）
- [ ] `reset_stale()` 解决什么问题？（Agent 崩溃后消息不会永远丢失）
- [ ] 邮件内容为什么只传路径引用而不传文档全文？
- [ ] `main.py` 如何实现"第一次分配、第二次验收"的模式切换？（检查邮箱状态，不靠 LLM）
- [ ] 两个沙盒如何共享文件？（Docker volume 挂载同一个 `workspace/shared` 目录）
