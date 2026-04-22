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
