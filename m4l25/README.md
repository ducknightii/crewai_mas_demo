# 第25课：团队角色体系——分工设计与行为规范

本课演示**通用数字员工框架**（`DigitalWorkerCrew`）：同一个类，换 workspace 就变成不同角色。
Manager 和 Dev 各自独立运行，互不通信（通信在第26课实现）。

> **核心教学点**：代码层面零角色特异性。角色身份完全由 workspace 下的四个文件决定。

---

## 目录结构

```
m4l25/
├── run_manager.py              # Manager 启动器（~30行，零角色逻辑）
├── run_dev.py                  # Dev 启动器（同上）
├── sandbox-docker-compose.yaml # 双沙盒配置
├── test_m4l25.py               # 集成测试（63个用例）
├── demo_input/
│   ├── project_requirement.md  # Manager 输入
│   └── feature_requirement.md  # Dev 输入
├── workspace/
│   ├── manager/                # Manager 的 workspace（soul/agent/user/memory）
│   └── dev/                    # Dev 的 workspace
```

---

## 快速开始

### 1. 清理环境（支持重跑）

```bash
cd /path/to/crewai_mas_demo/m4l25

# 清理 session 历史（可选，重跑时建议执行）
rm -rf workspace/manager/sessions workspace/dev/sessions

# 清理上次产出（可选）
rm -f workspace/manager/task_breakdown.md workspace/dev/tech_design.md
```

### 2. 启动沙盒

```bash
cd /path/to/crewai_mas_demo/m4l25

# 按需启动（--profile manager | dev | 同时两个）
docker compose -f sandbox-docker-compose.yaml --profile manager up -d
docker compose -f sandbox-docker-compose.yaml --profile dev up -d

# 或同时启动两个
docker compose -f sandbox-docker-compose.yaml --profile manager --profile dev up -d

# 验证（healthy 状态才可用）
docker ps | grep sandbox
curl -s http://localhost:8023/mcp | head -1  # Manager
curl -s http://localhost:8024/mcp | head -1  # Dev
```

### 3. 运行演示

从 `crewai_mas_demo/` 目录执行：

```bash
cd /path/to/crewai_mas_demo

# 演示 1：Manager — 任务拆解
python3 m4l25/run_manager.py 1>m4l25/agent_manager.log 2>m4l25/agent_manager.log.wf

# 演示 2：Dev — 技术设计
python3 m4l25/run_dev.py 1>m4l25/agent_dev.log 2>m4l25/agent_dev.log.wf
```

### 4. 运行测试

```bash
cd /path/to/crewai_mas_demo

# 离线测试（62个离线 + 1个 e2e，不需要沙盒/API）
python3 -m pytest m4l25/test_m4l25.py -v -m "not e2e"

# E2E 测试（需要沙盒 + LLM API）
python3 -m pytest m4l25/test_m4l25.py -v -m e2e -s
```

---

## 演示说明

### 演示 1：Manager

| 项目 | 说明 |
|------|------|
| 入口 | `run_manager.py` |
| Workspace | `workspace/manager/`（soul + agent含团队名册 + user + memory） |
| Skills | `workspace/manager/skills/`（sop_manager + memory-save，workspace 内置） |
| 沙盒端口 | 8023 |
| 输出 | `workspace/manager/task_breakdown.md` |

**观察重点**：Manager 的 NEVER 清单阻止它做技术设计，agent.md 中的团队名册让它知道把任务分给谁。

### 演示 2：Dev

| 项目 | 说明 |
|------|------|
| 入口 | `run_dev.py` |
| Workspace | `workspace/dev/`（soul + agent含职责边界 + user + memory） |
| Skills | `workspace/dev/skills/`（sop_dev + memory-save，workspace 内置） |
| 沙盒端口 | 8024 |
| 输出 | `workspace/dev/tech_design.md` |

**观察重点**：Dev 的 soul.md 声明「技术是唯一权威」，如果输入缺验收标准，触发 NEVER 规则退回。

---

## 常见问题

**Q：报 `ConnectionError`？**
→ 检查沙盒：`docker ps | grep sandbox`

**Q：报 `ModuleNotFoundError`？**
→ 确认从 `crewai_mas_demo/` 目录运行。

**Q：想清空重来？**
```bash
rm -rf workspace/manager/sessions workspace/dev/sessions
rm -f workspace/manager/task_breakdown.md workspace/dev/tech_design.md
```

**Q：停止沙盒？**
```bash
docker compose -f sandbox-docker-compose.yaml down
```
