# 第12-13课：工具设计哲学——Hook 拦截与安全控制

本课演示如何用 CrewAI 的 Hook 机制（`@before_tool_call`）拦截工具调用，结合 `contextvars` 实现多租户工作空间隔离和路径穿越防护。

> **核心教学点**：`@before_tool_call` Hook、`contextvars` 上下文隔离、路径穿越（Path Traversal）防御、多租户文件系统隔离

---

## 目录结构

```
m2l8/
├── m2l8_tools_call.py             # 主演示：Hook 拦截 + 路径安全
├── m2l8_context.py                # contextvars 上下文变量定义
├── workspace/
│   └── 1234567890/
│       └── CRONTAB.md             # Agent 产出的定时任务文件
└── agent.log                      # 运行日志
```

---

## 快速开始

```bash
cd /path/to/crewai_mas_demo
python3 m2l8/m2l8_tools_call.py
```

---

## 课堂代码演示学习指南

### 整体架构一览

```
用户请求
   │
   ▼
user_id.set("1234567890")     ← contextvars 设置当前用户
   │
   ▼
Crew.kickoff()
   │
   ▼
Agent 决定调用 FileWriterTool / FileReadTool
   │
   ▼
@before_tool_call（file_path_hook）
   │
   ├─ 1. 检查 tool_name 是否在白名单
   ├─ 2. 从 contextvars 获取 user_id
   ├─ 3. 构建用户工作空间：./workspace/{user_id}/
   ├─ 4. 路径穿越检测（Path.relative_to）
   │     ├─ 路径在工作空间内 → 放行（return None）
   │     ├─ 路径在工作空间外但可重定向 → 拼接后放行
   │     └─ 路径穿越攻击 → 阻断（return False）
   │
   ▼
Tool 执行（路径已被安全重写）
   │
   ▼
./workspace/1234567890/CRONTAB.md
```

### 学习路线

---

#### 第一步：看上下文变量定义

**阅读文件**：`m2l8_context.py`（全文 30 行）

| 变量 | 类型 | 用途 |
|------|------|------|
| `user_id` | `ContextVar[Optional[str]]` | 标识当前用户，用于工作空间隔离 |
| `request_id` | `ContextVar[Optional[str]]` | 请求链路追踪（本课未使用） |
| `task_id` | `ContextVar[Optional[str]]` | 任务标识（本课未使用） |

**理解要点**：`ContextVar` 是 Python 标准库 `contextvars` 的核心类——线程安全、支持异步，不需要通过函数参数传递。在 Hook 中通过 `user_id.get()` 直接获取，避免了在 Agent→Task→Tool 链路中层层传参。

---

#### 第二步：看 Hook 拦截函数

**阅读文件**：`m2l8_tools_call.py`（36-100 行）

| 逻辑段 | 行号 | 作用 |
|--------|------|------|
| 白名单检查 | 38-39 | 只拦截文件读写工具，其他工具直接放行 |
| 用户身份验证 | 42-45 | `user_id.get()` 为 None → 阻断并返回错误 |
| 工作空间创建 | 47-49 | 自动创建 `./workspace/{uid}/` |
| 路径提取 | 51-56 | 根据工具名提取不同字段（`file_path` vs `filename`） |
| 路径安全校验 | 60-80 | 两阶段检查：先 resolve 后 relative_to |

**理解要点**：Hook 的返回值控制工具执行流程——`return None` 表示放行（继续执行工具），`return False` 表示阻断（使用 `context.tool_result` 作为替代结果）。

---

#### 第三步：理解路径穿越防御

**阅读文件**：`m2l8_tools_call.py`（60-80 行）

```
阶段一：直接解析                    阶段二：拼接后解析
Path("../../etc/passwd").resolve()   Path("workspace/123/../../etc/passwd").resolve()
        │                                    │
        ▼                                    ▼
   /etc/passwd                          /etc/passwd
        │                                    │
   relative_to(workspace/123/)          relative_to(workspace/123/)
        │                                    │
   ValueError! → 进入阶段二              ValueError! → 阻断！
```

**理解要点**：`Path.relative_to()` 是 Python 标准库提供的路径安全校验方法。如果 A 不是 B 的子路径，会抛出 `ValueError`。两阶段检查确保无论 Agent 给出什么路径，都不会逃出用户工作空间。

---

#### 第四步：看三轮对话演示

**阅读文件**：`m2l8_tools_call.py`（120-227 行）

| 轮次 | 用户输入 | Agent 行为 | 文件操作 |
|------|---------|-----------|---------|
| 第1轮 | 创建定时任务 | 写入 CRONTAB.md | FileWriterTool → Hook 重定向到 workspace/1234567890/ |
| 第2轮 | 查询定时任务 | 读取 CRONTAB.md | FileReadTool → Hook 重定向 |
| 第3轮 | 修改任务时间 | 更新 CRONTAB.md | FileWriterTool → Hook 重定向 |

**理解要点**：Agent 只操作文件名（如 `CRONTAB.md`），Hook 透明地将所有路径重写到用户隔离的工作空间。Agent 完全不感知路径重写的存在。

---

### 学习检查清单

- [ ] `@before_tool_call` 的返回值有几种？（三种：`None` 放行、`False` 阻断、不返回等同 None）
- [ ] `context.tool_result` 在什么时候使用？（`return False` 阻断工具时，用它替代真实的工具返回值）
- [ ] `ContextVar` 相比全局变量的优势？（线程安全、异步安全、不需要函数参数传递）
- [ ] 为什么需要两阶段路径检查？（Agent 可能给出绝对路径或相对路径，两阶段确保都能正确处理）
- [ ] 如果 Agent 尝试写入 `../../etc/passwd`，会发生什么？（两阶段 relative_to 都失败 → Hook 返回 False → 工具不执行）
