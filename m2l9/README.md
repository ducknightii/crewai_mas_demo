# 第14课：MCP 协议——标准化工具接口

本课演示如何通过 MCP（Model Context Protocol）协议将外部工具服务标准化接入 Agent，实现工具过滤、多租户支持和工具缓存。

> **核心教学点**：`MCPServerHTTP` 配置、`create_static_tool_filter` 工具白名单、多租户 Header 注入、`cache_tools_list` 缓存优化

---

## 目录结构

```
m2l9/
├── m2l9_mcp.py    # MCP 协议集成演示
└── agent.log      # 运行日志
```

---

## 快速开始

```bash
# 先启动 MCP Server（需要另一个终端）
# MCP Server 代码不在本目录，需根据课程说明部署

cd /path/to/crewai_mas_demo
python3 m2l9/m2l9_mcp.py
```

---

## 课堂代码演示学习指南

### 整体架构一览

```
┌─────────────────────────────────────────────┐
│  CrewAI Agent（email_agent）                  │
│                                             │
│  mcps = [MCPServerHTTP(                     │
│    url = "http://localhost:8005/mcp",        │
│    headers = {                              │
│      "Authorization": "Bearer qqkkk",       │
│      "X-User-Id": "user01"    ← 多租户      │
│    },                                       │
│    tool_filter = static_filter  ← 白名单     │
│  )]                                         │
└───────────────┬─────────────────────────────┘
                │  HTTP (streamable)
                ▼
┌─────────────────────────────────────────────┐
│  MCP Server（localhost:8005）                 │
│                                             │
│  暴露的工具（可能有很多）：                      │
│  ✅ send_email       ← 白名单内              │
│  ✅ get_mail_list    ← 白名单内              │
│  ✅ get_mail_detail  ← 白名单内              │
│  ❌ delete_email     ← 被 filter 屏蔽       │
│  ❌ admin_tools      ← 被 filter 屏蔽       │
└─────────────────────────────────────────────┘
```

### 学习路线

---

#### 第一步：看 MCP 三种传输模式

**阅读文件**：`m2l9_mcp.py`（30-34 行）

| 类 | 传输方式 | 适用场景 |
|----|---------|---------|
| `MCPServerStdio` | 子进程 stdin/stdout | 本地工具、单机部署 |
| `MCPServerHTTP` | HTTP（支持 streaming） | **本课重点**：网络服务、多租户 |
| `MCPServerSSE` | Server-Sent Events | 实时推送场景 |

**理解要点**：三种传输模式都导入了，但本课只使用 `MCPServerHTTP`。CrewAI 的 MCP 实现兼容任何符合 MCP 协议的服务端。

---

#### 第二步：看工具过滤器

**阅读文件**：`m2l9_mcp.py`（45-47 行）

```python
static_filter = create_static_tool_filter(
    allowed_tool_names=["send_email", "get_mail_list", "get_mail_detail"]
)
```

**理解要点**：这是安全控制的核心——即使 MCP Server 暴露了删除、管理等危险工具，Agent 也只能使用白名单中的三个工具。代码还导入了 `create_dynamic_tool_filter`（运行时条件过滤）和 `ToolFilterContext`，但本课未使用。

---

#### 第三步：看 MCPServerHTTP 完整配置

**阅读文件**：`m2l9_mcp.py`（62-91 行）

| 参数 | 值 | 作用 |
|------|-----|------|
| `url` | `"http://localhost:8005/mcp"` | MCP 服务端地址 |
| `headers.Authorization` | `"Bearer qqkkk"` | 身份认证 |
| `headers.X-User-Id` | `"user01"` | 多租户标识 |
| `streamable` | `True` | 启用流式响应 |
| `cache_tools_list` | `True` | 缓存工具发现结果 |
| `tool_filter` | `static_filter` | 应用白名单过滤 |

**理解要点**：`headers` 中的 `X-User-Id` 实现了多租户——同一个 MCP Server 为多个用户服务，每个请求通过 Header 区分用户身份，Server 据此返回该用户的数据。

---

#### 第四步：看 Task 编排

**阅读文件**：`m2l9_mcp.py`（98-120 行）

| Task | 工具 | 作用 |
|------|------|------|
| `send_email_task` | `send_email` | 发送一封关于 MCP 邮件服务的邮件 |
| `get_email_task` | `get_mail_list` + `get_mail_detail` | 查收件箱 → 获取第一封邮件详情 |

**理解要点**：两个 Task 按 `Process.sequential` 顺序执行：先发邮件，再查收件箱。Agent 的工具完全来自 MCP Server，本地不定义任何工具。

---

### 学习检查清单

- [ ] MCP 协议解决了什么问题？（外部工具服务标准化接入，不需要在 Python 代码中定义工具）
- [ ] `create_static_tool_filter` 和 `create_dynamic_tool_filter` 的区别？（前者是静态白名单，后者支持运行时条件判断）
- [ ] `cache_tools_list=True` 优化了什么？（避免每次工具调用都向 MCP Server 重新请求工具列表）
- [ ] 多租户如何实现？（通过 HTTP Header `X-User-Id` 传递用户标识，MCP Server 据此隔离数据）
- [ ] Agent 的 `mcps` 参数接受什么？（MCP Server 实例列表，一个 Agent 可以连接多个 MCP Server）
