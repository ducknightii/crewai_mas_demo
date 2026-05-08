# 第6课：工欲善其器——基础代码环境准备

本课演示如何配置 LLM 接口，是后续所有课程的环境基础。

> **核心教学点**：LLM 类配置（OpenAI 兼容接口）、直接调用 vs Agent 调用

---

## 目录结构

```
m2l2/
└── m2l2_llm_openai.py    # LLM 配置参考
```

---

## 课堂代码演示学习指南

### 学习路线

**阅读文件**：`m2l2_llm_openai.py`（57 行）

| 重点区域 | 看什么 |
|---------|--------|
| `LLM()` 构造 | `model`, `api_key`, `base_url` 三个核心参数 |
| `llm.call()` | 直接调用 LLM——传 messages 列表，返回文本 |
| `Agent(llm=llm)` | 将 LLM 传给 Agent，通过 `agent.execute_task()` 间接调用 |

**理解要点**：CrewAI 的 `LLM` 类兼容所有 OpenAI 格式的 API（DeepSeek、Kimi、阿里云通义千问等），只需换 `base_url`。课程主要使用阿里云 DashScope（`llm/aliyun_llm.py`），本文件是配置参考。

### 学习检查清单

- [ ] `LLM(model, api_key, base_url)` 三个参数分别是什么？
- [ ] 直接调用 `llm.call()` 和通过 Agent 调用有什么区别？
- [ ] 如何切换到不同的 LLM 提供商？（换 `base_url` + `api_key`）
