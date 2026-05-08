# 第7课：定义 Agent——从"提示词工程"到"人设工程"

本课深入 Agent 定义，用一个小红书增长策略专家演示如何通过精心设计的 Backstory 塑造 Agent 行为。

> **核心教学点**：人设工程（Persona Engineering）、Backstory 即行为约束、`Agent.kickoff()` 直接交互

---

## 目录结构

```
m2l3/
└── m2l3_agent.py    # Agent 人设工程演示
```

---

## 快速开始

```bash
cd /path/to/crewai_mas_demo
python3 m2l3/m2l3_agent.py
```

---

## 课堂代码演示学习指南

### 学习路线

**阅读文件**：`m2l3_agent.py`（73 行）

---

#### 第一步：看 Backstory 的深度设计

| Backstory 组成 | 作用 |
|---------------|------|
| CES 评分算法 | 注入领域知识（评论8分 > 关注 > 收藏 > 点赞） |
| 反漏斗模型 | 注入方法论（先打最精准人群） |
| 语义工程 SOP | 注入工作流程（爆款标题设计步骤） |
| 行为边界 | 约束行为（只输出策略，绝不写最终文案） |

**理解要点**：Backstory 不是"背景介绍"，而是行为控制器——领域知识、方法论、SOP、禁止规则全部通过 Backstory 注入。

---

#### 第二步：看 Agent.kickoff() 直接交互

```python
content_strategist.kickoff(messages=[{"role": "user", "content": "..."}])
```

**理解要点**：不需要创建 Task 和 Crew，可以直接和 Agent 对话。适合快速测试人设效果。

---

### 学习检查清单

- [ ] Backstory 在 Agent 行为中扮演什么角色？（行为控制器，不是简单介绍）
- [ ] "人设工程"和"提示词工程"的区别是什么？（人设工程是系统性的角色塑造，提示词工程是单次指令优化）
- [ ] `Agent.kickoff()` 和 `Crew.kickoff()` 的区别？（前者直接交互，后者走完整 Task 流程）
