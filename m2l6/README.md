# 第10课：多模态模型——让你的 Agent 拥有"眼睛"

本课演示如何构建能"看图"的 Agent：通过自定义 AddImageToolLocal 加载本地图片，配合多模态视觉模型（qwen3-vl-plus）进行结构化分析。

> **核心教学点**：多模态 Agent 配置、图片处理流水线（本地文件→Base64→视觉模型）、AddImageToolLocal vs 内置 AddImageTool、双模型架构（文本+视觉）

---

## 目录结构

```
m2l6/
├── m2l6_agent.py                  # 多模态视觉分析 Agent 演示
├── 20260202161329_150_6.jpg       # 示例图片（三文鱼谷物碗美食照）
└── agent.log                      # 运行日志
```

---

## 快速开始

```bash
cd /path/to/crewai_mas_demo
python3 m2l6/m2l6_agent.py
```

---

## 课堂代码演示学习指南

### 整体架构一览

```
本地图片文件
   │
   ▼
AddImageToolLocal
   │  读取字节 → Base64 编码 → data:image/...;base64,...
   ▼
AliyunLLM._normalize_multimodal_tool_result()
   │  检测 Base64 → 重构为多模态 user message
   │  自动切换 model → image_model
   ▼
qwen3-vl-plus（视觉模型）
   │  分析图片内容
   ▼
ImageAnalysis（Pydantic 结构化输出）
   │  subject_description / atmosphere_vibe / visual_details
   │  image_quality_score / highlight_feature
   ▼
格式化打印
```

### 学习路线

---

#### 第一步：看 Pydantic 输出契约

**阅读文件**：`m2l6_agent.py`（48-75 行）

| 字段 | 类型 | 作用 |
|------|------|------|
| `subject_description` | str | 画面核心内容描述 |
| `atmosphere_vibe` | str | 风格/氛围形容词 |
| `visual_details` | List[str] | ≥3 个容易被忽略的细节 |
| `image_quality_score` | float | 1-10 分（构图30%+光影30%+清晰20%+色彩20%） |
| `highlight_feature` | str | Visual Hook——第一眼抓住视线的元素 |

**理解要点**：视觉模型的自由分析通过 Pydantic 契约约束为结构化数据，每个字段有明确的评分维度和最低数量要求。

---

#### 第二步：理解 Agent 的"假多模态"配置

**阅读文件**：`m2l6_agent.py`（104-148 行）

| 参数 | 值 | 为什么 |
|------|-----|--------|
| `multimodal` | `False` | 不使用 CrewAI 内置多模态管线 |
| `tools` | `[AddImageToolLocal()]` | 图片通过 Tool 注入，不走内置通道 |
| `image_model` | `"qwen3-vl-plus"` | AliyunLLM 自动切换的视觉模型 |
| `model` | `"qwen-plus"` | 文本任务使用的基础模型 |

**理解要点**：`multimodal=False` 是关键——使用 `AddImageToolLocal` 时不能开启 CrewAI 的内置多模态。图片处理完全由自定义 LLM 层的消息归一化完成。

---

#### 第三步：看图片处理工具

**阅读文件**：`tools/add_image_tool_local.py`

| 方法 | 作用 |
|------|------|
| `_local_path_to_base64_data_and_compress_url()` | 读取本地文件 → Base64 编码 → data URL |
| `_compress_image()` | PIL 缩放超过 4K 的图片 |
| `_run()` | 判断输入是本地路径还是 HTTP URL |

**理解要点**：这是自定义 Tool 的典型案例——输入 schema 与 CrewAI 内置 `AddImageTool` 保持一致（`image_url: str`），但实现了本地文件读取能力。

---

#### 第四步：看消息归一化（核心机制）

**阅读文件**：`llm/aliyun_llm.py`（搜索 `_normalize_multimodal_tool_result`）

| 场景 | 处理方式 |
|------|---------|
| Function Calling 模式 | Base64 出现在 `role=tool` 消息中 → 重构为 `user` 多模态消息 |
| ReAct 模式 | Base64 出现在 `role=assistant` 消息中 → 同样重构 |
| 模型切换 | 检测到图片 → payload 中 `model` 字段从 `qwen-plus` 切换为 `qwen3-vl-plus` |

**理解要点**：这一层是"桥梁"——CrewAI 的 Tool 输出格式和 DashScope 的多模态 API 格式不兼容，消息归一化负责在两者之间做翻译。

---

#### 第五步：看 Task 描述中的图片路径注入

**阅读文件**：`m2l6_agent.py`（82-93 行 + 159-214 行）

```python
image_path = Path(__file__).resolve().parent / "20260202161329_150_6.jpg"
```

**理解要点**：图片路径在 Task 描述中直接嵌入，Agent 会将路径传给 `AddImageToolLocal`。路径解析使用 `Path(__file__).resolve().parent` 确保无论从哪里运行都能找到图片。

---

### 学习检查清单

- [ ] 为什么 `multimodal=False` 而不是 `True`？（使用 AddImageToolLocal 时，多模态处理由 AliyunLLM 消息归一化层完成，不走 CrewAI 内置管线）
- [ ] 双模型架构如何工作？（文本任务用 qwen-plus，检测到图片自动切换为 qwen3-vl-plus）
- [ ] `AddImageToolLocal` 和内置 `AddImageTool` 的区别？（前者支持本地文件路径，后者只支持 URL）
- [ ] 消息归一化处理了哪两种场景？（Function Calling 模式和 ReAct 模式的 Base64 数据）
- [ ] `ImageAnalysis` 的 `image_quality_score` 评分维度？（构图30% + 光影30% + 清晰度20% + 色彩20%）
