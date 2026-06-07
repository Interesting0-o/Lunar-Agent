# Lunar — 带有计算心理学状态机的 AI 角色扮演引擎

基于 LangGraph 构建的 AI 伴侣系统，以《崩坏3》角色「月下誓约·予爱以心」为原型。

**核心理念**：不是用 prompt 咒语驱动角色人格，而是用可计算的多层心理状态机来模拟角色的内在情感变化。

## 架构

```
用户输入 → Perception Node → State Engine → LLM Node → 回复
              │                    │
        社交信号解析          多层状态更新
        (9维 + 4维)         (Surface / Internal
                              Hidden / Relationship)
```

### 四层心理状态模型

| 层 | 字段 | 含义 |
|---|------|------|
| **Surface** | expressiveness, warmth, sharpness... | 用户能看到的情绪表现 |
| **Internal** | energy, stress, loneliness, longing... | 角色真正感受到的情绪 |
| **Hidden** | suppressed_sadness, hidden_affection | 被压抑、不表达的情感 |
| **Relationship** | affection, trust, romantic_tension... | 角色对用户的关系评估 |

**关键机制**：内部状态 ≠ 表面表达。高 pride 角色会压抑好感，高 attachment 角色会放大被抛弃的恐惧——这就是"口是心非"的计算实现。

### 数据流

```
用户输入
  ↓
Perception Node → SocialSignals (affection/attention/intimacy/...) ×9
                → InteractionImpact (emotional_weight/memorability/...) ×4
  ↓
State Engine    → 基于权重规则表 + 特质修饰器更新 4 层状态
                → 检查隐藏情感是否突破阈值 → 触发剧情事件
  ↓
LLM Node        → 注入角色 system prompt，生成回复
```

## 技术栈

| 组件 | 技术 |
|------|------|
| 图编排 | LangGraph |
| 模型调用 | LangChain + Ollama |
| 本地模型 | qwen2.5:7b |
| 状态持久化 | SQLite (LangGraph Checkpointer) |
| Python | 3.13 |

## 项目结构

```
Lunar/
├── agent.py              # LangGraph 图定义与编译
├── nodes.py              # 四个图节点（inject_system / perception / state_engine / llm）
├── perception.py         # 感知层：社交信号提取 + 验证 + 重试
├── state_engine.py       # 状态引擎：权重规则表 + 特质修饰器 + 压抑/突破机制
├── state.py              # 状态类型定义（TypedDict）
├── model.py              # LLM 模型初始化
├── config.py             # 运行时配置（重试策略等）
├── character_prompt.py   # 角色人设 SYSTEM_PROMPT
├── perception_prompt.py  # 感知层提示词
├── main.py               # 入口
└── db/                   # SQLite 持久化
```

## 快速开始

```bash
# 1. 安装依赖
pip install langgraph langchain langchain-ollama

# 2. 拉取模型（确保 Ollama 已运行）
ollama pull qwen2.5:7b

# 3. 运行
python agent.py
```

## 已知局限 & 改进路线

| 优先级 | 问题 | 方案 |
|--------|------|------|
| P0 | SurfaceState 未注入 LLM prompt | `_compute_surface()` 结果动态写入 system prompt |
| P0 | 突破事件无实际效果 | 事件触发时修改 prompt 模板或切换回复策略 |
| P1 | 硬编码权重无法学习 | 权重表外部化为 JSON 参数文件，支持用户反馈微调 |
| P1 | 上下文窗口仅 4 条 | 扩展至 10~20 + 接入 Chroma 向量检索 |
| P1 | Checkpointer 未集成 | `graph.compile(checkpointer=saver)` |
| P2 | 无用户心理模型 | 增加 UserProfile 状态层 |

## License

MIT
